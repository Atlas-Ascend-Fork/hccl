/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <iterator>
#include <utility>
#include "channel.h"
#include "channel_request.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_launch_dl.h"

#include "ccu_temp_all_to_all_mesh1d_2Die.h"
#include "ccu_kernel_all_to_all_mesh2die.h"
#include "ccu_temp_all_to_all_mesh_1D.h"
#include "ccu_kernel_all_to_all_mesh1d.h"
#include "template_utils.h"

namespace ops_hccl {
constexpr int DEFAULT_PORT_NUM = 8;

std::vector<CostModelParam> CcuTempAllToAllMesh1D2Die::CalcCostCoeff(CalcCostCoeffParam param)
{
    // CLOS-only(如960 POD CLOS双平面): 无板内mesh链路，所有对端均经CLOS(两die平面)可达
    bool isClosOnly = (param.topoInfo != nullptr && param.topoInfo->level0Topo == Level0Shape::CLOS);
    // 2Die场景: 板内MESH（同Die）和跨Die CLOS（8端口）并行执行，取max(A)
    // 板内卡数 = 每个module的device数（2Die一般为4或8）
    u32 intraDieRankSize = (param.topoInfo != nullptr) ? param.topoInfo->deviceNumPerModule : param.rankSize;
    if (isClosOnly) {
        intraDieRankSize = 0; // CLOS-only无mesh路径
    } else if (intraDieRankSize == 0 || intraDieRankSize >= param.rankSize) {
        intraDieRankSize = param.rankSize / 2; // fallback: 对称分成两半
    }
    u32 interDieRankSize = param.rankSize - intraDieRankSize; // 另一个Die的对端数
    if (isClosOnly) {
        interDieRankSize = param.rankSize - 1; // CLOS-only: 全部对端走CLOS
    }

    int meshPortNum = 1; // MESH: portNum不参与公式
    // CLOS跨Die: 端口数由 executor 通过 topomatch v2 动态传入，直接聚合使用
    int closPortNum = 0;
    for (auto p : param.portNum) {
        closPortNum += static_cast<int>(p);
    }
    if (closPortNum <= 0) {
        closPortNum = DEFAULT_PORT_NUM; // fallback
    }
    // CLOS公式: A = n*(groupSize-1)/(portNum*bw)，groupSize含自己，所以 +1
    u32 closGroupSize = interDieRankSize;

    float A_mesh = 0.0f;
    float A_clos = 0.0f;
    float B = 0.0f; // CCU无需本地拷贝
    float C = 0.0f;
    float D = 0.0f;

    // 板内MESH: 同Die内intraDieRankSize个rank
    if (!isClosOnly) {
        CostModelManager::Global()->CalcMeshParam(
            param.dataRatio, CommTopo::COMM_TOPO_1DMESH, meshPortNum, intraDieRankSize, A_mesh, param.isPod);
    }
    // 板外CLOS: 跨Die到interDieRankSize个对端，closPortNum端口共享
    CostModelManager::Global()->CalcMeshParam(
        param.dataRatio, CommTopo::COMM_TOPO_CLOS, closPortNum, closGroupSize, A_clos, param.isPod);

    // 两条路径并行，A取max
    float A = std::max(A_mesh, A_clos);
    if (param.isPod) { // pod场景ccu带宽更高
        A *= 0.8;
    }
    // kernelNum为两路并行(mesh+CLOS)折算的CCU任务数估计值, 非实际kernel下发数
    int kernelNum = 8;
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::CCU, C);
    CostModelManager::Global()->CalcLaunchParams(0, EngineType::CCU, D);

    HCCL_INFO(
        "[CcuTempAllToAllMesh1D2Die::CalcCostCoeff] rankSize=%u intraDie=%u interDie=%u dataRatio=%f "
        "A_mesh=%e A_clos=%e A=%e B=%e C=%e kernelNum=%d meshPortNum=%d closPortNum=%d",
        param.rankSize, intraDieRankSize, interDieRankSize, param.dataRatio, A_mesh, A_clos, A, B, C, kernelNum,
        meshPortNum, closPortNum);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

CcuTempAllToAllMesh1D2Die::CcuTempAllToAllMesh1D2Die(
    const OpParam& param, RankId rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    templateRankSize_ = subCommRanks[0].size();
    myRank_ = CalcRankIdxInSubComm(rankId, subCommRanks, myRank_);
}

CcuTempAllToAllMesh1D2Die::~CcuTempAllToAllMesh1D2Die() {}

HcclResult CcuTempAllToAllMesh1D2Die::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_PRT_RET(
        subCommRanks_.size() != 1 || subCommRanks_[0].empty(),
        HCCL_ERROR(
            "[CcuTempAllToAllMesh1D2Die][CalcRes] Invalid subCommRanks[%zu] or subCommRanks empty.",
            subCommRanks_.size()),
        HcclResult::HCCL_E_INTERNAL);
    HCCL_DEBUG(
        "[CcuTempAllToAllMesh1D2Die][CalcRes] rankSize[%u] subCommRanks0[%zu].", templateRankSize_,
        subCommRanks_[0].size());

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));
    HCCL_INFO("[CcuTempAllToAllMesh1D2Die][CalcRes] channelDescs size[%zu]", channelDescs.size());

    CHK_RET(PartitionChannels(comm, rankIdToChannelDesc_));
    double ratio = 1.0;
    CHK_RET(CalcDieSplitRatio(
        comm, myRank_, is2Plus6_, kernelChannels_[KERNEL_CLOS_MAJOR], kernelChannels_[KERNEL_CLOS_MINOR], ratio));
    // 纯CLOS(如960 POD CLOS双平面): 无板内mesh链路, FULLMESH槽位无远端channel、仅剩本地拷贝,
    // 不再单独占用一个kernel; 将本地拷贝随数据切分分别并入两个CLOS kernel
    // (major/minor各自拷贝本slice对应的本地数据段, kernel侧GroupCopy的goSize即CalGoSize(本slice)),
    // 落为两kernel: CLOS_MAJOR(含本地拷贝) + CLOS_MINOR(含本地拷贝)。
    if (is2Plus6_ && kernelCount_ == MAX_KERNEL_NUM_2DIE && kernelChannels_[KERNEL_FULLMESH].empty()
        && !kernelChannels_[KERNEL_CLOS_MAJOR].empty() && !kernelChannels_[KERNEL_CLOS_MINOR].empty()) {
        kernelChannels_[KERNEL_FULLMESH] = std::move(kernelChannels_[KERNEL_CLOS_MAJOR]);
        kernelRankGroup_[KERNEL_FULLMESH] = std::move(kernelRankGroup_[KERNEL_CLOS_MAJOR]);
        kernelRankGroup_[KERNEL_FULLMESH].push_back(myRank_);
        kernelChannels_[KERNEL_CLOS_MAJOR] = std::move(kernelChannels_[KERNEL_CLOS_MINOR]);
        kernelRankGroup_[KERNEL_CLOS_MAJOR] = std::move(kernelRankGroup_[KERNEL_CLOS_MINOR]);
        kernelRankGroup_[KERNEL_CLOS_MAJOR].push_back(myRank_);
        kernelChannels_[KERNEL_CLOS_MINOR].clear();
        kernelRankGroup_[KERNEL_CLOS_MINOR].clear();
        kernelCount_ = MAX_KERNEL_NUM_2DIE - 1;
        kernelWithMyRank_ = {true, true, false};
        isClosOnly_ = true;
        HCCL_INFO(
            "[CcuTempAllToAllMesh1D2Die][CalcRes] CLOS-only topology, merge local copy into CLOS kernels, "
            "kernelCount=2 (major+minor), dieSplitRatio=%f.",
            ratio);
    }
    resourceRequest.dieSplitRatio = ratio;
    uint32_t slaveThreadNum = kernelCount_ - 1;
    resourceRequest.notifyNumOnMainThread = slaveThreadNum;
    resourceRequest.slaveThreadNum = slaveThreadNum;
    resourceRequest.notifyNumPerThread.assign(slaveThreadNum, 1);
    resourceRequest.channels.emplace_back(channelDescs);
    HCCL_INFO("[CcuTempAllToAllMesh1D2Die][CalcRes] resourceRequest.channels[%zu]", resourceRequest.channels.size());
    resourceRequest.ccuKernelNum.push_back(kernelCount_);

    for (uint32_t i = 0; i < kernelCount_; i++) {
        CcuKernelInfo kernelInfo;
        CHK_SAFETY_FUNC_RET(
            strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllToAllMesh2DieKernel"));
        kernelInfo.kernelFunc = reinterpret_cast<void*>(CcuAllToAllMesh2DieKernel);
        auto kernelArg = std::make_shared<CcuKernelArgAllToAllMesh2Die>();
        kernelArg->rankId = myRank_;
        kernelArg->opParam = param;
        kernelArg->subCommRanks = subCommRanks_;
        kernelArg->withMyRank = kernelWithMyRank_[i];
        kernelArg->rankGroup = kernelRankGroup_[i];
        kernelArg->peerBatchSize
            = (topoInfo != nullptr && topoInfo->isPod) ? PEER_BATCH_SIZE_POD : PEER_BATCH_SIZE_SERVER;
        kernelInfo.setKernelArg(kernelArg);
        kernelInfo.channels = kernelChannels_[i];
        resourceRequest.ccuKernelInfos.emplace_back(kernelInfo);
        HCCL_INFO("[CcuTempAllToAllMesh1D2Die][CalcRes] kernel[%u], peerBatchSize=%u", i, kernelArg->peerBatchSize);
        HCCL_DEBUG(
            "[CcuTempAllToAllMesh1D2Die][CalcRes] kernel[%u], channels=%zu, withMyRank=%u, ccuKernelInfos=%zu", i,
            kernelChannels_[i].size(), kernelWithMyRank_[i], resourceRequest.ccuKernelInfos.size());
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::PartitionChannels(
    HcclComm comm, std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc)
{
    std::map<uint32_t, std::vector<HcclChannelDesc>> singleChByDie, multiChByDie;
    CHK_RET(SplitChannelsByDie(comm, myRank_, rankIdToChannelDesc, singleChByDie, multiChByDie, is2Plus6_));
    CHK_RET(PartitionChannelsFor2Die(
        comm, singleChByDie, multiChByDie, is2Plus6_, myRank_, kernelCount_, fullmeshDieId_, kernelChannels_,
        kernelRankGroup_, "CcuTempAllToAllMesh1D2Die"));
    // 防御：2Die场景下仅一个die存在多通道链路时（如非对称子通信域），空CLOS槽位退化为
    // 单CLOS kernel(不切分数据)；kernelCount降为2，后续线程/notify均按2计算。
    if (is2Plus6_ && kernelCount_ == MAX_KERNEL_NUM_2DIE
        && (kernelChannels_[KERNEL_CLOS_MAJOR].empty() != kernelChannels_[KERNEL_CLOS_MINOR].empty())) {
        uint32_t nonEmpty = kernelChannels_[KERNEL_CLOS_MAJOR].empty() ? KERNEL_CLOS_MINOR : KERNEL_CLOS_MAJOR;
        if (nonEmpty != KERNEL_CLOS_MAJOR) {
            kernelChannels_[KERNEL_CLOS_MAJOR] = std::move(kernelChannels_[KERNEL_CLOS_MINOR]);
            kernelRankGroup_[KERNEL_CLOS_MAJOR] = std::move(kernelRankGroup_[KERNEL_CLOS_MINOR]);
        }
        kernelChannels_[KERNEL_CLOS_MINOR].clear();
        kernelRankGroup_[KERNEL_CLOS_MINOR].clear();
        kernelCount_ = MAX_KERNEL_NUM_2DIE - 1;
        is2Plus6_ = false; // 单CLOS路径无需按die切分数据
        HCCL_INFO("[CcuTempAllToAllMesh1D2Die][PartitionChannels] only one die has multi-channel links, "
                  "degenerate to single CLOS kernel, no data split.");
    }
    CHK_RET(ReorderAndLogKernelGroups(
        myRank_, static_cast<uint32_t>(subCommRanks_[0].size()), is2Plus6_, kernelCount_, kernelChannels_,
        kernelRankGroup_, "CcuTempAllToAllMesh1D2Die"));
    return HcclResult::HCCL_SUCCESS;
}

// 按dieSplitRatio计算major kernel的切分大小，并以splitAlignUnit_(数据类型粒度)对齐，
// 保证切分点不落在数据元素中间；major+minor恒等于full，两侧均为0或>=对齐粒度。
uint64_t CcuTempAllToAllMesh1D2Die::CalcMajorSliceSize(uint64_t full) const
{
    if (full == 0 || splitAlignUnit_ == 0 || splitAlignUnit_ >= full) {
        // 数据不足一个对齐粒度时整体由major承担(major端口更多)，minor为0由kernel侧跳过
        return full;
    }
    // 先四舍五入到最近整数再对齐到元素边界，带宽分配更均衡
    uint64_t ideal = static_cast<uint64_t>(static_cast<double>(full) * dieSplitRatio_ + 0.5);
    uint64_t aligned = (ideal + splitAlignUnit_ / 2) / splitAlignUnit_ * splitAlignUnit_;
    return std::min(std::max(aligned, splitAlignUnit_), full);
}

HcclResult CcuTempAllToAllMesh1D2Die::CalcFillArgsInfo(uint32_t kernelIdx, uint64_t& sliceSize, uint64_t& sliceOffset)
{
    const uint64_t full = sliceSize;
    if (isClosOnly_) {
        // CLOS-only两kernel: 槽0=CLOS_MAJOR(含本地拷贝), 槽1=CLOS_MINOR(含本地拷贝);
        // 本地拷贝的GroupCopy goSize=CalGoSize(本slice), 与远端写同一份数据切片
        if (kernelIdx == KERNEL_FULLMESH) {
            sliceSize = CalcMajorSliceSize(full);
            sliceOffset = 0;
        } else {
            uint64_t majorSize = CalcMajorSliceSize(full);
            sliceSize = full - majorSize;
            sliceOffset = majorSize;
        }
    } else if (kernelIdx == KERNEL_FULLMESH) {
        sliceSize = full;
        sliceOffset = 0;
    } else if (is2Plus6_ && kernelIdx == KERNEL_CLOS_MAJOR) {
        sliceSize = CalcMajorSliceSize(full);
        sliceOffset = 0;
    } else if (is2Plus6_ && kernelIdx == KERNEL_CLOS_MINOR) {
        uint64_t majorSize = CalcMajorSliceSize(full);
        sliceSize = full - majorSize;
        sliceOffset = majorSize;
    } else {
        sliceSize = full;
        sliceOffset = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::LaunchKernels(
    uint32_t kernelCount, uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t sliceStride,
    const LoopGroupConfig& config, const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    for (uint32_t i = 0; i < kernelCount; i++) {
        uint64_t sliceSize = templateDataParams.sliceSize;
        uint64_t sliceOffset = 0;
        CHK_RET(CalcFillArgsInfo(i, sliceSize, sliceOffset));
        auto goSize = CalGoSize(sliceSize, config, GetCcuVersion());
        std::vector<uint64_t> taskArgs
            = {inputAddr + sliceOffset, outputAddr, token, sliceSize, sliceStride, sliceStride * myRank_ + sliceOffset};
        for (auto val : goSize) {
            taskArgs.push_back(val);
        }
        HCCL_INFO(
            "[CcuTempAllToAllMesh1D2Die][KernelRun] kernel[%u] sliceSize[%llu] sliceOffset[%llu] "
            "rankGroupSize[%zu] withMyRank[%u]",
            i, sliceSize, sliceOffset, kernelRankGroup_[i].size(), kernelWithMyRank_[i]);
        CcuResult launchRet = HcommCcuKernelLaunch(
            templateResource.threads[i], templateResource.ccuKernels[i], taskArgs.data(), taskArgs.size());
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllToAllMesh1D2Die][KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
        CcuKernelSubmitInfo submitInfo;
        CHK_RET(FillCachedArgs(
            submitInfo, taskArgs[0], taskArgs[1], taskArgs[2], taskArgs[3], taskArgs[4], taskArgs[5], taskArgs[6],
            taskArgs[7], taskArgs[8], taskArgs[9], buffInfo_.outBuffBaseOff));
        submitInfo.cachedArgs[11] = buffInfo_.inBuffBaseOff + sliceOffset;
        submitInfo.kernelHandle = templateResource.ccuKernels[i];
        templateResource.submitInfos.push_back(submitInfo);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::KernelRun(
    const OpParam& param, const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempAllToAllMesh1D2Die] Run");
    opMode_ = param.opMode;
    buffInfo_ = templateDataParams.buffInfo;
    CHK_PRT_RET(
        subCommRanks_.empty() || subCommRanks_[0].empty(),
        HCCL_ERROR("[CcuTempAllToAllMesh1D2Die][KernelRun] subCommRanks empty."), HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        buffInfo_.inputPtr == nullptr || buffInfo_.outputPtr == nullptr,
        HCCL_ERROR(
            "[CcuTempAllToAllMesh1D2Die][KernelRun] Rank[%d] input[%#llx] or output[%#llx] is null", myRank_,
            buffInfo_.inputPtr, buffInfo_.outputPtr),
        HcclResult::HCCL_E_PTR);

    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    // rankSize为1时无sdispls[1]，退化为count粒度stride（仅本地拷贝路径可达）
    u64 strideDisplsIdx = (templateDataParams.sdispls.size() > 1) ? 1 : 0;
    uint64_t outputSliceStride
        = templateDataParams.sdispls[strideDisplsIdx] * DATATYPE_SIZE_TABLE[param.all2AllDataDes.recvType]
          - buffInfo_.inBuffBaseOff;
    // 切分对齐粒度: send/recv类型大小的较大值(数据类型字节数均为2的幂, 较大值即最小公倍数)
    splitAlignUnit_ = std::max(
        DATATYPE_SIZE_TABLE[param.all2AllDataDes.sendType], DATATYPE_SIZE_TABLE[param.all2AllDataDes.recvType]);
    splitAlignUnit_ = std::max<uint64_t>(splitAlignUnit_, 1);
    HCCL_INFO(
        "[CcuTempAllToAllMesh1D2Die][KernelRun] begin. Rank[%d], input[%#llx/%#llx], output[%#llx/%#llx]", myRank_,
        inputAddr, param.inputPtr, outputAddr, param.outputPtr);
    uint32_t kernelCount = templateResource.ccuKernels.size();
    is2Plus6_ = (kernelCount == MAX_KERNEL_NUM_2DIE);
    if (templateResource.dieSplitRatio > 0.0) {
        dieSplitRatio_ = templateResource.dieSplitRatio;
    }
    // CLOS-only两kernel模式判定(运行时结构推断, 与is2Plus6_的kernelCount推断同一模式):
    // kernelCount==2 且 dieSplitRatio为有效切分比例(非1.0)。闭环论证:
    // CalcDieSplitRatio仅在is2Plus6且MAJOR/MINOR两槽非空时按bwCoeff计算, 其余恒为1.0;
    // CLOS-only的CalcRes compaction恰好使kernelCount降为2且两槽非空(ratio有效);
    // 非2Plus6场景(950 normal)与退化场景(is2Plus6被置false)均为kernelCount==2且ratio==1.0。
    isClosOnly_ = (kernelCount == MAX_KERNEL_NUM_2DIE - 1 && dieSplitRatio_ > 0.0 && dieSplitRatio_ < 1.0);
    if (isClosOnly_) {
        kernelWithMyRank_ = {true, true, false};
        HCCL_INFO(
            "[CcuTempAllToAllMesh1D2Die][KernelRun] CLOS-only mode, 2 kernels with local copy merged, "
            "dieSplitRatio=%f.",
            dieSplitRatio_);
    }

    uint32_t subThreadCount = kernelCount - 1;
    std::vector<ThreadHandle> subThreads(
        templateResource.threads.begin() + 1, templateResource.threads.begin() + 1 + subThreadCount);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, std::vector<u32>(subThreadCount, 0)));

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
    CHK_RET(LaunchKernels(
        kernelCount, inputAddr, outputAddr, token, outputSliceStride, config, templateDataParams, templateResource));

    std::vector<u32> notifyIdxSubToMain(subThreadCount);
    for (uint32_t i = 0; i < subThreadCount; i++) {
        notifyIdxSubToMain[i] = i;
    }
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    HCCL_DEBUG("[CcuTempAlltoAllMesh1D2Die][KernelRun] end. Rank[%d]", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllMesh1D2Die::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    (void)param;
    constexpr u32 argInIdx = 0;
    constexpr u32 argOutIdx = 1;
    constexpr u32 metaOutBaseOffIdx = 10;
    constexpr u32 metaInCombineOffIdx = 11;
    constexpr u32 taskArgSize = 10;

    u32 kernelCount = static_cast<u32>(tempFastLaunchCtx.ccuKernelSubmitInfos.size());
    if (kernelCount == 0) {
        HCCL_INFO("[CcuTempAllToAllMesh1D2Die][FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempAllToAllMesh1D2Die][FastLaunch] start, kernelCount[%u]", kernelCount);

    uint64_t inputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr);
    uint64_t outputAddr = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr);

    if (kernelCount > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxMainToSub(kernelCount - 1, 0);
        CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxMainToSub));
    }

    for (u32 i = 0; i < kernelCount; i++) {
        uint64_t* args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[i].cachedArgs);
        args[argInIdx] = inputAddr + args[metaInCombineOffIdx];
        args[argOutIdx] = outputAddr + args[metaOutBaseOffIdx];

        CcuResult launchRet = HcommCcuKernelLaunch(
            tempFastLaunchCtx.threads[i], tempFastLaunchCtx.ccuKernelSubmitInfos[i].kernelHandle,
            reinterpret_cast<void*>(args), taskArgSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllToAllMesh1D2Die][FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    if (kernelCount > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxSubToMain(kernelCount - 1);
        for (u32 i = 0; i < kernelCount - 1; i++) {
            notifyIdxSubToMain[i] = i;
        }
        CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxSubToMain));
    }
    HCCL_DEBUG("[CcuTempAllToAllMesh1D2Die][FastLaunch] end");
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
