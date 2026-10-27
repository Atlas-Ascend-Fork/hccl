/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_hd.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {
constexpr int POD_TASK_NUM_OVERHEAD = 5;
// D 项经验常数：冻结 spec ar-hd-aicpu-20261016 §8（与 InsTempAllGatherNHR 实测经验值对齐，
// pairwise a73c637f 同款命名常量先例）；HD 为 preferred-only（仅显式配置可选，不参与默认成本
// 竞争），常数值仅影响 estimate 输出，未标定不作为性能结论
constexpr auto HD_TASK_LAUNCH_COEFF = 1.2e-6f; // D 项：taskNum 个 task 的下发时延

std::vector<CostModelParam> InsTempAllGatherHD::CalcCostCoeff(CalcCostCoeffParam param)
{
    // HD 调度要求 rank 数为 2 的整数次幂（主门控在 TopoMatchTwoLevel），非 2^N 不参与选路
    if (param.rankSize == 0 || (param.rankSize & (param.rankSize - 1)) != 0) {
        return {};
    }
    CommTopo netType = param.netType;
    int portNum = (param.netType == CommTopo::COMM_TOPO_CLOS && param.portNum.size() >= 2) ?
                      (param.portNum[0] + param.portNum[1]) :
                      param.portNum[0];
    int log2R = 0;
    for (u32 r = param.rankSize; r > 1; r >>= 1) {
        log2R++;
    }
    int kernelNum = 8 * log2R - 7;
    kernelNum = std::max(kernelNum, 10);
    u32 rEff = static_cast<u32>(5 * log2R / 3);
    rEff = std::max(rEff, 2u);
    int taskNum = CostModelManager::CalcTransTaskNum(rEff) + CostModelManager::CalcSyncTaskNum(rEff) * 2;
    taskNum = (!param.isPod) ? taskNum : taskNum + POD_TASK_NUM_OVERHEAD;

    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;
    CostModelManager::Global()->CalcNHRParams(param.dataRatio, netType, portNum, param.rankSize, A, param.isPod);
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B);
    }

    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AICPU, C);
    D = HD_TASK_LAUNCH_COEFF * taskNum;
    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

InsTempAllGatherHD::InsTempAllGatherHD(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempAllGatherHD::~InsTempAllGatherHD() {}

HcclResult InsTempAllGatherHD::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> level1Channels;
    bool isUBX = topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix;
    if (isUBX) {
        CHK_RET(CalcChannelRequestNhrMultiJettyUbx(comm, param, topoInfo, subCommRanks_, level1Channels));
    } else {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, myChannelDescs));
        for (auto channel : myChannelDescs) {
            if (channel.channelProtocol != CommProtocol::COMM_PROTOCOL_ROCE) {
                level1Channels.push_back(channel);
            }
        }
    }
    resourceRequest.channels.push_back(level1Channels);
    // 非 UBX 分支过滤 ROCE 通道后可能为空：CalcChannelsPerRankMin 空输入返回 1，
    // 会静默空跑（无数据搬运却"成功"返回），此处显式拦截
    CHK_PRT_RET(
        level1Channels.empty(), HCCL_ERROR("[InsTempAllGatherHD][CalcRes] level1Channels is empty."),
        HcclResult::HCCL_E_INTERNAL);
    channelsPerRank_ = CalcChannelsPerRankMin(level1Channels);
    if (isUBX && channelsPerRank_ > MAX_JETTY_NUM) {
        HCCL_ERROR(
            "[InsTempAllGatherHD][CalcRes] channelsPerRank_ %u is greater than MAX_JETTY_NUM %u.", channelsPerRank_,
            MAX_JETTY_NUM);
    } else {
        HCCL_DEBUG("[InsTempAllGatherHD][CalcRes] channelsPerRank_ is %u.", channelsPerRank_);
    }
    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherHD::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 threadNum = GetThreadNum();
    CHK_PRT_RET(
        threadNum == 0, HCCL_ERROR("[InsTempAllGatherHD][GetRes] threadNum is 0, channelsPerRank_ invalid."),
        HcclResult::HCCL_E_INTERNAL);
    HCCL_INFO("[InsTempAllGatherHD][GetRes] threadNum[%u]", threadNum);
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherHD::GetThreadNum() const { return channelsPerRank_; }

HcclResult InsTempAllGatherHD::SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    CHK_PRT_RET(channels.empty(), HCCL_ERROR("[SetchannelsPerRank] channels is empty."), HCCL_E_INTERNAL);
    channelsPerRank_ = CalcChannelsPerRankMin(channels);
    HCCL_INFO("[InsTempAllGatherHD][SetchannelsPerRank] channelsPerRank_[%u]", channelsPerRank_);
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherHD::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = templateRankSize_;
    return scratchMultiple;
}

HcclResult InsTempAllGatherHD::PrepareDataSplitForMultiChannel(const TemplateResource& templateResource)
{
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalDataCount = tempAlgParams_.sliceSize / dataTypeSize;
    if (templateResource.channels.empty() || templateResource.channels.begin()->second.empty()) {
        dataSplit_.assign(1, tempAlgParams_.sliceSize);
        dataOffset_.assign(1, 0);
        if (tempAlgParams_.tailSize > 0) {
            dataSplitTail_.assign(1, tempAlgParams_.tailSize);
            dataOffsetTail_.assign(1, 0);
        }
        return HCCL_SUCCESS;
    }
    std::vector<u64> elemCountOut;
    CHK_RET(CalcDataSplitByPortGroup(
        totalDataCount, dataTypeSize, templateResource.channels.begin()->second, elemCountOut, dataSplit_,
        dataOffset_));
    if (tempAlgParams_.tailSize > 0) {
        u64 totalDataCountTail = tempAlgParams_.tailSize / dataTypeSize;
        CHK_RET(CalcDataSplitByPortGroup(
            totalDataCountTail, dataTypeSize, templateResource.channels.begin()->second, elemCountOut, dataSplitTail_,
            dataOffsetTail_));
    }

    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherHD::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllGatherHD] Run start");
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        HCCL_INFO("[InsTempAllGatherHD] Rank [%u], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }
    // 防御性校验：HD 调度要求框间 rank 数为 2 的整数次幂（主门控在 TopoMatchTwoLevel）
    CHK_PRT_RET(
        (templateRankSize_ & (templateRankSize_ - 1)) != 0,
        HCCL_ERROR("[InsTempAllGatherHD][KernelRun] templateRankSize_[%u] is not a power of two.", templateRankSize_),
        HcclResult::HCCL_E_NOT_SUPPORT);
    // 防御性校验：GetRankFromMap 按 HD 步骤表下标访问 subCommRanks_[0]，rankList 数必须与模板 rank 数一致
    CHK_PRT_RET(
        subCommRanks_.size() == 0, HCCL_ERROR("[InsTempAllGatherHD][KernelRun] subCommRanks is empty."),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        subCommRanks_[0].size() != templateRankSize_,
        HCCL_ERROR(
            "[InsTempAllGatherHD][KernelRun] rank count[%zu] is invalid in rank list, templateRankSize[%u].",
            subCommRanks_[0].size(), templateRankSize_),
        HcclResult::HCCL_E_INTERNAL);
    threadNum_ = GetThreadNum();
    if (templateResource.threads.size() < threadNum_) {
        HCCL_ERROR(
            "[InsTempAllGatherHD] Rank [%u], thread num[%zu] is not as expected[%u].", myRank_,
            templateResource.threads.size(), threadNum_);
        return HcclResult::HCCL_E_INTERNAL;
    }
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    isOutputCclBuff_ = (tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER);

    bool isPcieProtocol = IsPcieProtocol(templateResource.channels); // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocol;                                     // 是否使用Read模式
    HCCL_DEBUG("[InsTempAllGatherHD] Use Dma Read[%d]", isDmaRead_);
    CHK_RET(PrepareDataSplitForMultiChannel(templateResource));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(LocalDataCopy(templateResource.threads, channelIdx)); // input buffer拷贝到scratch buffer上
        if (templateRankSize_ > 1) {
            CHK_RET(RunAllGatherHD(templateResource.threads, templateResource.channels, channelIdx));
        }
        CHK_RET(PostLocalCopy(templateResource.threads[channelIdx], channelIdx));
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[InsTempAllGatherHD] Run End");
    return HCCL_SUCCESS;
}

u32 InsTempAllGatherHD::GetRankFromMap(const u32 algRankIdx) const { return subCommRanks_[0].at(algRankIdx); }

// 计算每步的对端以及块编号：距离递增倍增，伙伴 = algRank XOR D，
// 整段发送持有区段[heldStart, heldStart+D)，接收补半区段[heldStart^D, heldStart^D+D)
HcclResult InsTempAllGatherHD::GetHDStepInfo(u32 step, AicpuNHRStepInfo& stepInfo)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    stepInfo.step = step;
    stepInfo.myRank = myAlgRank;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();

    u32 dist = 1u << step;
    u32 peerIdx = myAlgRank ^ dist;
    u32 heldStart = myAlgRank & ~(dist - 1);

    stepInfo.toRank = peerIdx;
    stepInfo.fromRank = peerIdx; // HD 收发同伙伴
    stepInfo.nSlices = dist;

    stepInfo.txSliceIdxs.reserve(dist);
    stepInfo.rxSliceIdxs.reserve(dist);
    for (u32 i = 0; i < dist; i++) {
        stepInfo.txSliceIdxs.push_back(heldStart + i);
        stepInfo.rxSliceIdxs.push_back((heldStart ^ dist) + i);

        HCCL_DEBUG(
            "[AG-HD][GetHDStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, heldStart + i, (heldStart ^ dist) + i);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherHD::LocalDataCopy(const std::vector<ThreadHandle>& threads, const u32& channelIdx)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    u64 sliceSize = tempAlgParams_.sliceSize;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 partialSliceSize = dataSplit_[channelIdx];
    u64 partialOffset = dataOffset_[channelIdx];
    // 尾块模式
    if (tempAlgParams_.tailSize != 0 && myAlgRank == templateRankSize_ - 1) {
        partialSliceSize = dataSplitTail_[channelIdx];
        partialOffset = dataOffsetTail_[channelIdx];
    }
    for (u64 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 inOff = tempAlgParams_.inputSliceStride * myAlgRank + inBaseOff + partialOffset;
        u64 sliceCount = partialSliceSize / dataTypeSize;

        if (enableRemoteMemAccess_) {
            const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
            const u64 outOff = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff + partialOffset;
            if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr && inOff == outOff) {
                continue;
            }
            DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, partialSliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOff, partialSliceSize, sliceCount);
            CHK_RET(LocalCopy(threads[channelIdx], srcSlice, dstSlice));
        } else {
            u64 scratchBaseoff = 0;
            u64 scOff = 0;
            if (isOutputCclBuff_) {
                scratchBaseoff = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
                scOff = tempAlgParams_.outputSliceStride * myAlgRank + scratchBaseoff + partialOffset;
            } else {
                const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
                scratchBaseoff = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
                scOff = tempAlgParams_.sliceSize * myAlgRank + scratchBaseoff + partialOffset;
            }
            if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.hcclBuff.addr && inOff == scOff) {
                continue;
            }
            DataSlice srcSlices(tempAlgParams_.buffInfo.inputPtr, inOff, partialSliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scOff, partialSliceSize, sliceCount);
            CHK_RET(LocalCopy(threads[channelIdx], srcSlices, dstSlice));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherHD::PostLocalCopy(const ThreadHandle& thread, const u32& channelIdx)
{
    if (enableRemoteMemAccess_) {
        HCCL_INFO("[InsTempAllGatherHD] PostLocalCopy skip because remote memory access enabled");
        return HcclResult::HCCL_SUCCESS;
    }
    if (tempAlgParams_.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempAllGatherHD] PostLocalCopy skip because output is scratch");
        return HcclResult::HCCL_SUCCESS;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    for (u64 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (u32 algRank = 0; algRank < templateRankSize_; algRank++) {
            u64 partialSliceSize = dataSplit_[channelIdx];
            u64 partialOffset = dataOffset_[channelIdx];
            // 尾块模式
            if (tempAlgParams_.tailSize != 0 && algRank == templateRankSize_ - 1) {
                partialSliceSize = dataSplitTail_[channelIdx];
                partialOffset = dataOffsetTail_[channelIdx];
            }
            u64 sliceCount = partialSliceSize / dataTypeSize;
            u64 scratchOffset = tempAlgParams_.sliceSize * algRank + scratchBase + partialOffset;
            u64 outOffset = tempAlgParams_.outputSliceStride * algRank + outBaseOff + partialOffset;
            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, partialSliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset, partialSliceSize, sliceCount);
            CHK_RET(LocalCopy(thread, srcSlice, dstSlice));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherHD::RunAllGatherHD(
    const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const u32& channelIdx)
{
    u32 nSteps = 0;
    for (u32 tmp = templateRankSize_; tmp > 1; tmp >>= 1, nSteps++) {
    }

    for (u32 step = 0; step < nSteps; ++step) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetHDStepInfo(step, stepInfo));
        u32 fromRankKey = GetRankFromMap(stepInfo.fromRank);
        u32 toRankKey = GetRankFromMap(stepInfo.toRank);
        CHK_PRT_RET(
            channels.count(fromRankKey) == 0 || channelIdx >= channels.at(fromRankKey).size()
                || channels.count(toRankKey) == 0 || channelIdx >= channels.at(toRankKey).size(),
            HCCL_ERROR(
                "[InsTempAllGatherHD][%s] rank[%u] invalid channel access, fromRankKey[%u] toRankKey[%u] "
                "channelIdx[%u] channels.size[%zu] fromChannelSize[%zu] toChannelSize[%zu]",
                __func__, myRank_, fromRankKey, toRankKey, channelIdx, channels.size(),
                channels.count(fromRankKey) ? channels.at(fromRankKey).size() : 0,
                channels.count(toRankKey) ? channels.at(toRankKey).size() : 0),
            HCCL_E_INTERNAL);
        const ChannelInfo& channelRecv = channels.at(fromRankKey)[channelIdx];
        const ChannelInfo& channelSend = channels.at(toRankKey)[channelIdx];
        HCCL_DEBUG(
            "[InsTempAllGatherHD] rank[%u] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] nSlices[%u]",
            myRank_, templateRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        void* localBuff
            = (!enableRemoteMemAccess_) ? tempAlgParams_.buffInfo.hcclBuff.addr : tempAlgParams_.buffInfo.outputPtr;
        void* sendRemoteBuff
            = (!enableRemoteMemAccess_) ? channelSend.remoteCclMem.addr : channelSend.remoteOutputGraphMode.addr;
        void* recvRemoteBuff
            = (!enableRemoteMemAccess_) ? channelRecv.remoteCclMem.addr : channelRecv.remoteOutputGraphMode.addr;

        const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
        // HD 倍增收集：整段发送持有区段，接收补半区段
        for (u64 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
            for (u32 i = 0; i < stepInfo.nSlices; ++i) {
                u32 txIdx = stepInfo.txSliceIdxs[i];
                u32 rxIdx = stepInfo.rxSliceIdxs[i];
                u64 txPartialOffset = (txIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ?
                                          dataOffsetTail_[channelIdx] :
                                          dataOffset_[channelIdx];
                u64 rxPartialOffset = (rxIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ?
                                          dataOffsetTail_[channelIdx] :
                                          dataOffset_[channelIdx];
                u64 txSliceSize = (txIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ?
                                      dataSplitTail_[channelIdx] :
                                      dataSplit_[channelIdx];
                u64 rxSliceSize = (rxIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ?
                                      dataSplitTail_[channelIdx] :
                                      dataSplit_[channelIdx];

                u64 txScratchOff = 0;
                u64 rxScratchOff = 0;
                if (isOutputCclBuff_) {
                    const u64 scratchBase
                        = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
                    txScratchOff = scratchBase + tempAlgParams_.outputSliceStride * txIdx + txPartialOffset;
                    rxScratchOff = scratchBase + tempAlgParams_.outputSliceStride * rxIdx + rxPartialOffset;
                } else {
                    const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
                    const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
                    txScratchOff = scratchBase + tempAlgParams_.sliceSize * txIdx + txPartialOffset;
                    rxScratchOff = scratchBase + tempAlgParams_.sliceSize * rxIdx + rxPartialOffset;
                }
                const u64 outBase = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
                u64 txOutOff = outBase + tempAlgParams_.outputSliceStride * txIdx + txPartialOffset;
                u64 rxOutOff = outBase + tempAlgParams_.outputSliceStride * rxIdx + rxPartialOffset;

                // 对称内存路径下源/目的切换为 output 与图模式远端指针（OSS 地址）
                u64 txOff = (!enableRemoteMemAccess_) ? txScratchOff : txOutOff;
                u64 rxOff = (!enableRemoteMemAccess_) ? rxScratchOff : rxOutOff;
                txSrcSlices.emplace_back(localBuff, txOff, txSliceSize, txSliceSize / dataTypeSize);
                txDstSlices.emplace_back(sendRemoteBuff, txOff, txSliceSize, txSliceSize / dataTypeSize);
                rxSrcSlices.emplace_back(recvRemoteBuff, rxOff, rxSliceSize, rxSliceSize / dataTypeSize);
                rxDstSlices.emplace_back(localBuff, rxOff, rxSliceSize, rxSliceSize / dataTypeSize);
            }
        }

        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        TxRxChannels sendRecvChannels(channelSend, channelRecv);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);

        if (isDmaRead_) {
            CHK_PRT_RET(
                SendRecvRead(sendRecvInfo, threads[channelIdx]),
                HCCL_ERROR("[InsTempAllGatherHD] SendRecv batch failed (step=%u)", step), HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                SendRecvBatchWrite(sendRecvInfo, threads[channelIdx]),
                HCCL_ERROR("[InsTempAllGatherHD] SendRecv batch failed (step=%u)", step), HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherHD::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAllGatherHD::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

} // namespace ops_hccl
