/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_hd.h"

namespace ops_hccl {
constexpr int TASK_NUM_EXTRA_OVERHEAD = 8;
// 经验常数：冻结 spec ar-hd-aicpu-20261016 §8（与 InsTempReduceScatterNHR 实测经验值对齐，
// pairwise a73c637f 同款命名常量先例）；HD 为 preferred-only（仅显式配置可选，不参与默认成本
// 竞争），常数值仅影响 estimate 输出，未标定不作为性能结论
constexpr auto HD_TRANS_TASK_SCALE = 1.5;    // taskNum 传输项缩放（保持原 double 字面量 1.5 语义）
constexpr auto HD_TASK_LAUNCH_COEFF = 1e-6f; // D 项：taskNum 个 task 的下发时延

std::vector<CostModelParam> InsTempReduceScatterHD::CalcCostCoeff(CalcCostCoeffParam param)
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
    int kernelNum = 21;
    int taskNum = CostModelManager::CalcTransTaskNum((log2R + 1)) * HD_TRANS_TASK_SCALE
                  + CostModelManager::CalcSyncTaskNum((log2R + 1)) * 2;
    taskNum = (!param.isPod) ? taskNum : taskNum * 2;
    taskNum = taskNum + TASK_NUM_EXTRA_OVERHEAD;
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    float B1 = 0.0f;
    float B2 = 0.0f;
    // RS HD: 1dpod64单server pod内CLOS无跨pod端口预留, 不做isPod端口减半(减半使A斜率高估~1.74x)
    CostModelManager::Global()->CalcNHRParams(param.dataRatio, netType, portNum, param.rankSize, A, param.isPod, false);
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B1);
    }
    if (param.outputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B2);
    }
    B = B1 * (param.rankSize / 2) + B2;
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AICPU, C);
    // nhr实测和理论估计相差较大，先用经验值
    D = HD_TASK_LAUNCH_COEFF * taskNum;

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

InsTempReduceScatterHD::InsTempReduceScatterHD(
    const OpParam& param, const u32 rankId, // 传通信域的u32，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempReduceScatterHD::~InsTempReduceScatterHD() {}

HcclResult InsTempReduceScatterHD::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> channels;
    bool isUBX = topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix;
    if (isUBX) {
        CHK_RET(CalcChannelRequestNhrMultiJettyUbx(comm, param, topoInfo, subCommRanks_, channels));
    } else {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, myChannelDescs));
        for (auto channel : myChannelDescs) {
            if (channel.channelProtocol != CommProtocol::COMM_PROTOCOL_ROCE) {
                channels.push_back(channel);
            }
        }
    }
    resourceRequest.channels.push_back(channels);
    u32 channelsPerRank = CalcChannelsPerRank(channels);
    HCCL_INFO("[InsTempReduceScatterHD][CalcRes] channelsPerRank: [%u].", channelsPerRank);
    channelsPerRank_ = channelsPerRank;
    // 非 UBX 分支过滤 ROCE 通道后可能为空，channelsPerRank 为 0 时 GetRes 的 threadNum-1 会 u32 下溢
    CHK_PRT_RET(
        channelsPerRank_ == 0, HCCL_ERROR("[InsTempReduceScatterHD][CalcRes] channelsPerRank_ is 0."),
        HcclResult::HCCL_E_INTERNAL);
    if (isUBX && channelsPerRank_ > MAX_JETTY_NUM) {
        HCCL_ERROR(
            "[InsTempReduceScatterHD][CalcRes] channelsPerRank_ %u is greater than MAX_JETTY_NUM %u.", channelsPerRank_,
            MAX_JETTY_NUM);
    } else {
        HCCL_DEBUG("[InsTempReduceScatterHD][CalcRes] channelsPerRank_ is %u.", channelsPerRank_);
    }
    GetRes(resourceRequest);
    HCCL_INFO(
        "[InsTempReduceScatterHD][CalcRes] slaveThreadNum: [%u], notifyNumOnMainThread: [%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterHD::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 threadNum = 1 * channelsPerRank_;
    CHK_PRT_RET(
        threadNum == 0, HCCL_ERROR("[InsTempReduceScatterHD][GetRes] threadNum is 0, channelsPerRank_ invalid."),
        HcclResult::HCCL_E_INTERNAL);
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    HCCL_INFO(
        "[GetRes] channelsPerRank_: [%u], slaveThreadNum: [%u].", channelsPerRank_, resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterHD::GetThreadNum() const { return 1 * channelsPerRank_; }

HcclResult InsTempReduceScatterHD::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempReduceScatterHD] GenExtIns start");
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        HCCL_INFO("[InsTempReduceScatterHD] sliceSize and tailSize are both 0, skip");
        return HCCL_SUCCESS;
    }
    // 防御性校验：HD 调度要求框间 rank 数为 2 的整数次幂（主门控在 TopoMatchTwoLevel）
    CHK_PRT_RET(
        (templateRankSize_ & (templateRankSize_ - 1)) != 0,
        HCCL_ERROR(
            "[InsTempReduceScatterHD][KernelRun] templateRankSize_[%u] is not a power of two.", templateRankSize_),
        HcclResult::HCCL_E_NOT_SUPPORT);
    // 防御性校验：RunHD 按 HD 步骤表下标访问 subCommRanks_[0]，rankList 数必须与模板 rank 数一致
    CHK_PRT_RET(
        subCommRanks_.size() == 0, HCCL_ERROR("[InsTempReduceScatterHD][KernelRun] subCommRanks is empty."),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        subCommRanks_[0].size() != templateRankSize_,
        HCCL_ERROR(
            "[InsTempReduceScatterHD][KernelRun] rank count[%zu] is invalid in rank list, templateRankSize[%u].",
            subCommRanks_[0].size(), templateRankSize_),
        HcclResult::HCCL_E_INTERNAL);
    tempAlgParams_ = tempAlgParams;
    channels_ = templateResource.channels;
    CHK_PRT_RET(
        channels_.empty(), HCCL_ERROR("[InsTempReduceScatterHD][KernelRun] channels is empty."),
        HcclResult::HCCL_E_INTERNAL);
    for (const auto& channelsByRank : channels_) {
        CHK_PRT_RET(
            channelsByRank.second.empty(),
            HCCL_ERROR(
                "[InsTempReduceScatterHD][KernelRun] channel list of remote rank[%u] is empty.", channelsByRank.first),
            HcclResult::HCCL_E_INTERNAL);
    }
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    supportSymmetricMemAccess_ = param.supportSymmetricMemory;

    bool isPcieProtocol = IsPcieProtocol(channels_); // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocol;                     // 是否使用Read模式
    doPreCopy_ = false;
    skipStep0TxPreCopy_ = false;
    step0TxSliceIdxs_.clear();
    HCCL_DEBUG("[InsTempReduceScatterHD] Use Dma Read[%d]", isDmaRead_);

    if (supportSymmetricMemAccess_) {
        HCCL_INFO("[InsTempReduceScatterHD] Remote mem access enabled, skip PreCopy, use read+reduce on input");
    } else if (!isDmaRead_ && templateRankSize_ > 1) {
        skipStep0TxPreCopy_ = true;
        HCCL_INFO("[InsTempReduceScatterHD] Skip step0 tx pre-copy enabled");
    }

    std::vector<u64> elemCountOut;
    std::vector<u64> sizeOut;
    std::vector<u64> elemOffset;
    u64 totalDataCount = tempAlgParams.sliceSize / dataTypeSize_;
    CHK_RET(CalcDataSplitByPortGroup(
        totalDataCount, dataTypeSize_, channels_.begin()->second, elemCountOut, sizeOut, elemOffset));
    elemOffset_ = elemOffset;
    sizeOut_ = sizeOut;

    if (tempAlgParams.tailSize > 0) {
        std::vector<u64> elemCountOutTail;
        std::vector<u64> sizeOutTail;
        std::vector<u64> elemOffsetTail;
        u64 totalDataCountTail = tempAlgParams.tailSize / dataTypeSize_;
        CHK_RET(CalcDataSplitByPortGroup(
            totalDataCountTail, dataTypeSize_, channels_.begin()->second, elemCountOutTail, sizeOutTail,
            elemOffsetTail));
        elemOffsetTail_ = elemOffsetTail;
        sizeOutTail_ = sizeOutTail;
    }

    threadNum_ = GetThreadNum();
    CHK_PRT_RET(
        threadNum_ > templateResource.threads.size(),
        HCCL_ERROR(
            "[InsTempReduceScatterHD][KernelRun] thread num[%u] more than threads size[%zu].", threadNum_,
            templateResource.threads.size()),
        HcclResult::HCCL_E_INTERNAL);
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    if (skipStep0TxPreCopy_) {
        CHK_RET(CollectStep0TxSliceIdxs());
    }

    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_PRT_RET(
            channelIdx >= sizeOut.size() || channelIdx >= elemOffset.size(),
            HCCL_ERROR("[InsTempReduceScatterHD] channelIdx[%u] out of bounds", channelIdx), HCCL_E_INTERNAL);
        if (!supportSymmetricMemAccess_) {
            CHK_RET(LocalDataCopy(templateResource.threads, channelIdx));
        }
    }
    if (templateRankSize_ <= 1) {
        for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
            CHK_RET(PostLocalCopy(templateResource.threads, channelIdx));
        }
        if (threadNum_ > 1) {
            std::vector<ThreadHandle> subThreads(
                templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
            GetNotifyIdxSubToMain(notifyIdxSubToMain_);
            CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
        }
        return HcclResult::HCCL_SUCCESS;
    }
    // 适配并发场景
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(RunHD(templateResource.threads, channelIdx));
    }
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(PostLocalCopy(templateResource.threads, channelIdx));
    }
    // 后同步
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterHD::LocalDataCopy(const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterHD][LocalDataCopy] empty threads"),
        HcclResult::HCCL_E_INTERNAL);
    ThreadHandle q = threads[channelIdx];
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    for (u32 localRandId = 0; localRandId < templateRankSize_; ++localRandId) {
        if (skipStep0TxPreCopy_ && step0TxSliceIdxs_.count(localRandId) > 0) {
            continue;
        }
        std::vector<u64> sizeOut;
        std::vector<u64> elemOffset;
        sizeOut = sizeOut_;
        elemOffset = elemOffset_;
        if (localRandId == templateRankSize_ - 1 && tempAlgParams_.tailSize > 0) {
            sizeOut = sizeOutTail_;
            elemOffset = elemOffsetTail_;
        }
        for (u64 rpt = 0; rpt < rptNum; ++rpt) {
            const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
            const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;

            const u64 inOff = inBaseOff + localRandId * tempAlgParams_.inputSliceStride + elemOffset[channelIdx];
            const u64 scOff = scratchBase + localRandId * tempAlgParams_.sliceSize + elemOffset[channelIdx];

            DataSlice src = DataSlice(tempAlgParams_.buffInfo.inputPtr, inOff, sizeOut[channelIdx]);
            DataSlice dst = DataSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scOff, sizeOut[channelIdx]);

            // 如果源slice和目标slice的type类型相同且base偏移相同，则不需要做拷贝
            if (tempAlgParams_.buffInfo.inBuffType != tempAlgParams_.buffInfo.hcclBuffType
                || inBaseOff != scratchBase) {
                doPreCopy_ = true;
                CHK_RET(LocalCopy(q, src, dst));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterHD::PostLocalCopy(const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    CHK_PRT_RET(threads.empty(), HCCL_ERROR("[RS-HD][PostLocalCopy] empty queue"), HcclResult::HCCL_E_INTERNAL);

    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));
    std::vector<u64> sizeOut;
    std::vector<u64> elemOffset;
    if (myAlgIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize > 0) {
        sizeOut = sizeOutTail_;
        elemOffset = elemOffsetTail_;
    } else {
        sizeOut = sizeOut_;
        elemOffset = elemOffset_;
    }
    ThreadHandle q = threads[channelIdx];

    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        // 如果做了前拷贝，则数据在ccl上紧密排列，按照sliceSize跳过间隔
        u64 scOff = scratchBase + tempAlgParams_.sliceSize * myAlgIdx + elemOffset[channelIdx];
        void* srcAddr = tempAlgParams_.buffInfo.hcclBuff.addr;
        if (supportSymmetricMemAccess_) {
            // 对称内存路径：数据在input上，按照inputSliceStride跳过间隔
            scOff = inBaseOff + tempAlgParams_.inputSliceStride * myAlgIdx + elemOffset[channelIdx];
            srcAddr = tempAlgParams_.buffInfo.inputPtr;
        } else if (!doPreCopy_) {
            // 如果没做前拷贝，则ccl buffer继承input相关的所有参数，按照inputSliceStride跳过间隔
            scOff = scratchBase + tempAlgParams_.inputSliceStride * myAlgIdx + elemOffset[channelIdx];
        }
        const u64 outOff = outBaseOff + myAlgIdx * tempAlgParams_.outputSliceStride + elemOffset[channelIdx];

        DataSlice src = DataSlice(srcAddr, scOff, sizeOut[channelIdx]);
        DataSlice dst = DataSlice(tempAlgParams_.buffInfo.outputPtr, outOff, sizeOut[channelIdx]);

        if (tempAlgParams_.buffInfo.hcclBuffType != tempAlgParams_.buffInfo.outBuffType || scOff != outOff) {
            CHK_RET(LocalCopy(q, src, dst));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterHD::RunHD(const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    CHK_PRT_RET(threads.empty(), HCCL_ERROR("[RS-HD][RunHD] empty queue"), HcclResult::HCCL_E_INTERNAL);

    if (templateRankSize_ <= 1)
        return HcclResult::HCCL_SUCCESS;

    ThreadHandle q = threads[channelIdx];

    // 步进参数
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    // 预计算步骤列表（算法序）
    std::vector<AicpuNHRStepInfo> steps;
    CHK_RET(GetHDStepInfoList(steps));
    for (u32 s = 0; s < steps.size(); ++s) {
        const auto& st = steps[s];

        const u32 recvFromRank = subCommRanks_[0][st.fromRank];
        const u32 sendToRank = subCommRanks_[0][st.toRank];
        CHK_PRT_RET(
            recvFromRank == static_cast<u32>(-1) || sendToRank == static_cast<u32>(-1),
            HCCL_ERROR("[RS-HD][RunHD] rank map failed: from[%u] to[%u]", st.fromRank, st.toRank),
            HcclResult::HCCL_E_INTERNAL);

        CHK_PRT_RET(
            channels_.count(recvFromRank) == 0 || channels_.count(sendToRank) == 0
                || channelIdx >= channels_[recvFromRank].size() || channelIdx >= channels_[sendToRank].size(),
            HCCL_ERROR(
                "[RS-HD][RunHD] link missing: recvFrom=%d sendTo=%d channelIdx=%u", recvFromRank, sendToRank,
                channelIdx),
            HcclResult::HCCL_E_INTERNAL);
        ChannelInfo linkRecv = channels_[recvFromRank].at(channelIdx);
        ChannelInfo linkSend = channels_[sendToRank].at(channelIdx);

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        txSrcSlices.reserve(st.nSlices * rptNum);
        txDstSlices.reserve(st.nSlices * rptNum);
        rxSrcSlices.reserve(st.nSlices * rptNum);
        rxDstSlices.reserve(st.nSlices * rptNum);

        void* sendRemoteCclBuffAddr
            = supportSymmetricMemAccess_ ? linkSend.remoteInputGraphMode.addr : linkSend.remoteCclMem.addr;
        void* recvRemoteCclBuffAddr
            = supportSymmetricMemAccess_ ? linkRecv.remoteInputGraphMode.addr : linkRecv.remoteCclMem.addr;
        // RS：在 SCRATCH 上进行 HD 折半规约交换
        for (u64 rpt = 0; rpt < rptNum; ++rpt) {
            const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
            const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
            for (u32 i = 0; i < st.nSlices; ++i) {
                const u32 txIdx = st.txSliceIdxs[i]; // 算法序
                const u32 rxIdx = st.rxSliceIdxs[i];

                const u64 txelemOffset = (txIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize > 0) ?
                                             elemOffsetTail_[channelIdx] :
                                             elemOffset_[channelIdx];
                const u64 rxelemOffset = (rxIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize > 0) ?
                                             elemOffsetTail_[channelIdx] :
                                             elemOffset_[channelIdx];
                // 如果做了前拷贝，则数据在ccl上紧密排列，按照sliceSize跳过间隔
                u64 txScOff = scratchBase + tempAlgParams_.sliceSize * txIdx + txelemOffset;
                u64 rxScOff = scratchBase + tempAlgParams_.sliceSize * rxIdx + rxelemOffset;
                if (supportSymmetricMemAccess_) {
                    // 对称内存路径：数据在input上，按照inputSliceStride跳过间隔
                    txScOff = inBaseOff + tempAlgParams_.inputSliceStride * txIdx + txelemOffset;
                    rxScOff = inBaseOff + tempAlgParams_.inputSliceStride * rxIdx + rxelemOffset;
                } else if (!doPreCopy_) {
                    // 如果没做前拷贝，则ccl buffer继承input相关的所有参数，按照inputSliceStride跳过间隔
                    txScOff = scratchBase + tempAlgParams_.inputSliceStride * txIdx + txelemOffset;
                    rxScOff = scratchBase + tempAlgParams_.inputSliceStride * rxIdx + rxelemOffset;
                }

                const u64 txSliceSize = (txIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize > 0) ?
                                            sizeOutTail_[channelIdx] :
                                            sizeOut_[channelIdx];
                const u64 rxSliceSize = (rxIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize > 0) ?
                                            sizeOutTail_[channelIdx] :
                                            sizeOut_[channelIdx];

                void* txSrcAddr = tempAlgParams_.buffInfo.hcclBuff.addr;
                u64 txSrcOff = txScOff;
                u64 txDstOff = txScOff;
                if (supportSymmetricMemAccess_) {
                    txSrcAddr = tempAlgParams_.buffInfo.inputPtr;
                } else if (skipStep0TxPreCopy_ && s == 0) {
                    txSrcAddr = tempAlgParams_.buffInfo.inputPtr;
                    txSrcOff = inBaseOff + tempAlgParams_.inputSliceStride * txIdx + txelemOffset;
                }

                DataSlice txSrcSlice = DataSlice(
                    txSrcAddr, txSrcOff, txSliceSize,
                    txSliceSize / DATATYPE_SIZE_TABLE[dataType_]); // 发送源
                DataSlice txDstSlice = DataSlice(
                    sendRemoteCclBuffAddr, txDstOff, txSliceSize,
                    txSliceSize / DATATYPE_SIZE_TABLE[dataType_]); // 发送目标
                txSrcSlices.push_back(txSrcSlice);
                txDstSlices.push_back(txDstSlice);
                DataSlice rxSrcSlice = DataSlice(
                    recvRemoteCclBuffAddr, rxScOff, rxSliceSize,
                    rxSliceSize / DATATYPE_SIZE_TABLE[dataType_]); // 发送源
                DataSlice rxDstSlice = DataSlice(
                    supportSymmetricMemAccess_ ? tempAlgParams_.buffInfo.inputPtr :
                                                 tempAlgParams_.buffInfo.hcclBuff.addr,
                    rxScOff, rxSliceSize,
                    rxSliceSize / DATATYPE_SIZE_TABLE[dataType_]); // 发送目标
                rxSrcSlices.push_back(rxSrcSlice);
                rxDstSlices.push_back(rxDstSlice);
            }
        }

        SendRecvReduceInfo info{
            {linkSend, linkRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_, reduceOp_};

        if (supportSymmetricMemAccess_) {
            CHK_PRT_RET(
                SendRecvBatchReadReduce(info, threads[channelIdx]),
                HCCL_ERROR("[RS-HD][RunHD] SendRecvBatchReadReduce failed (step=%u)", st.step),
                HcclResult::HCCL_E_INTERNAL);
        } else if (isDmaRead_) {
            CHK_PRT_RET(
                SendRecvReadReduce(info, threads[channelIdx]),
                HCCL_ERROR("[RS-HD][RunHD] SendRecvReadReduce failed (step=%u)", st.step), HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                SendRecvBatchWriteReduce(info, threads[channelIdx]),
                HCCL_ERROR("[RS-HD][RunHD] SendRecvBatchWriteReduce failed (step=%u)", st.step),
                HcclResult::HCCL_E_INTERNAL);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterHD::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    HCCL_INFO("[InsTempReduceScatterHD][CalcScratchMultiple] templateScratchMultiplier[%llu]", templateRankSize_);
    return templateRankSize_;
}

// 计算每步的对端以及块编号：距离递减折半，伙伴 = algRank XOR D，
// 发送补半区段[sentStart, sentStart+D)，保留含本 rank 的半区段[keepStart, keepStart+D)
HcclResult InsTempReduceScatterHD::GetHDStepInfoList(std::vector<AicpuNHRStepInfo>& stepInfoList)
{
    // 将本rank号转换成算法使用的索引号
    u32 u32x = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], u32x));

    stepInfoList.clear();
    step0TxSliceIdxs_.clear();

    u32 nSteps = 0;
    for (u32 tmp = templateRankSize_; tmp > 1; tmp >>= 1, nSteps++) {
    }
    stepInfoList.resize(nSteps);
    for (u32 step = 0; step < nSteps; step++) {
        u32 dist = templateRankSize_ >> (step + 1);
        u32 peerIdx = u32x ^ dist;
        u32 keepStart = u32x & ~(dist - 1);
        u32 sentStart = peerIdx & ~(dist - 1);

        AicpuNHRStepInfo& currStepInfo = stepInfoList[step];
        currStepInfo.step = step;
        currStepInfo.myRank = u32x;
        currStepInfo.nSlices = dist;
        // HD 收发同伙伴
        currStepInfo.toRank = peerIdx;
        currStepInfo.fromRank = peerIdx;

        currStepInfo.txSliceIdxs.reserve(dist);
        currStepInfo.rxSliceIdxs.reserve(dist);
        for (u32 i = 0; i < dist; i++) {
            currStepInfo.txSliceIdxs.push_back(sentStart + i);
            if (step == 0) {
                step0TxSliceIdxs_.insert(sentStart + i);
            }
            currStepInfo.rxSliceIdxs.push_back(keepStart + i);
            HCCL_DEBUG(
                "[InsTempReduceScatterHD][GetHDStepInfoList] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, sentStart + i,
                keepStart + i);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

// 仅预计算首步发送区段编号集合（LocalDataCopy 跳过 PreCopy 用），不生成完整步骤表
HcclResult InsTempReduceScatterHD::CollectStep0TxSliceIdxs()
{
    u32 algRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], algRank));

    step0TxSliceIdxs_.clear();
    if (templateRankSize_ <= 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    // 首步距离 N/2：发送对端所在半区段 [sentStart, sentStart + N/2)
    u32 dist = templateRankSize_ >> 1;
    u32 peerIdx = algRank ^ dist;
    u32 sentStart = peerIdx & ~(dist - 1);
    for (u32 i = 0; i < dist; i++) {
        step0TxSliceIdxs_.insert(sentStart + i);
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempReduceScatterHD::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = 1 * channelsPerRank_;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempReduceScatterHD::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = 1 * channelsPerRank_;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

} // namespace ops_hccl
