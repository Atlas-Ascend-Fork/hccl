/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_nhr.h"

namespace ops_hccl {
std::vector<CostModelParam> ReduceNHR::CalcCostCoeff(CalcCostCoeffParam param)
{
    CommTopo netType = param.netType;
    int portNum = (param.isPod && param.netType == CommTopo::COMM_TOPO_CLOS && param.portNum.size() >= 2) ?
                      (param.portNum[0] + param.portNum[1]) :
                      param.portNum[0];
    int kernelNum = 30;
    int log2R = 0;
    for (u32 r = param.rankSize; r > 1; r >>= 1) {
        log2R++;
    }
    int taskNum
        = CostModelManager::CalcTransTaskNum((log2R + 1)) * 4 + CostModelManager::CalcSyncTaskNum((log2R + 1)) * 6;
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    CostModelManager::Global()->CalcNHRParams(param.dataRatio * 2, netType, portNum, param.rankSize, A, param.isPod);
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio * param.rankSize * 2, EngineType::AICPU, B);
    }
    C = 4e-6 * kernelNum;
    D = 1e-6 * taskNum;
    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    HCCL_DEBUG("[%s] CalcCostCoeff A=%f B=%f C=%f D=%f.", __func__, A, B, C, D);
    return params;
}

ReduceNHR::ReduceNHR(
    const OpParam& param,
    const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

ReduceNHR::~ReduceNHR() {}

void ReduceNHR::SetRoot(u32 root) const
{
    (void)root;
    return;
}

bool ReduceNHR::IsRootAlgRank(u32 algRank) const
{
    return algRank < subCommRanks_[0].size() && subCommRanks_[0][algRank] == root_;
}

void* ReduceNHR::GetLocalWorkspacePtr() const
{
    if (rootUsesOutputWorkspace_ && myRank_ == root_) {
        return buffInfo_.outputPtr;
    }
    return buffInfo_.hcclBuff.addr;
}

u64 ReduceNHR::GetLocalWorkspaceBaseOff() const
{
    if (rootUsesOutputWorkspace_ && myRank_ == root_) {
        return buffInfo_.outBuffBaseOff;
    }
    return buffInfo_.hcclBuffBaseOff;
}

void* ReduceNHR::GetRemoteWorkspacePtr(const ChannelInfo& channel, u32 remoteAlgRank) const
{
    if (rootUsesOutputWorkspace_ && IsRootAlgRank(remoteAlgRank)) {
        return channel.remoteOutputGraphMode.addr;
    }
    return channel.remoteCclMem.addr;
}

u64 ReduceNHR::GetRemoteWorkspaceBaseOff(u32 remoteAlgRank) const
{
    if (rootUsesOutputWorkspace_ && IsRootAlgRank(remoteAlgRank)) {
        return buffInfo_.outBuffBaseOff;
    }
    return buffInfo_.hcclBuffBaseOff;
}

HcclResult ReduceNHR::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> level1Channels;
    // MESH_1D_CLOS 层0 CLOS/MESH 实例尺寸对称（GCD>1），Level1Nhr 不可达；本分支仅单级拓扑可达
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestNhrMultiJetty(comm, param, topoInfo, subCommRanks_, myChannelDescs));
        for (const auto& channel : myChannelDescs) {
            if (channel.channelProtocol == COMM_PROTOCOL_UB_CTP) {
                level1Channels.push_back(channel);
            }
        }
        HCCL_INFO("[ReduceNHR::CalcRes] channels total[%zu] ubctp[%zu]", myChannelDescs.size(), level1Channels.size());
    } else {
        CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    }
    CHK_PRT_RET(
        level1Channels.empty(), HCCL_ERROR("[ReduceNHR::CalcRes] no UB_CTP channel after filter"),
        HcclResult::HCCL_E_INTERNAL);
    resourceRequest.channels.push_back(level1Channels);
    // 多通道: 按 Min 口径取每对端通道数——数据切分与通道循环必须
    // 落在最少数对端之内, 避免异构链路数下 channelIdx 越过某对端实际通道数
    channelsPerRank_ = CalcChannelsPerRankMin(level1Channels);
    HCCL_INFO("[ReduceNHR][CalcRes] channelsPerRank[%u] level1Channels[%zu].", channelsPerRank_, level1Channels.size());
    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult ReduceNHR::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 threadNum = 1 * channelsPerRank_;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    HCCL_INFO(
        "[ReduceNHR][GetRes] channelsPerRank_[%u], slaveThreadNum[%u], notifyNumOnMainThread[%u].", channelsPerRank_,
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    return HcclResult::HCCL_SUCCESS;
}

u64 ReduceNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 1;
}

HcclResult
ReduceNHR::KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[ReduceNHR] rank[%d] KernelRun start", myRank_);
    // 处理数据量为0的场景
    CHK_PRT_RET(
        tempAlgParams.sliceSize == 0, HCCL_INFO("[ReduceNHR] sliceSize is 0, no need to process"), HCCL_SUCCESS);

    CHK_PRT_RET(templateRankSize_ == 0, HCCL_ERROR("[ReduceNHR] rankSize is 0"), HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(root_ == UINT32_MAX, HCCL_ERROR("[ReduceNHR] root is invalid"), HcclResult::HCCL_E_INTERNAL);

    dataType_ = param.DataDes.dataType;
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataTypeSize == 0, HCCL_ERROR("[ReduceNHR] dataTypeSize is 0"), HcclResult::HCCL_E_INTERNAL);

    buffInfo_ = tempAlgParams.buffInfo;
    // 对称内存下 root 的工作区改到 userOut；与 executor 的 rootOutputStage（outBuffType=OUTPUT）对齐。
    rootUsesOutputWorkspace_ = param.supportSymmetricMemory && buffInfo_.outBuffType != BufferType::HCCL_BUFFER;
    CHK_PRT_RET(
        rootUsesOutputWorkspace_ && myRank_ == root_ && buffInfo_.outputPtr == nullptr,
        HCCL_ERROR("[ReduceNHR][KernelRun] root output ptr is null, localRank[%u], root[%u]", myRank_, root_),
        HcclResult::HCCL_E_INTERNAL);

    bool isPcieProtocol = IsPcieProtocol(templateResource.channels); // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocol;                                     // 是否使用Read模式
    HCCL_DEBUG("[ReduceNHR] Use Dma Read[%d]", isDmaRead_);

    CHK_RET(getMyAlgRank());
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    HCCL_INFO("[KernelRun] sliceSize: %u, count_: %u, typeSize: %u", tempAlgParams.sliceSize, count_, dataTypeSize);

    const std::map<u32, std::vector<ChannelInfo>>& channels = templateResource.channels;
    HCCL_DEBUG(
        "[Kernel Run] myRank_[%u], channels.size():%zu, channels first[%u]", myRank_, channels.size(),
        channels.begin()->first);

    // 多通道: 运行期刷新统一按 Min 口径。executor 侧 SetchannelsPerRank 按 Max 刷新,
    // 这里重取 Min, 保证切分表/通道循环/线程申请三者同口径, 不越过任一对端的实际通道数。
    if (!templateResource.channels.empty()) {
        channelsPerRank_ = CalcChannelsPerRankMin(templateResource.channels);
    }
    threadNum_ = GetThreadNum();
    CHK_PRT_RET(
        templateResource.threads.size() < threadNum_,
        HCCL_ERROR("[ReduceNHR] threads.size()[%zu] < threadNum_[%u]", templateResource.threads.size(), threadNum_),
        HcclResult::HCCL_E_INTERNAL);
    std::vector<u64> elemCountOut;
    CHK_RET(CalcDataSplitByPortGroup(
        tempAlgParams.sliceSize / dataTypeSize, dataTypeSize, channels.begin()->second, elemCountOut, sizeOut_,
        elemOffset_));
    HCCL_INFO("[ReduceNHR] channelsPerRank_[%u] threadNum_[%u]", channelsPerRank_, threadNum_);

    // 1. 前同步
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    // 2. 每通道前拷贝: userIn 子区 -> 工作区子区, 由本通道队列自行搬运
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(PreCopy(tempAlgParams, templateResource.threads, channelIdx));
    }

    // 3. 每通道: 对端第 channelIdx 条链路 + 队列 channelIdx, 在本通道子区内跑完整 reduce+gather
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_PRT_RET(
            channelIdx >= sizeOut_.size() || channelIdx >= elemOffset_.size(),
            HCCL_ERROR("[ReduceNHR] channelIdx[%u] out of bounds, sizeOutSize[%zu]", channelIdx, sizeOut_.size()),
            HcclResult::HCCL_E_INTERNAL);
        if (sizeOut_[channelIdx] == 0) {
            continue; // 小消息非均匀切分可能产生空通道
        }
        // 切片(按本通道子区大小, 偏移在 RunReduce/RunGather 中叠加 elemOffset_[channelIdx])
        CHK_RET(CalcSlice(sizeOut_[channelIdx]));
        CHK_RET(RunReduce(templateResource.threads, channels, channelIdx));
        CHK_RET(RunGather(templateResource.threads, channels, channelIdx));
    }

    // 4. 每通道回拷: root 将本通道工作区子区 -> userOut 子区, 仍由本通道队列搬运(AR 同款)
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(PostCopy(tempAlgParams, templateResource.threads, channelIdx));
    }

    // 5. 后同步: 等全部通道收尾(含各自队列上的回拷)后, 主线程才算本算子完成
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[ReduceNHR] rank[%d] KernelRun finished", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: 将数据按照rank切分为chunk 块，给后续的reduce操作使用
 * param: dataSize: 待处理的输入数据大小
 * return: sliceInfoVec: 存储数据切分结果
 * return: HcclResult
 */
HcclResult ReduceNHR::CalcSlice(u64 dataSize)
{
    HCCL_INFO("[ReduceNHR] rank[%d] CalcSlice start", dataSize);
    // 按 rank 切分数据（与 AllReduceNHR 保持一致）
    sliceInfoVec_ = RankSliceInfo(templateRankSize_);

    u64 unitAlignSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 chunkSize = RoundUpWithDivisor(dataSize, (templateRankSize_ * unitAlignSize)) / templateRankSize_;

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; rankIdx++) {
        u64 currChunkSize = std::min<u64>(dataSize - accumOff, chunkSize);
        sliceInfoVec_[rankIdx].emplace_back(SliceInfo{accumOff, currChunkSize});
        HCCL_DEBUG(
            "[ReduceNHR] rankIdx [%d] CalcSlice accumOff[%u], currChunkSize[%u]", rankIdx, accumOff, currChunkSize);
        accumOff += currChunkSize;
    }

    CHK_PRT_RET(
        (sliceInfoVec_[templateRankSize_ - 1][0].offset + sliceInfoVec_[templateRankSize_ - 1][0].size != dataSize),
        HCCL_ERROR("[ReduceNHR] chunkSize:[%llu], Rank:[%d], SliceInfo calculation error!", chunkSize, myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
ReduceNHR::PreCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    const BuffInfo& buffInfo = tempAlgParams.buffInfo;
    // 单算子模式，需要先将数据拷贝到工作区; 按通道子区切分, 由本通道队列自搬(AR 同款)
    reduceInBuffType_ = BufferType::HCCL_BUFFER;
    reduceInBuffBaseOff_ = buffInfo.hcclBuffBaseOff;
    reduceOutBuffType_ = buffInfo.outBuffType;
    reduceOutBuffBaseOff_ = buffInfo.outBuffBaseOff;

    CHK_PRT_RET(
        channelIdx >= sizeOut_.size() || channelIdx >= elemOffset_.size(),
        HCCL_ERROR("[ReduceNHR][PreCopy] channelIdx[%u] out of bounds, sizeOutSize[%zu]", channelIdx, sizeOut_.size()),
        HcclResult::HCCL_E_INTERNAL);
    const u64 chSize = sizeOut_[channelIdx];
    const u64 chOff = elemOffset_[channelIdx];
    if (chSize == 0) {
        return HcclResult::HCCL_SUCCESS; // 空通道(小消息非均匀切分), 通信环节同样会跳过
    }

    if (rootUsesOutputWorkspace_ && myRank_ == root_) {
        // 对称内存：root 工作区改到 userOut，省去 CCL 中转与 PostCopy；
        // PostCopy 对对称 root 会提前返回，无需更新 reduceInBuff*/reduceOutBuff*。
        if (buffInfo.inBuffType != BufferType::HCCL_BUFFER) {
            HCCL_INFO("[ReduceNHR][PreCopy] symmetric root copy userIn to userOut, channelIdx[%u]", channelIdx);
            void* const src = static_cast<void*>(static_cast<u8*>(buffInfo.inputPtr) + buffInfo.inBuffBaseOff + chOff);
            void* const dst
                = static_cast<void*>(static_cast<u8*>(buffInfo.outputPtr) + buffInfo.outBuffBaseOff + chOff);
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(threads[channelIdx], dst, src, chSize)));
        } else {
            HCCL_INFO("[ReduceNHR][PreCopy] symmetric root skip precopy, channelIdx[%u]", channelIdx);
        }
    } else if (buffInfo.inBuffType != BufferType::HCCL_BUFFER) {
        HCCL_INFO("[ReduceNHR][PreCopy] Opbase copy userIn to scratchBuffer, channelIdx[%u]", channelIdx);
        void* const src = static_cast<void*>(static_cast<u8*>(buffInfo.inputPtr) + buffInfo.inBuffBaseOff + chOff);
        void* const dst
            = static_cast<void*>(static_cast<u8*>(buffInfo.hcclBuff.addr) + buffInfo.hcclBuffBaseOff + chOff);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(threads[channelIdx], dst, src, chSize)));
    } else {
        HCCL_INFO("[ReduceNHR][PreCopy] skip precopy, channelIdx[%u]", channelIdx);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult ReduceNHR::RunReduce(
    const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels, u32 channelIdx)
{
    HCCL_INFO("[ReduceNHR][RunReduce] channelIdx[%u] start", channelIdx);
    ThreadHandle q = threads[channelIdx];
    std::vector<AicpuNHRStepInfo> stepInfoList;
    CHK_RET(GetStepInfoList(stepInfoList));

    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    for (const auto& stepInfo : stepInfoList) {
        HCCL_DEBUG(
            "[ReduceNHR][RunReduce] step[%u], myRank[%u], toRank[%u], fromRank[%u], nSlices[%u].", stepInfo.step,
            stepInfo.myRank, stepInfo.toRank, stepInfo.fromRank, stepInfo.nSlices);

        const std::vector<ChannelInfo>& recvChannels = channels.at(subCommRanks_.at(0).at(stepInfo.fromRank));
        const std::vector<ChannelInfo>& sendChannels = channels.at(subCommRanks_.at(0).at(stepInfo.toRank));
        CHK_PRT_RET(
            channelIdx >= recvChannels.size() || channelIdx >= sendChannels.size(),
            HCCL_ERROR(
                "[ReduceNHR][RunReduce] link missing: channelIdx[%u] recvSize[%zu] sendSize[%zu]", channelIdx,
                recvChannels.size(), sendChannels.size()),
            HcclResult::HCCL_E_INTERNAL);
        const ChannelInfo& channelRecv = recvChannels.at(channelIdx);
        const ChannelInfo& channelSend = sendChannels.at(channelIdx);

        void* const localPtr = GetLocalWorkspacePtr();
        const u64 localBaseOff = GetLocalWorkspaceBaseOff();
        void* const sendPtr = GetRemoteWorkspacePtr(channelSend, stepInfo.toRank);
        const u64 sendBaseOff = GetRemoteWorkspaceBaseOff(stepInfo.toRank);
        void* const recvPtr = GetRemoteWorkspacePtr(channelRecv, stepInfo.fromRank);
        const u64 recvBaseOff = GetRemoteWorkspaceBaseOff(stepInfo.fromRank);
        if (rootUsesOutputWorkspace_ && IsRootAlgRank(stepInfo.toRank) && sendPtr == nullptr) {
            HCCL_ERROR(
                "[ReduceNHR][RunReduce] remote root output addr is null, localRank[%u], remoteRank[%u]", myRank_,
                subCommRanks_.at(0).at(stepInfo.toRank));
            return HcclResult::HCCL_E_INTERNAL;
        }
        if (rootUsesOutputWorkspace_ && IsRootAlgRank(stepInfo.fromRank) && recvPtr == nullptr) {
            HCCL_ERROR(
                "[ReduceNHR][RunReduce] remote root output addr is null, localRank[%u], remoteRank[%u]", myRank_,
                subCommRanks_.at(0).at(stepInfo.fromRank));
            return HcclResult::HCCL_E_INTERNAL;
        }

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        // 硬性不变量：同一步内各 rx slice 取自 sliceInfoVec_ 的不同分片，root/local 目标区间互不重叠；
        // 因此 root 直写 userOut 后 SendRecvBatchWriteReduce 的融合归约加法序仍然确定。
        // 在 inBuff 上进行 ReduceScatter 操作(多通道: 叠加本通道子区偏移)
        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            const u64 txSliceOff = sliceInfoVec_[stepInfo.txSliceIdxs[i]][0].offset;
            const u64 txSize = sliceInfoVec_[stepInfo.txSliceIdxs[i]][0].size;
            const u64 rxSliceOff = sliceInfoVec_[stepInfo.rxSliceIdxs[i]][0].offset;
            const u64 rxSize = sliceInfoVec_[stepInfo.rxSliceIdxs[i]][0].size;
            const u64 txSrcOffset = txSliceOff + elemOffset_[channelIdx] + localBaseOff;
            const u64 txDstOffset = txSliceOff + elemOffset_[channelIdx] + sendBaseOff;
            const u64 rxSrcOffset = rxSliceOff + elemOffset_[channelIdx] + recvBaseOff;
            const u64 rxDstOffset = rxSliceOff + elemOffset_[channelIdx] + localBaseOff;
            DataSlice txSrcSlice = DataSlice(localPtr, txSrcOffset, txSize, txSize / dataTypeSize);
            DataSlice txDstSlice = DataSlice(sendPtr, txDstOffset, txSize, txSize / dataTypeSize);
            DataSlice rxSrcSlice = DataSlice(recvPtr, rxSrcOffset, rxSize, rxSize / dataTypeSize);
            DataSlice rxDstSlice = DataSlice(localPtr, rxDstOffset, rxSize, rxSize / dataTypeSize);
            txSrcSlices.push_back(txSrcSlice);
            txDstSlices.push_back(txDstSlice);
            rxSrcSlices.push_back(rxSrcSlice);
            rxDstSlices.push_back(rxDstSlice);
        }
        SendRecvReduceInfo sendRecvReduceInfo{
            {channelSend, channelRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_, reduceOp_};

        if (isDmaRead_) {
            CHK_PRT_RET(
                SendRecvReadReduce(sendRecvReduceInfo, q), HCCL_ERROR("[ReduceNHR] RunReduce SendRecvReduce failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                SendRecvBatchWriteReduce(sendRecvReduceInfo, q),
                HCCL_ERROR("[ReduceNHR] RunReduce SendRecvReduce failed"), HcclResult::HCCL_E_INTERNAL);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult ReduceNHR::RunGather(
    const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels, u32 channelIdx)
{
    ThreadHandle q = threads[channelIdx];
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    u32 nSteps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        CHK_PRT_RET(
            stepInfo.fromRank >= templateRankSize_ || stepInfo.toRank >= templateRankSize_,
            HCCL_ERROR(
                "[ReduceNHR][RunGather] step[%u], stepInfo.fromRank[%u], stepInfo.toRank[%u], templateRankSize_", step,
                stepInfo.fromRank, stepInfo.toRank, templateRankSize_),
            HcclResult::HCCL_E_INTERNAL);

        const std::vector<ChannelInfo>& recvChannels = channels.at(subCommRanks_.at(0).at(stepInfo.fromRank));
        const std::vector<ChannelInfo>& sendChannels = channels.at(subCommRanks_.at(0).at(stepInfo.toRank));
        CHK_PRT_RET(
            channelIdx >= recvChannels.size() || channelIdx >= sendChannels.size(),
            HCCL_ERROR(
                "[ReduceNHR][RunGather] link missing: channelIdx[%u] recvSize[%zu] sendSize[%zu]", channelIdx,
                recvChannels.size(), sendChannels.size()),
            HcclResult::HCCL_E_INTERNAL);
        const ChannelInfo& channelRecv = recvChannels.at(channelIdx);
        const ChannelInfo& channelSend = sendChannels.at(channelIdx);

        void* const localPtr = GetLocalWorkspacePtr();
        const u64 localBaseOff = GetLocalWorkspaceBaseOff();
        void* const sendPtr = GetRemoteWorkspacePtr(channelSend, stepInfo.toRank);
        const u64 sendBaseOff = GetRemoteWorkspaceBaseOff(stepInfo.toRank);
        void* const recvPtr = GetRemoteWorkspacePtr(channelRecv, stepInfo.fromRank);
        const u64 recvBaseOff = GetRemoteWorkspaceBaseOff(stepInfo.fromRank);
        if (rootUsesOutputWorkspace_ && IsRootAlgRank(stepInfo.toRank) && sendPtr == nullptr) {
            HCCL_ERROR(
                "[ReduceNHR][RunGather] remote root output addr is null, localRank[%u], remoteRank[%u]", myRank_,
                subCommRanks_.at(0).at(stepInfo.toRank));
            return HcclResult::HCCL_E_INTERNAL;
        }
        if (rootUsesOutputWorkspace_ && IsRootAlgRank(stepInfo.fromRank) && recvPtr == nullptr) {
            HCCL_ERROR(
                "[ReduceNHR][RunGather] remote root output addr is null, localRank[%u], remoteRank[%u]", myRank_,
                subCommRanks_.at(0).at(stepInfo.fromRank));
            return HcclResult::HCCL_E_INTERNAL;
        }

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            const u64 txSliceOff = sliceInfoVec_[stepInfo.txSliceIdxs[i]][0].offset;
            const u64 txSize = sliceInfoVec_[stepInfo.txSliceIdxs[i]][0].size;
            const u64 rxSliceOff = sliceInfoVec_[stepInfo.rxSliceIdxs[i]][0].offset;
            const u64 rxSize = sliceInfoVec_[stepInfo.rxSliceIdxs[i]][0].size;
            const u64 txSrcOffset = txSliceOff + elemOffset_[channelIdx] + localBaseOff;
            const u64 txDstOffset = txSliceOff + elemOffset_[channelIdx] + sendBaseOff;
            const u64 rxSrcOffset = rxSliceOff + elemOffset_[channelIdx] + recvBaseOff;
            const u64 rxDstOffset = rxSliceOff + elemOffset_[channelIdx] + localBaseOff;

            txSrcSlices.push_back(DataSlice(localPtr, txSrcOffset, txSize, txSize / dataTypeSize));
            txDstSlices.push_back(DataSlice(sendPtr, txDstOffset, txSize, txSize / dataTypeSize));
            rxSrcSlices.push_back(DataSlice(recvPtr, rxSrcOffset, rxSize, rxSize / dataTypeSize));
            rxDstSlices.push_back(DataSlice(localPtr, rxDstOffset, rxSize, rxSize / dataTypeSize));
        }

        TxRxChannels sendRecvChannels(channelSend, channelRecv);
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});

        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);
        if (isDmaRead_) {
            CHK_PRT_RET(
                SendRecvRead(sendRecvInfo, q), HCCL_ERROR("[ReduceNHR] RunGather send/recv failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                SendRecvBatchWrite(sendRecvInfo, q), HCCL_ERROR("[ReduceNHR] RunGather send/recv failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult ReduceNHR::PostCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads, u32 channelIdx) const
{
    if (reduceOutBuffType_ == BufferType::HCCL_BUFFER) {
        HCCL_DEBUG("[ReduceNHR][PostCopy] skip postcopy rank[%d]", myRank_);
        // 希望输出结果在HCCL_BUFFER上，NHR算法做完已经在HCCL_BUFFER上，无需再进行后拷贝
        return HCCL_SUCCESS;
    }
    // PostCopy 仅在 OPBASE 并且在 root 上执行（root 收到完整结果后写回用户 out）
    const BuffInfo& buffInfo = tempAlgParams.buffInfo;
    if (rootUsesOutputWorkspace_ && myRank_ == root_) {
        HCCL_DEBUG("[ReduceNHR][PostCopy] symmetric root already in userOut, skip");
        return HcclResult::HCCL_SUCCESS;
    }
    if (myRank_ != root_) {
        HCCL_DEBUG("[ReduceNHR][PostCopy] not root, skip postcopy rank[%d]", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        channelIdx >= sizeOut_.size() || channelIdx >= elemOffset_.size(),
        HCCL_ERROR("[ReduceNHR][PostCopy] channelIdx[%u] out of bounds, sizeOutSize[%zu]", channelIdx, sizeOut_.size()),
        HcclResult::HCCL_E_INTERNAL);
    if (sizeOut_[channelIdx] == 0) {
        return HcclResult::HCCL_SUCCESS; // 空通道(小消息非均匀切分), 无数据可回拷
    }

    HCCL_INFO("[ReduceNHR][PostCopy] Opbase root copy scratchBuffer to userOut, channelIdx[%u]", channelIdx);
    void* const srcBasePtr
        = (reduceInBuffType_ == BufferType::HCCL_BUFFER) ? buffInfo.hcclBuff.addr : buffInfo.inputPtr;
    void* const dstBasePtr = buffInfo.outputPtr;
    void* const src = static_cast<void*>(static_cast<u8*>(srcBasePtr) + reduceInBuffBaseOff_ + elemOffset_[channelIdx]);
    void* const dst
        = static_cast<void*>(static_cast<u8*>(dstBasePtr) + reduceOutBuffBaseOff_ + elemOffset_[channelIdx]);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(threads[channelIdx], dst, src, sizeOut_[channelIdx])));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult ReduceNHR::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo) const
{
    u32 rankIdx = myIdx_;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;

    // 计算通信对象
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 sendTo = (rankIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = rankIdx;
    u32 rxSliceIdx = (rankIdx - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG("[ReduceNHR][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

//  计算每轮收发的对端以及slice编号
HcclResult ReduceNHR::GetStepInfoList(std::vector<AicpuNHRStepInfo>& stepInfoList) const
{
    // ReduceNHR 将本 rank 号转换成算法使用的索引号
    u32 rankIdx = myIdx_;
    stepInfoList.clear();

    u32 nSteps = GetNHRStepNum(templateRankSize_);
    stepInfoList.resize(nSteps);
    for (u32 step = 0; step < nSteps; step++) {
        // ReduceNHR 数据份数和数据编号增量
        u32 nSlices = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
        u32 deltaSliceIndex = 1 << (step + 1);
        // ReduceNHR 计算通信对象
        u32 deltaRank = 1 << step;
        u32 sendTo = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
        u32 recvFrom = (rankIdx + deltaRank) % templateRankSize_;

        u32 txSliceIdx = sendTo;
        u32 rxSliceIdx = rankIdx;

        AicpuNHRStepInfo& currStepInfo = stepInfoList[step];
        currStepInfo.toRank = sendTo;
        currStepInfo.fromRank = recvFrom;
        currStepInfo.step = step;
        currStepInfo.myRank = rankIdx;
        currStepInfo.nSlices = nSlices;

        // ReduceNHR 计算本rank在每轮收/发中的slice编号
        currStepInfo.txSliceIdxs.reserve(nSlices);
        currStepInfo.rxSliceIdxs.reserve(nSlices);
        for (u32 i = 0; i < nSlices; i++) {
            currStepInfo.txSliceIdxs.push_back(txSliceIdx);
            currStepInfo.rxSliceIdxs.push_back(rxSliceIdx);
            HCCL_DEBUG("[ReduceNHR][GetStepInfoList] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);
            txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
            rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

std::pair<std::vector<DataSlice>, std::vector<DataSlice>>
ReduceNHR::getTxRxSlices(const AicpuNHRStepInfo& stepInfo, const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    std::vector<DataSlice> txSlices;
    std::vector<DataSlice> rxSlices;

    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    for (u32 i = 0; i < stepInfo.nSlices; i++) {
        u32 txRank = stepInfo.txSliceIdxs[i];
        u32 rxRank = stepInfo.rxSliceIdxs[i];
        u64 txOffset = sliceInfoVec_.at(txRank).at(0).offset + buffInfo_.hcclBuffBaseOff;
        u64 txSize = sliceInfoVec_.at(txRank).at(0).size;
        u64 rxOffset = sliceInfoVec_.at(rxRank).at(0).offset + buffInfo_.hcclBuffBaseOff;
        u64 rxSize = sliceInfoVec_.at(rxRank).at(0).size;
        void* const txAddr
            = (txRank == myRank_) ? buffInfo_.hcclBuff.addr : channels.at(txRank).at(0).remoteCclMem.addr;
        void* const rxAddr
            = (rxRank == myRank_) ? buffInfo_.hcclBuff.addr : channels.at(rxRank).at(0).remoteCclMem.addr;
        HCCL_DEBUG(
            "[getTxRxSlices] myRank:%u, i:%u, txRank:%u, txOffset:%u, txSize:%u, rxRank:%u,rxOffset:%u, "
            "rxSize:%u, txAddr:[%#llx], rxAddr:[%#llx]",
            myRank_, i, txRank, txOffset, txSize, rxRank, rxOffset, rxSize, txAddr, rxAddr);

        // 默认reduceInBuffType_为SCRATCH的场景
        txSlices.emplace_back(txAddr, txOffset, txSize, txSize / dataTypeSize);
        rxSlices.emplace_back(rxAddr, rxOffset, rxSize, rxSize / dataTypeSize);
    }
    return std::make_pair(txSlices, rxSlices);
}

HcclResult ReduceNHR::getMyAlgRank()
{
    CHK_PRT_RET(
        subCommRanks_.empty() || subCommRanks_[0].empty(), HCCL_ERROR("[ReduceNHR] subCommRanks_[0] is empty"),
        HCCL_E_INTERNAL);
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    CHK_PRT_RET(
        iter == subCommRanks_[0].end(), HCCL_ERROR("[ReduceNHR] Cannot find myRank[%u] in subCommRanks_[0]", myRank_),
        HCCL_E_INTERNAL);
    myIdx_ = static_cast<u32>(std::distance(subCommRanks_[0].begin(), iter));
    CHK_PRT_RET(
        myIdx_ >= templateRankSize_,
        HCCL_ERROR(
            "[ReduceNHR] rank idx[%u] in virtRankMap is invalid, it should be less than rankSize[%u]", myIdx_,
            templateRankSize_),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

void ReduceNHR::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = 1 * channelsPerRank_;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void ReduceNHR::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = 1 * channelsPerRank_;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

u64 ReduceNHR::GetThreadNum() const { return 1 * channelsPerRank_; }

} // namespace ops_hccl
