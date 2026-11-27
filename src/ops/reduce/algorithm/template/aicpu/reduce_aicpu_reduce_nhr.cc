/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_aicpu_reduce_nhr.h"
#include "cost_model.h"

namespace ops_hccl {
std::vector<CostModelParam> ReduceAicpuReduceNHR::CalcCostCoeff(CalcCostCoeffParam param)
{
    // 口径: cost 乘子 dataSize 为单 rank 全量输入,
    // dataRatio×rankSize×dataSize 为本模板单 rank 输入全量
    // (sole 整域 dataRatio=1/rankSize, dataRatio×rankSize=1.0)
    //   传输项 A: 单轮整域 NHR AllGather, 每 rank 各持 1 个全量 slot,
    //             gather 后发送 (rankSize-1) 份全量
    //   B: 入拷贝(PreCopy 全量入 cclBuff) + 出拷贝(全量),
    //      输入/输出已在 cclBuff 的场景由入参 BufferType 跳过
    //   C: 4×NHR 步收发 + 拷贝/规约任务(入拷贝 1 + 出拷贝 1 + 规约 rankSize-1)
    //      + 规约前一次强制同步
    //      (折算 SYNC_EQUIV_TASKS 个 2us task)
    //      (64bit/PROD 的规约吞吐后续在 B 中按 count×单 count 耗时折算, 此处不建模)
    CommTopo netType = param.netType;
    int portNum = 1;
    if (!param.portNum.empty()) {
        portNum = (param.netType == CommTopo::COMM_TOPO_CLOS && param.portNum.size() >= 2) ?
                      (param.portNum[0] + param.portNum[1]) :
                      param.portNum[0];
    }
    if (portNum <= 0) {
        portNum = 1;
    }
    float A = 0.0f;
    CostModelManager::Global()->CalcNHRParams(
        param.dataRatio * param.rankSize, netType, portNum, param.rankSize, A, param.isPod);

    // B: 入拷贝/出拷贝各 1 份全量
    float B1 = 0.0f;
    float B2 = 0.0f;
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio * param.rankSize, EngineType::AICPU, B1);
    }
    if (param.outputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio * param.rankSize, EngineType::AICPU, B2);
    }
    const float B = B1 + B2;

    // AICPU 规约工作量: RunReduce 串行规约 (rankSize-1) 次全量
    // (每份 dataRatio×rankSize×dataSize 字节),
    // 64bit/PROD 时由框架 per-call 按 count×1.2ns 折算进 B
    const float reduceCopies
        = static_cast<float>(param.rankSize - 1) * param.dataRatio * static_cast<float>(param.rankSize);

    constexpr int SYNC_EQUIV_TASKS = 50;
    int nSteps = static_cast<int>(GetNHRStepNum(param.rankSize));
    int taskNum = static_cast<int>(param.rankSize) - 1 + 4 * nSteps + 2;
    float C = 0.0f;
    CostModelManager::Global()->CalcLatencyParams(taskNum + SYNC_EQUIV_TASKS, EngineType::AICPU, C);

    // D: 任务下发固定项(不随数据量变化)
    float D = 1e-6f * taskNum;

    HCCL_INFO(
        "[ReduceAicpuReduceNHR][CalcCostCoeff] algName[%s] rankSize[%u] "
        "A=%e B=%e(copy=%e) C=%e D=%e reduceCopies=%f.",
        param.algName != nullptr ? param.algName : "(null)", param.rankSize, A, B, B1 + B2, C, D, reduceCopies);
    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D, reduceCopies});
    return params;
}

ReduceAicpuReduceNHR::ReduceAicpuReduceNHR(
    const OpParam& param,
    const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

ReduceAicpuReduceNHR::~ReduceAicpuReduceNHR() {}

void ReduceAicpuReduceNHR::SetRoot(u32 root)
{
    (void)root;
    return;
}

HcclResult ReduceAicpuReduceNHR::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    // 多通道: 按 Min 口径取每对端通道数——数据切分与通道循环必须落在最少数
    channelsPerRank_ = CalcChannelsPerRankMin(level1Channels);
    HCCL_INFO(
        "[ReduceAicpuReduceNHR][CalcRes] channelsPerRank[%u] level1Channels[%zu].", channelsPerRank_,
        level1Channels.size());
    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult ReduceAicpuReduceNHR::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 threadNum = 1 * channelsPerRank_;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    HCCL_INFO(
        "[ReduceAicpuReduceNHR][GetRes] channelsPerRank_[%u], slaveThreadNum[%u], notifyNumOnMainThread[%u].",
        channelsPerRank_, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    return HcclResult::HCCL_SUCCESS;
}

u64 ReduceAicpuReduceNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

HcclResult ReduceAicpuReduceNHR::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[ReduceAicpuReduceNHR] rank[%d] KernelRun start", myRank_);
    // 处理数据量为0的场景
    CHK_PRT_RET(
        tempAlgParams.sliceSize == 0, HCCL_INFO("[ReduceAicpuReduceNHR] sliceSize is 0, no need to process"),
        HCCL_SUCCESS);

    CHK_PRT_RET(
        templateRankSize_ == 0, HCCL_ERROR("[ReduceAicpuReduceNHR] rankSize is 0"), HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(root_ == UINT32_MAX, HCCL_ERROR("[ReduceAicpuReduceNHR] root is invalid"), HcclResult::HCCL_E_INTERNAL);

    dataType_ = param.DataDes.dataType;
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataTypeSize == 0, HCCL_ERROR("[ReduceAicpuReduceNHR] dataTypeSize is 0"), HcclResult::HCCL_E_INTERNAL);

    buffInfo_ = tempAlgParams.buffInfo;

    bool isPcieProtocol = IsPcieProtocol(templateResource.channels); // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocol;                                     // 是否使用Read模式
    HCCL_DEBUG("[ReduceAicpuReduceNHR] Use Dma Read[%d]", isDmaRead_);

    CHK_RET(GetAlgRank(myRank_, myIdx_));
    CHK_PRT_RET(
        myIdx_ >= templateRankSize_,
        HCCL_ERROR(
            "[ReduceAicpuReduceNHR] rank idx[%u] in virtRankMap is invalid, it should be less than rankSize[%u]",
            myIdx_, templateRankSize_),
        HcclResult::HCCL_E_INTERNAL);
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    HCCL_INFO(
        "[ReduceAicpuReduceNHR][KernelRun] sliceSize: %u, count_: %u, typeSize: %u", tempAlgParams.sliceSize, count_,
        dataTypeSize);

    const std::map<u32, std::vector<ChannelInfo>>& channels = templateResource.channels;
    HCCL_DEBUG(
        "[ReduceAicpuReduceNHR][Kernel Run] myRank_[%u], channels.size():%zu, channels first[%u]", myRank_,
        channels.size(), channels.begin()->first);

    // 多通道: 运行期刷新统一按 Min 口径(与 CalcRes 同款)。executor 侧 SetchannelsPerRank 按 Max 刷新
    if (!templateResource.channels.empty()) {
        channelsPerRank_ = CalcChannelsPerRankMin(templateResource.channels);
    }
    threadNum_ = GetThreadNum();
    CHK_PRT_RET(
        templateResource.threads.size() < threadNum_,
        HCCL_ERROR(
            "[ReduceAicpuReduceNHR] threads.size()[%zu] < threadNum_[%u]", templateResource.threads.size(), threadNum_),
        HcclResult::HCCL_E_INTERNAL);
    std::vector<u64> elemCountOut;
    CHK_RET(CalcDataSplitByPortGroup(
        tempAlgParams.sliceSize / dataTypeSize, dataTypeSize, channels.begin()->second, elemCountOut, sizeOut_,
        elemOffset_));
    HCCL_INFO("[ReduceAicpuReduceNHR] channelsPerRank_[%u] threadNum_[%u]", channelsPerRank_, threadNum_);

    // 1. 前同步: 纯起跑号令。前/后拷贝与通信均按通道在自己的队列上按序入队, 无跨队列数据依赖
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    // 2. 切片(slot 布局, 各通道共用) + 每通道前拷贝: userIn 子区 -> 本rank slot(myIdx_) 子区, 由本通道队列自搬
    CHK_RET(CalcSlice(tempAlgParams.sliceSize));
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(PreCopy(tempAlgParams, templateResource.threads, channelIdx));
    }

    // 3. 每通道 AllGather: 对端第 channelIdx 条链路 + 队列 channelIdx, 在本通道子区内跑完整 NHR gather
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_PRT_RET(
            channelIdx >= sizeOut_.size() || channelIdx >= elemOffset_.size(),
            HCCL_ERROR(
                "[ReduceAicpuReduceNHR] channelIdx[%u] out of bounds, sizeOutSize[%zu]", channelIdx, sizeOut_.size()),
            HcclResult::HCCL_E_INTERNAL);
        if (sizeOut_[channelIdx] == 0) {
            continue; // 小消息非均匀切分可能产生空通道
        }
        CHK_RET(RunGather(templateResource.threads, channels, channelIdx));
    }

    // 4. root 规约: 初始拷贝(每通道队列自搬) -> 批次切换 End/Start 一次 + 逐队列 Join
    if (myRank_ == root_) {
        for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
            if (sizeOut_[channelIdx] == 0) {
                continue;
            }
            CHK_RET(RunReduceInitCopy(tempAlgParams, templateResource.threads, channelIdx));
        }
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
        for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
            CHK_RET(static_cast<HcclResult>(HcommThreadJoin(templateResource.threads[channelIdx], CUSTOM_TIMEOUT)));
        }
        for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
            if (sizeOut_[channelIdx] == 0) {
                continue;
            }
            CHK_RET(RunReduceLocal(templateResource.threads, tempAlgParams, channelIdx));
        }
    }

    // 5. 后同步: 等全部通道收尾(含各自队列上的规约)后, 主线程才算本算子完成
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[ReduceAicpuReduceNHR] rank[%d] KernelRun finished", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: 将数据按照rank切分为chunk 块，给后续的reduce操作使用
 * param: dataSize: 待处理的输入数据大小
 * return: sliceInfoVec: 存储数据切分结果
 * return: HcclResult
 */
HcclResult ReduceAicpuReduceNHR::CalcSlice(u64 chunkSize)
{
    HCCL_INFO("[ReduceAicpuReduceNHR] chunkSize[%llu] CalcSlice start", chunkSize);
    // 按 rank 切分数据（与 AllReduceAicpuReduceNHR 保持一致）
    sliceInfoVec_ = RankSliceInfo(templateRankSize_);

    u64 unitAlignSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 dataSize = chunkSize * templateRankSize_;

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; rankIdx++) {
        u64 currChunkSize = std::min<u64>(dataSize - accumOff, chunkSize);
        sliceInfoVec_[rankIdx].emplace_back(SliceInfo{accumOff, currChunkSize});
        HCCL_DEBUG(
            "[ReduceAicpuReduceNHR] rankIdx [%d] CalcSlice accumOff[%llu], currChunkSize[%llu]", rankIdx, accumOff,
            currChunkSize);
        accumOff += currChunkSize;
    }

    CHK_PRT_RET(
        (sliceInfoVec_[templateRankSize_ - 1][0].offset + sliceInfoVec_[templateRankSize_ - 1][0].size != dataSize),
        HCCL_ERROR(
            "[ReduceAicpuReduceNHR] chunkSize:[%llu], Rank:[%d], SliceInfo calculation error!", chunkSize, myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult ReduceAicpuReduceNHR::PreCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    HCCL_INFO("[ReduceAicpuReduceNHR][PreCopy] channelIdx[%u] start", channelIdx);
    const BuffInfo& buffInfo = tempAlgParams.buffInfo;

    CHK_PRT_RET(
        channelIdx >= sizeOut_.size() || channelIdx >= elemOffset_.size(),
        HCCL_ERROR(
            "[ReduceAicpuReduceNHR][PreCopy] channelIdx[%u] out of bounds, sizeOutSize[%zu]", channelIdx,
            sizeOut_.size()),
        HcclResult::HCCL_E_INTERNAL);
    const u64 chSize = sizeOut_[channelIdx];
    const u64 chOff = elemOffset_[channelIdx];
    if (chSize == 0) {
        return HcclResult::HCCL_SUCCESS;
    }

    // 本通道子区: userIn 子区 -> 本rank slot(myIdx_) 子区, 由本通道队列自搬
    const DataSlice srcSlice(buffInfo.inputPtr, buffInfo.inBuffBaseOff + chOff, chSize);
    const DataSlice dstSlice(
        buffInfo.hcclBuff.addr, buffInfo.hcclBuffBaseOff + myIdx_ * tempAlgParams.sliceSize + chOff, chSize);
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads[channelIdx], srcSlice, dstSlice)));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult ReduceAicpuReduceNHR::RunGather(
    const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels, u32 channelIdx)
{
    HCCL_INFO("[ReduceAicpuReduceNHR][RunGather] channelIdx[%u] start", channelIdx);
    ThreadHandle q = threads[channelIdx];
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    const u64 chSize = sizeOut_[channelIdx]; // 调用方保证非空通道

    u32 nSteps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));

        const std::vector<ChannelInfo>& recvChannels = channels.at(subCommRanks_.at(0).at(stepInfo.fromRank));
        const std::vector<ChannelInfo>& sendChannels = channels.at(subCommRanks_.at(0).at(stepInfo.toRank));
        CHK_PRT_RET(
            channelIdx >= recvChannels.size() || channelIdx >= sendChannels.size(),
            HCCL_ERROR(
                "[ReduceAicpuReduceNHR][RunGather] link missing: channelIdx[%u] recvSize[%zu] sendSize[%zu]",
                channelIdx, recvChannels.size(), sendChannels.size()),
            HcclResult::HCCL_E_INTERNAL);
        const ChannelInfo& channelRecv = recvChannels.at(channelIdx);
        const ChannelInfo& channelSend = sendChannels.at(channelIdx);

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        // 多通道: 每个 slice 只搬本通道子区(offset 叠加 elemOffset_[channelIdx], size 取 sizeOut_[channelIdx])
        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            u64 txOffset = sliceInfoVec_[stepInfo.txSliceIdxs[i]][0].offset + elemOffset_[channelIdx]
                           + buffInfo_.hcclBuffBaseOff;
            u64 rxOffset = sliceInfoVec_[stepInfo.rxSliceIdxs[i]][0].offset + elemOffset_[channelIdx]
                           + buffInfo_.hcclBuffBaseOff;

            txSrcSlices.push_back(DataSlice(buffInfo_.hcclBuff.addr, txOffset, chSize, chSize / dataTypeSize));
            txDstSlices.push_back(DataSlice(channelSend.remoteCclMem.addr, txOffset, chSize, chSize / dataTypeSize));
            rxSrcSlices.push_back(DataSlice(channelRecv.remoteCclMem.addr, rxOffset, chSize, chSize / dataTypeSize));
            rxDstSlices.push_back(DataSlice(buffInfo_.hcclBuff.addr, rxOffset, chSize, chSize / dataTypeSize));
        }

        TxRxChannels sendRecvChannels(channelSend, channelRecv);
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});

        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
        if (isDmaRead_) {
            CHK_PRT_RET(
                SendRecvRead(sendRecvInfo, q), HCCL_ERROR("[ReduceAicpuReduceNHR] RunGather send/recv failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                SendRecvWrite(sendRecvInfo, q), HCCL_ERROR("[ReduceAicpuReduceNHR] RunGather send/recv failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
    }

    HCCL_INFO("[ReduceAicpuReduceNHR][RunGather] channelIdx[%u] end", channelIdx);
    return HcclResult::HCCL_SUCCESS;
}

// root 串行规约的初始拷贝: slot0 本通道子区 -> userOut 本通道子区(队列内按序, 作为后续规约的初值)
HcclResult ReduceAicpuReduceNHR::RunReduceInitCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    HCCL_INFO("[ReduceAicpuReduceNHR][RunReduceInitCopy] channelIdx[%u] start", channelIdx);
    const BuffInfo& buffInfo = tempAlgParams.buffInfo;
    const u64 chSize = sizeOut_[channelIdx];
    const u64 chOff = elemOffset_[channelIdx];
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    const DataSlice srcSlice(buffInfo.hcclBuff.addr, buffInfo.hcclBuffBaseOff + chOff, chSize, chSize / dataTypeSize);
    const DataSlice dstSlice(buffInfo.outputPtr, buffInfo.outBuffBaseOff + chOff, chSize, chSize / dataTypeSize);
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads[channelIdx], srcSlice, dstSlice)));

    return HcclResult::HCCL_SUCCESS;
}

// root 串行 AICPU 规约: 其余各 slot 的本通道子区依次累加到 userOut 本通道子区(每通道队列内按序执行)
HcclResult ReduceAicpuReduceNHR::RunReduceLocal(
    const std::vector<ThreadHandle>& threads, const TemplateDataParams& tempAlgParams, u32 channelIdx)
{
    HCCL_INFO("[ReduceAicpuReduceNHR][RunReduceLocal] channelIdx[%u] start", channelIdx);
    const BuffInfo& buffInfo = tempAlgParams.buffInfo;
    const u64 sliceSize = tempAlgParams.sliceSize;
    const u64 chSize = sizeOut_[channelIdx];
    const u64 chOff = elemOffset_[channelIdx];
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    const DataSlice dstSlice(buffInfo.outputPtr, buffInfo.outBuffBaseOff + chOff, chSize, chSize / dataTypeSize);
    for (u32 idx = 1; idx < subCommRanks_.at(0).size(); ++idx) {
        const DataSlice srcSlice(
            buffInfo.hcclBuff.addr, buffInfo.hcclBuffBaseOff + sliceSize * idx + chOff, chSize, chSize / dataTypeSize);

        CHK_RET(static_cast<HcclResult>(LocalReduce(threads[channelIdx], srcSlice, dstSlice, dataType_, reduceOp_)));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult ReduceAicpuReduceNHR::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo)
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

        HCCL_DEBUG(
            "[ReduceAicpuReduceNHR][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

//  计算每轮收发的对端以及slice编号
HcclResult ReduceAicpuReduceNHR::GetStepInfoList(std::vector<AicpuNHRStepInfo>& stepInfoList)
{
    // 将本 rank 号转换成算法使用的索引号
    u32 rankIdx = myIdx_;
    stepInfoList.clear();

    u32 nSteps = GetNHRStepNum(templateRankSize_);
    stepInfoList.resize(nSteps);
    for (u32 step = 0; step < nSteps; step++) {
        // 计算通信对象
        u32 deltaRank = 1 << step;
        u32 sendTo = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
        u32 recvFrom = (rankIdx + deltaRank) % templateRankSize_;

        // 数据份数和数据编号增量
        u32 nSlices = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
        u32 deltaSliceIndex = 1 << (step + 1);
        u32 txSliceIdx = sendTo;
        u32 rxSliceIdx = rankIdx;

        AicpuNHRStepInfo& currStepInfo = stepInfoList[step];
        currStepInfo.step = step;
        currStepInfo.myRank = rankIdx;
        currStepInfo.nSlices = nSlices;
        currStepInfo.toRank = sendTo;
        currStepInfo.fromRank = recvFrom;

        // 计算本rank在每轮收/发中的slice编号
        currStepInfo.txSliceIdxs.reserve(nSlices);
        currStepInfo.rxSliceIdxs.reserve(nSlices);
        for (u32 i = 0; i < nSlices; i++) {
            currStepInfo.txSliceIdxs.push_back(txSliceIdx);
            currStepInfo.rxSliceIdxs.push_back(rxSliceIdx);
            HCCL_DEBUG(
                "[ReduceAicpuReduceNHR][GetStepInfoList] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx,
                rxSliceIdx);
            txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
            rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult ReduceAicpuReduceNHR::GetAlgRank(u32 rank, u32& algRank) const
{
    CHK_PRT_RET(
        subCommRanks_.empty() || subCommRanks_[0].empty(),
        HCCL_ERROR("[ReduceAicpuReduceNHR] subCommRanks_[0] is empty"), HcclResult::HCCL_E_INTERNAL);
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), rank);
    CHK_PRT_RET(
        iter == subCommRanks_[0].end(),
        HCCL_ERROR("[ReduceAicpuReduceNHR] Cannot find rank[%u] in subCommRanks_[0]", rank),
        HcclResult::HCCL_E_INTERNAL);
    algRank = static_cast<u32>(std::distance(subCommRanks_[0].begin(), iter));
    return HcclResult::HCCL_SUCCESS;
}

void ReduceAicpuReduceNHR::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = 1 * channelsPerRank_;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void ReduceAicpuReduceNHR::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = 1 * channelsPerRank_;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

u64 ReduceAicpuReduceNHR::GetThreadNum() const { return 1 * channelsPerRank_; }

} // namespace ops_hccl
