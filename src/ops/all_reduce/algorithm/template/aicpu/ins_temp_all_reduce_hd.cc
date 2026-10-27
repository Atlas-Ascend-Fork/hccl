/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_reduce_hd.h"

namespace ops_hccl {

// C/D 项经验常数：冻结 spec ar-hd-aicpu-20261016 §8（与 InsTempAllReduceNHR 实测经验值对齐，
// pairwise a73c637f 同款命名常量先例）；HD 为 preferred-only（仅显式配置可选，不参与默认成本
// 竞争），常数值仅影响 estimate 输出，未标定不作为性能结论
constexpr auto HD_KERNEL_LATENCY_COEFF = 4e-6f; // C 项：kernelNum 个 kernel 的展开时延
constexpr auto HD_TASK_LAUNCH_COEFF = 1e-6f;    // D 项：taskNum 个 task 的下发时延

InsTempAllReduceHD::InsTempAllReduceHD(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempAllReduceHD::~InsTempAllReduceHD() {}

std::vector<CostModelParam> InsTempAllReduceHD::CalcCostCoeff(CalcCostCoeffParam param)
{
    // HD 调度要求 rank 数为 2 的整数次幂（主门控在 TopoMatchOneLevel），非 2^N 不参与选路
    if (param.rankSize == 0 || (param.rankSize & (param.rankSize - 1)) != 0) {
        return {};
    }
    CommTopo netType = param.netType;
    int portNum = (param.netType == CommTopo::COMM_TOPO_CLOS && param.portNum.size() >= 2) ?
                      (param.portNum[0] + param.portNum[1]) :
                      param.portNum[0];
    int kernelNum = 30;
    int log2R = 0;
    for (u32 r = param.rankSize; r > 1; r >>= 1) {
        log2R++;
    }
    // 传输+主从流同步+两次本地拷贝：HD 共 2m 步收发 + 收尾
    int taskNum = CostModelManager::CalcTransTaskNum((2 * log2R + 1)) * 4
                  + CostModelManager::CalcSyncTaskNum((2 * log2R + 1)) * 12 + 2;
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    // HD 与 NHR 每 rank 总传输量同为 2(N-1)/N*S，RS+AG 两程 dataRatio 乘 2
    CostModelManager::Global()->CalcNHRParams(param.dataRatio * 2, netType, portNum, param.rankSize, A, param.isPod);
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio * param.rankSize * 2, EngineType::AICPU, B);
    } else {
        B = 0.0f;
    }
    // hd实测和理论估计相差较大，先用经验值
    C = HD_KERNEL_LATENCY_COEFF * kernelNum;
    D = HD_TASK_LAUNCH_COEFF * taskNum;
    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    HCCL_DEBUG("[%s] CalcCostCoeff A=%f B=%f C=%f D=%f.", __func__, A, B, C, D);
    return params;
}

u64 InsTempAllReduceHD::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u32 multiple = 1;
    return multiple;
}

HcclResult InsTempAllReduceHD::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> level1Channels;
    bool isUBX = topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix;
    if (isUBX) {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestNhrMultiJetty(comm, param, topoInfo, subCommRanks_, myChannelDescs));
        for (auto channel : myChannelDescs) {
            if (channel.channelProtocol == COMM_PROTOCOL_UB_CTP) {
                level1Channels.push_back(channel);
            }
        }
        HCCL_DEBUG("[InsTempAllReduceHD::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    }
    HCCL_DEBUG("[InsTempAllReduceHD][CalcRes] level1Channels.size() is %zu.", level1Channels.size());
    resourceRequest.channels.push_back(level1Channels);
    channelsPerRank_ = CalcChannelsPerRank(level1Channels);
    if (isUBX && channelsPerRank_ > MAX_JETTY_NUM) {
        HCCL_ERROR(
            "[InsTempAllReduceHD][CalcRes] channelsPerRank_ %u is greater than MAX_JETTY_NUM %u.", channelsPerRank_,
            MAX_JETTY_NUM);
    } else {
        HCCL_DEBUG("[InsTempAllReduceHD][CalcRes] channelsPerRank_ is %u.", channelsPerRank_);
    }
    GetRes(resourceRequest);

    HCCL_INFO("[InsTempAllReduceHD] Calculate resource finished.");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceHD::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 threadNum = channelsPerRank_;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAllReduceHD::GetThreadNum() const { return channelsPerRank_; }

void InsTempAllReduceHD::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 slaveThreadNum = GetThreadNum() - 1;
    for (u32 i = 0; i < slaveThreadNum; i++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempAllReduceHD::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 notifyNum = GetThreadNum() - 1;
    for (u32 i = 0; i < notifyNum; i++) {
        notifyIdxSubToMain.push_back(i);
    }
}

HcclResult InsTempAllReduceHD::ValidateChannels(const std::map<u32, std::vector<ChannelInfo>>& channels) const
{
    CHK_PRT_RET(
        channelsPerRank_ == 0, HCCL_ERROR("[InsTempAllReduceHD][ValidateChannels] channelsPerRank is zero."),
        HcclResult::HCCL_E_INTERNAL);
    for (const auto& channelsByRank : channels) {
        CHK_PRT_RET(
            channelsByRank.second.size() < channelsPerRank_,
            HCCL_ERROR(
                "[InsTempAllReduceHD][ValidateChannels] remoteRank[%u] has [%zu] channels, expected at least [%u].",
                channelsByRank.first, channelsByRank.second.size(), channelsPerRank_),
            HcclResult::HCCL_E_INTERNAL);
    }
    // RS 折半、AG 倍增及最后一步 Read 使用相同的一组 XOR 伙伴，在任何任务下发前统一检查。
    for (u32 distance = templateRankSize_ >> 1; distance > 0; distance >>= 1) {
        u32 remoteRank = rankList_.at(myRankIdx_ ^ distance);
        CHK_PRT_RET(
            channels.count(remoteRank) == 0,
            HCCL_ERROR("[InsTempAllReduceHD][ValidateChannels] missing HD partner remoteRank[%u].", remoteRank),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceHD::PrepareDataSplitForMultiChannel(const TemplateResource& templateResource)
{
    CHK_PRT_RET(
        templateResource.channels.empty() || templateResource.channels.begin()->second.empty(),
        HCCL_ERROR("[InsTempAllReduceHD][PrepareDataSplitForMultiChannel] channels is empty."), HCCL_E_INTERNAL);
    std::vector<u64> elemCountOut;
    u64 totalDataCount = sliceSize_ / dataTypeSize_;
    CHK_RET(CalcDataSplitByPortGroup(
        totalDataCount, dataTypeSize_, templateResource.channels.begin()->second, elemCountOut, dataSplit_,
        dataOffset_));
    if (tailSize_ != sliceSize_ && tailSize_ > 0) {
        std::vector<u64> elemCountOutTail;
        u64 totalDataCountTail = tailSize_ / dataTypeSize_;
        CHK_RET(CalcDataSplitByPortGroup(
            totalDataCountTail, dataTypeSize_, templateResource.channels.begin()->second, elemCountOutTail,
            dataSplitTail_, dataOffsetTail_));
    } else {
        dataSplitTail_ = dataSplit_;
        dataOffsetTail_ = dataOffset_;
    }
    CHK_PRT_RET(
        dataSplit_.size() < channelsPerRank_ || dataOffset_.size() < channelsPerRank_
            || dataSplitTail_.size() < channelsPerRank_ || dataOffsetTail_.size() < channelsPerRank_,
        HCCL_ERROR(
            "[InsTempAllReduceHD][PrepareDataSplitForMultiChannel] split sizes[%zu,%zu,%zu,%zu] are insufficient "
            "for channelsPerRank[%u].",
            dataSplit_.size(), dataOffset_.size(), dataSplitTail_.size(), dataOffsetTail_.size(), channelsPerRank_),
        HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempAllReduceHD::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllReduceHD] KernelRun Start.");

    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        HCCL_INFO("[InsTempAllReduceHD] Rank [%u], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        subCommRanks_.size() == 0, HCCL_ERROR("[InsTempAllReduceHD][KernelRun] subCommRanks is empty."),
        HcclResult::HCCL_E_INTERNAL);
    rankList_ = subCommRanks_.at(0);
    CHK_PRT_RET(
        rankList_.size() != templateRankSize_,
        HCCL_ERROR("[InsTempAllReduceHD][KernelRun] rank count is invalid in rank list.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    // 防御性校验：HD 调度要求 rank 数为 2 的整数次幂（主门控在 TopoMatchOneLevel）
    CHK_PRT_RET(
        (templateRankSize_ & (templateRankSize_ - 1)) != 0,
        HCCL_ERROR("[InsTempAllReduceHD][KernelRun] templateRankSize_[%u] is not a power of two.", templateRankSize_),
        HcclResult::HCCL_E_NOT_SUPPORT);

    // 获取当前rank在rank列表中的序号
    CHK_RET(GetAlgRank(myRank_, rankList_, myRankIdx_));

    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    tempAlgParams_ = tempAlgParams;
    supportSymmetricMemAccess_ = param.supportSymmetricMemory;

    // 防御性校验：sliceSize 应为 count * dataTypeSize，不一致时返回错误防止上游语义漂移被静默吞掉
    CHK_PRT_RET(
        processSize_ != count_ * dataTypeSize_,
        HCCL_ERROR(
            "[InsTempAllReduceHD][KernelRun] processSize_[%llu] != count_[%llu] * dataTypeSize_[%llu], "
            "upstream sliceSize is inconsistent.",
            processSize_, count_, dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);

    CHK_RET(ValidateChannels(templateResource.channels));
    bool isPcieProtocol = IsPcieProtocol(templateResource.channels); // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocol;                                     // 是否使用Read模式
    HCCL_DEBUG("[InsTempAllReduceHD] Use Dma Read[%d]", isDmaRead_);

    // 仅在非 DMA Read、非对称内存、rank 数大于 1 时启用首步 tx 跳过 PreCopy 优化
    skipStep0TxPreCopy_ = (!supportSymmetricMemAccess_) && (!isDmaRead_) && (templateRankSize_ > 1);
    step0TxSliceIdxs_.clear();
    if (skipStep0TxPreCopy_) {
        HCCL_INFO("[InsTempAllReduceHD] Skip step0 tx pre-copy enabled");
    }

    if (count_ == 0) {
        HCCL_WARNING("[InsTempAllReduceHD][KernelRun] data count is 0.");
        return HcclResult::HCCL_SUCCESS;
    }

    // 数据切片：向下取整，前N-1个rank各拿sliceSize_，最后rank拿tailSize_
    sliceSize_ = (count_ / templateRankSize_) * dataTypeSize_;
    tailSize_ = count_ * dataTypeSize_ - sliceSize_ * (templateRankSize_ - 1);

    CHK_RET(PrepareDataSplitForMultiChannel(templateResource));
    threadNum_ = GetThreadNum();
    CHK_PRT_RET(
        threadNum_ > templateResource.threads.size(),
        HCCL_ERROR(
            "[InsTempAllReduceHD][KernelRun] thread num[%u] more than threads size[%zu].", threadNum_,
            templateResource.threads.size()),
        HcclResult::HCCL_E_INTERNAL);

    // 预计算 ReduceScatter 和 AllGather 步骤信息，供后续复用
    CHK_RET(GetReduceScatterStepInfoList(reduceScatterSteps_));
    CHK_RET(GetAllGatherStepInfoList(allGatherSteps_));
    // 收集首步 tx slice 编号，供 PreCopy 跳过和 RunReduceScatterHD 直发 input 使用
    if (skipStep0TxPreCopy_ && !reduceScatterSteps_.empty()) {
        for (u32 txIdx : reduceScatterSteps_.front().txSliceIdxs) {
            step0TxSliceIdxs_.insert(txIdx);
        }
    }
    // AllGather 最后一步可优化为 Read 模式，远端数据直接读入 output buffer（对称内存路径不支持）
    readLastStepToOutput_ = (!supportSymmetricMemAccess_) && CanReadLastStepToOutput();
    HCCL_DEBUG("[InsTempAllReduceHD] Read last step to output[%d]", readLastStepToOutput_);

    // 前同步：通知子线程开始 PreCopy
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    // 将数据从input拷贝到hcclBuffer上，多 channel 并发（对称内存路径跳过）
    if (!supportSymmetricMemAccess_) {
        for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
            CHK_RET(PreCopy(tempAlgParams, templateResource.threads, channelIdx));
        }
    }
    // 单 rank：all-reduce 退化为 input→output 的直接拷贝，跳过通信步骤
    if (templateRankSize_ <= 1) {
        for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
            CHK_RET(PostCopy(tempAlgParams, templateResource.threads, channelIdx));
        }
        if (threadNum_ > 1) {
            std::vector<ThreadHandle> subThreads(
                templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
            GetNotifyIdxSubToMain(notifyIdxSubToMain_);
            CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
        }
        return HcclResult::HCCL_SUCCESS;
    }
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        // TwoShot算法，第一步ReduceScatter（折半）
        CHK_RET(RunReduceScatterHD(tempAlgParams, templateResource.channels, templateResource.threads, channelIdx));
    }
    // 对称内存路径：RS 完成后需等待所有线程完成归约，再拷贝 input[mySlice] 到 output[mySlice]
    if (supportSymmetricMemAccess_ && threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    if (supportSymmetricMemAccess_) {
        u64 mySliceSize = (myRankIdx_ == templateRankSize_ - 1) ? tailSize_ : sliceSize_;
        u64 mySliceOffset = myRankIdx_ * sliceSize_;
        DataSlice copySrcSlice(
            tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff + mySliceOffset, mySliceSize,
            mySliceSize / dataTypeSize_);
        DataSlice copyDstSlice(
            tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff + mySliceOffset, mySliceSize,
            mySliceSize / dataTypeSize_);
        CHK_RET(LocalCopy(templateResource.threads[0], copySrcSlice, copyDstSlice));
    }
    if (supportSymmetricMemAccess_ && threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        // TwoShot算法，第二步AllGather（倍增）
        CHK_RET(RunAllGatherHD(tempAlgParams, templateResource.channels, templateResource.threads, channelIdx));
    }
    // 将数据从hcclBuffer上拷贝到output上，多 channel 并发（对称内存路径跳过）
    if (!supportSymmetricMemAccess_) {
        for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
            CHK_RET(PostCopy(tempAlgParams, templateResource.threads, channelIdx));
        }
    }
    // 后同步：等待所有 channel 的 PreCopy/通信/PostCopy 完成后通知主线程
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[InsTempAllReduceHD] KernelRun finished.");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceHD::PreCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads, u32 channelIdx) const
{
    HCCL_INFO("[InsTempAllReduceHD] PreCopy data from input to hccl buffer, channelIdx[%u]", channelIdx);

    void* localInBuffPtr = tempAlgParams.buffInfo.inputPtr;
    void* localHcclBuffPtr = tempAlgParams.buffInfo.hcclBuff.addr;
    u64 inBuffBaseOffset = tempAlgParams.buffInfo.inBuffBaseOff;
    u64 hcclBuffBaseOffset = tempAlgParams.buffInfo.hcclBuffBaseOff;

    // templateRankSize_ == 1 时无 ReduceScatter 步骤，跳过 PreCopy，
    // PostCopy 直接从 input 拷贝到 output，避免 input→cclBuff→output 的冗余搬运
    if (reduceScatterSteps_.empty()) {
        return HcclResult::HCCL_SUCCESS;
    }

    // 如果 input 与 cclBuff 基地址相同且 base 偏移相同（物理同一块内存），跳过整个 PreCopy
    if (localInBuffPtr == localHcclBuffPtr && inBuffBaseOffset == hcclBuffBaseOffset) {
        HCCL_INFO("[InsTempAllReduceHD] PreCopy skip because input is scratch");
        return HcclResult::HCCL_SUCCESS;
    }

    // ReduceScatter 第一步 txSliceIdxs 中的块将通过 WriteReduce 直接从 input 写入远端 cclBuff，
    // 无需 pre-copy 到本地 cclBuff；其余块需要拷贝到本地 cclBuff，供后续步骤使用
    // 仅在 skipStep0TxPreCopy_ 启用时跳过（DMA Read 路径下远端从 cclBuff 读取，必须 pre-copy）
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; ++rankIdx) {
        if (skipStep0TxPreCopy_ && step0TxSliceIdxs_.count(rankIdx) > 0) {
            continue;
        }
        u64 txSz = (rankIdx == templateRankSize_ - 1) ? dataSplitTail_[channelIdx] : dataSplit_[channelIdx];
        u64 txOff = (rankIdx == templateRankSize_ - 1) ? dataOffsetTail_[channelIdx] : dataOffset_[channelIdx];
        u64 inOff = inBuffBaseOffset + rankIdx * sliceSize_ + txOff;
        u64 scOff = hcclBuffBaseOffset + rankIdx * sliceSize_ + txOff;
        DataSlice copySrcSlice(localInBuffPtr, inOff, txSz, txSz / dataTypeSize_);
        DataSlice copyDstSlice(localHcclBuffPtr, scOff, txSz, txSz / dataTypeSize_);
        CHK_RET(LocalCopy(threads.at(channelIdx), copySrcSlice, copyDstSlice));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceHD::RunReduceScatterHD(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    void* localHcclBuffPtr
        = supportSymmetricMemAccess_ ? tempAlgParams.buffInfo.inputPtr : tempAlgParams.buffInfo.hcclBuff.addr;
    void* localInBuffPtr = tempAlgParams.buffInfo.inputPtr;
    u64 hcclBuffBaseOffset
        = supportSymmetricMemAccess_ ? tempAlgParams.buffInfo.inBuffBaseOff : tempAlgParams.buffInfo.hcclBuffBaseOff;
    u64 inBuffBaseOffset = tempAlgParams.buffInfo.inBuffBaseOff;

    for (u32 stepIdx = 0; stepIdx < reduceScatterSteps_.size(); ++stepIdx) {
        auto& stepInfo = reduceScatterSteps_[stepIdx];
        const bool isFirstStep = (stepIdx == 0);

        CHK_PRT_RET(
            channels.count(rankList_.at(stepInfo.fromRank)) == 0,
            HCCL_ERROR(
                "[InsTempAllReduceHD][RunReduceScatterHD] remoteRank[%u] is not in channels.",
                rankList_.at(stepInfo.fromRank)),
            HcclResult::HCCL_E_INTERNAL);
        CHK_PRT_RET(
            channels.count(rankList_.at(stepInfo.toRank)) == 0,
            HCCL_ERROR(
                "[InsTempAllReduceHD][RunReduceScatterHD] remoteRank[%u] is not in channels.",
                rankList_.at(stepInfo.toRank)),
            HcclResult::HCCL_E_INTERNAL);

        const ChannelInfo& recvChannel = channels.at(rankList_.at(stepInfo.fromRank)).at(channelIdx);
        const ChannelInfo& sendChannel = channels.at(rankList_.at(stepInfo.toRank)).at(channelIdx);
        std::vector<DataSlice> recvSrcSlicesList;
        std::vector<DataSlice> sendSrcSlicesList;
        std::vector<DataSlice> recvDstSlicesList;
        std::vector<DataSlice> sendDstSlicesList;

        void* sendRemoteHcclBuffPtr
            = supportSymmetricMemAccess_ ? sendChannel.remoteInputGraphMode.addr : sendChannel.remoteCclMem.addr;
        void* recvRemoteHcclBuffPtr
            = supportSymmetricMemAccess_ ? recvChannel.remoteInputGraphMode.addr : recvChannel.remoteCclMem.addr;
        // 第一步 tx 从 input 直接 WriteReduce 到远端 cclBuff；后续步骤从本地 cclBuff 发送
        // 对称内存路径下 localHcclBuffPtr 已指向 input，无需区分首步
        // DMA Read 路径下 skipStep0TxPreCopy_ 为 false，首步从 cclBuff 发送
        void* sendSrcPtr = (skipStep0TxPreCopy_ && isFirstStep) ? localInBuffPtr : localHcclBuffPtr;
        u64 sendSrcBaseOffset = (skipStep0TxPreCopy_ && isFirstStep) ? inBuffBaseOffset : hcclBuffBaseOffset;

        // 在 cclBuff 块视图上进行 HD 折半归约：发送补半区段，保留含本 rank 的半区段并对端归约
        for (u32 idx = 0; idx < stepInfo.nSlices; ++idx) {
            u32 txIdx = stepInfo.txSliceIdxs.at(idx);
            u32 rxIdx = stepInfo.rxSliceIdxs.at(idx);
            u64 txOff = (txIdx == templateRankSize_ - 1) ? dataOffsetTail_[channelIdx] : dataOffset_[channelIdx];
            u64 txSz = (txIdx == templateRankSize_ - 1) ? dataSplitTail_[channelIdx] : dataSplit_[channelIdx];
            u64 rxOff = (rxIdx == templateRankSize_ - 1) ? dataOffsetTail_[channelIdx] : dataOffset_[channelIdx];
            u64 rxSz = (rxIdx == templateRankSize_ - 1) ? dataSplitTail_[channelIdx] : dataSplit_[channelIdx];
            sendSrcSlicesList.emplace_back(
                sendSrcPtr, sendSrcBaseOffset + txIdx * sliceSize_ + txOff, txSz, txSz / dataTypeSize_);
            sendDstSlicesList.emplace_back(
                sendRemoteHcclBuffPtr, hcclBuffBaseOffset + txIdx * sliceSize_ + txOff, txSz, txSz / dataTypeSize_);
            recvSrcSlicesList.emplace_back(
                recvRemoteHcclBuffPtr, hcclBuffBaseOffset + rxIdx * sliceSize_ + rxOff, rxSz, rxSz / dataTypeSize_);
            recvDstSlicesList.emplace_back(
                localHcclBuffPtr, hcclBuffBaseOffset + rxIdx * sliceSize_ + rxOff, rxSz, rxSz / dataTypeSize_);
        }
        SendRecvReduceInfo sendRecvReduceInfo{
            {sendChannel, recvChannel},
            {{sendSrcSlicesList, sendDstSlicesList}, {recvSrcSlicesList, recvDstSlicesList}},
            dataType_,
            reduceOp_};

        if (supportSymmetricMemAccess_) {
            CHK_PRT_RET(
                SendRecvBatchReadReduce(sendRecvReduceInfo, threads.at(channelIdx)),
                HCCL_ERROR("[InsTempAllReduceHD] RunReduceScatterHD SendRecvBatchReadReduce failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else if (isDmaRead_) {
            CHK_PRT_RET(
                SendRecvReadReduce(sendRecvReduceInfo, threads.at(channelIdx)),
                HCCL_ERROR("[InsTempAllReduceHD] RunReduceScatterHD SendRecvReadReduce failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                SendRecvBatchWriteReduce(sendRecvReduceInfo, threads.at(channelIdx)),
                HCCL_ERROR("[InsTempAllReduceHD] RunReduceScatterHD SendRecvBatchWriteReduce failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceHD::RunAllGatherHD(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    void* localHcclBuffPtr
        = supportSymmetricMemAccess_ ? tempAlgParams.buffInfo.outputPtr : tempAlgParams.buffInfo.hcclBuff.addr;
    u64 hcclBuffBaseOffset
        = supportSymmetricMemAccess_ ? tempAlgParams.buffInfo.outBuffBaseOff : tempAlgParams.buffInfo.hcclBuffBaseOff;

    lastStepReadSliceIdxs_.clear();

    for (u32 stepIdx = 0; stepIdx < allGatherSteps_.size(); ++stepIdx) {
        auto& stepInfo = allGatherSteps_[stepIdx];
        const bool isLastStep = (stepIdx == allGatherSteps_.size() - 1);
        // Read 优化：AllGather 最后一步远端数据直接读入 output buffer，跳过中间 cclBuff 中转
        if (readLastStepToOutput_ && isLastStep) {
            CHK_RET(RunLastStepReadToOutput(tempAlgParams, channels, threads, channelIdx));
            return HcclResult::HCCL_SUCCESS;
        }

        CHK_PRT_RET(
            channels.count(rankList_.at(stepInfo.fromRank)) == 0,
            HCCL_ERROR(
                "[InsTempAllReduceHD][RunAllGatherHD] remoteRank[%u] is not in channels.",
                rankList_.at(stepInfo.fromRank)),
            HcclResult::HCCL_E_INTERNAL);
        CHK_PRT_RET(
            channels.count(rankList_.at(stepInfo.toRank)) == 0,
            HCCL_ERROR(
                "[InsTempAllReduceHD][RunAllGatherHD] remoteRank[%u] is not in channels.",
                rankList_.at(stepInfo.toRank)),
            HcclResult::HCCL_E_INTERNAL);

        const ChannelInfo& recvChannel = channels.at(rankList_.at(stepInfo.fromRank)).at(channelIdx);
        const ChannelInfo& sendChannel = channels.at(rankList_.at(stepInfo.toRank)).at(channelIdx);
        std::vector<DataSlice> sendSrcSlicesList;
        std::vector<DataSlice> sendDstSlicesList;
        std::vector<DataSlice> recvSrcSlicesList;
        std::vector<DataSlice> recvDstSlicesList;

        void* sendRemoteHcclBuffPtr
            = supportSymmetricMemAccess_ ? sendChannel.remoteOutputGraphMode.addr : sendChannel.remoteCclMem.addr;
        void* recvRemoteHcclBuffPtr
            = supportSymmetricMemAccess_ ? recvChannel.remoteOutputGraphMode.addr : recvChannel.remoteCclMem.addr;

        // HD 倍增收集：整段发送当前持有区段，接收补半区段
        for (u32 idx = 0; idx < stepInfo.nSlices; ++idx) {
            u32 txIdx = stepInfo.txSliceIdxs.at(idx);
            u32 rxIdx = stepInfo.rxSliceIdxs.at(idx);
            u64 txOff = (txIdx == templateRankSize_ - 1) ? dataOffsetTail_[channelIdx] : dataOffset_[channelIdx];
            u64 txSz = (txIdx == templateRankSize_ - 1) ? dataSplitTail_[channelIdx] : dataSplit_[channelIdx];
            u64 rxOff = (rxIdx == templateRankSize_ - 1) ? dataOffsetTail_[channelIdx] : dataOffset_[channelIdx];
            u64 rxSz = (rxIdx == templateRankSize_ - 1) ? dataSplitTail_[channelIdx] : dataSplit_[channelIdx];
            sendSrcSlicesList.emplace_back(
                localHcclBuffPtr, hcclBuffBaseOffset + txIdx * sliceSize_ + txOff, txSz, txSz / dataTypeSize_);
            sendDstSlicesList.emplace_back(
                sendRemoteHcclBuffPtr, hcclBuffBaseOffset + txIdx * sliceSize_ + txOff, txSz, txSz / dataTypeSize_);
            recvSrcSlicesList.emplace_back(
                recvRemoteHcclBuffPtr, hcclBuffBaseOffset + rxIdx * sliceSize_ + rxOff, rxSz, rxSz / dataTypeSize_);
            recvDstSlicesList.emplace_back(
                localHcclBuffPtr, hcclBuffBaseOffset + rxIdx * sliceSize_ + rxOff, rxSz, rxSz / dataTypeSize_);
        }
        SendRecvInfo sendRecvInfo{
            {sendChannel, recvChannel},
            {{sendSrcSlicesList, sendDstSlicesList}, {recvSrcSlicesList, recvDstSlicesList}}};

        if (supportSymmetricMemAccess_) {
            CHK_PRT_RET(
                SendRecvBatchRead(sendRecvInfo, threads.at(channelIdx)),
                HCCL_ERROR("[InsTempAllReduceHD] RunAllGatherHD SendRecvBatchRead failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else if (isDmaRead_) {
            CHK_PRT_RET(
                SendRecvRead(sendRecvInfo, threads.at(channelIdx)),
                HCCL_ERROR("[InsTempAllReduceHD] RunAllGatherHD SendRecv failed"), HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                SendRecvBatchWrite(sendRecvInfo, threads.at(channelIdx)),
                HCCL_ERROR("[InsTempAllReduceHD] RunAllGatherHD SendRecv failed"), HcclResult::HCCL_E_INTERNAL);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

bool InsTempAllReduceHD::CanReadLastStepToOutput() const
{
    // 仅在非 DMA Read 路径、output 与 cclBuff 不物理重叠时才能启用 Read 优化
    bool outputIsScratch = tempAlgParams_.buffInfo.outputPtr == tempAlgParams_.buffInfo.hcclBuff.addr
                           && tempAlgParams_.buffInfo.outBuffBaseOff == tempAlgParams_.buffInfo.hcclBuffBaseOff;
    return !isDmaRead_ && tempAlgParams_.buffInfo.outBuffType == BufferType::OUTPUT && !outputIsScratch;
}

bool InsTempAllReduceHD::IsLastStepReadSlice(u32 algRank) const { return lastStepReadSliceIdxs_.count(algRank) > 0; }

HcclResult InsTempAllReduceHD::RunLastStepReadToOutput(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    void* localOutBuffPtr = tempAlgParams.buffInfo.outputPtr;
    u64 hcclBuffBaseOffset = tempAlgParams.buffInfo.hcclBuffBaseOff;
    u64 outBuffBaseOffset = tempAlgParams.buffInfo.outBuffBaseOff;

    auto& lastStepInfo = allGatherSteps_.back();
    CHK_PRT_RET(
        channels.count(rankList_.at(lastStepInfo.fromRank)) == 0,
        HCCL_ERROR(
            "[InsTempAllReduceHD][RunLastStepReadToOutput] remoteRank[%u] is not in channels.",
            rankList_.at(lastStepInfo.fromRank)),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        channels.count(rankList_.at(lastStepInfo.toRank)) == 0,
        HCCL_ERROR(
            "[InsTempAllReduceHD][RunLastStepReadToOutput] remoteRank[%u] is not in channels.",
            rankList_.at(lastStepInfo.toRank)),
        HcclResult::HCCL_E_INTERNAL);

    const ChannelInfo& recvChannel = channels.at(rankList_.at(lastStepInfo.fromRank)).at(channelIdx);
    const ChannelInfo& sendChannel = channels.at(rankList_.at(lastStepInfo.toRank)).at(channelIdx);

    // 对端在其自身的最后一步通过 Read 直接从本 rank 的 cclBuff 拉取这些块，本 rank 无需发送；
    // sendChannel 仅用于 ACK/DATA_SIGNAL 同步
    std::vector<DataSlice> rxSrcSlicesList;
    std::vector<DataSlice> rxDstSlicesList;
    void* recvRemoteHcclBuffPtr = recvChannel.remoteCclMem.addr;

    for (u32 idx = 0; idx < lastStepInfo.nSlices; ++idx) {
        u32 rxIdx = lastStepInfo.rxSliceIdxs.at(idx);
        u64 rxOff = (rxIdx == templateRankSize_ - 1) ? dataOffsetTail_[channelIdx] : dataOffset_[channelIdx];
        u64 rxSz = (rxIdx == templateRankSize_ - 1) ? dataSplitTail_[channelIdx] : dataSplit_[channelIdx];
        u64 rxScratchOff = hcclBuffBaseOffset + rxIdx * sliceSize_ + rxOff;
        u64 rxOutputOff = outBuffBaseOffset + rxIdx * sliceSize_ + rxOff;

        rxSrcSlicesList.emplace_back(recvRemoteHcclBuffPtr, rxScratchOff, rxSz, rxSz / dataTypeSize_);
        rxDstSlicesList.emplace_back(localOutBuffPtr, rxOutputOff, rxSz, rxSz / dataTypeSize_);
        lastStepReadSliceIdxs_.insert(rxIdx);
    }

    // 最后一步使用 Read 模式：远端数据直接读入 output buffer，本 rank 无需发送数据
    const std::vector<DataSlice> emptySlices;
    TxRxSlicesList readSlicesList({emptySlices, emptySlices}, {rxSrcSlicesList, rxDstSlicesList});
    TxRxChannels sendRecvChannels(sendChannel, recvChannel);
    SendRecvInfo readInfo(sendRecvChannels, readSlicesList);

    CHK_PRT_RET(
        SendRecvBatchRead(readInfo, threads.at(channelIdx)),
        HCCL_ERROR("[InsTempAllReduceHD] RunLastStepReadToOutput SendRecvBatchRead failed"),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceHD::PostCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads, u32 channelIdx) const
{
    HCCL_INFO("[InsTempAllReduceHD][PostCopy] Opbase copy from scratchBuffer to userOut, channelIdx[%u]", channelIdx);

    void* localHcclBuffPtr = tempAlgParams.buffInfo.hcclBuff.addr;
    void* localOutBuffPtr = tempAlgParams.buffInfo.outputPtr;
    void* localInBuffPtr = tempAlgParams.buffInfo.inputPtr;
    u64 hcclBuffBaseOffset = tempAlgParams.buffInfo.hcclBuffBaseOff;
    u64 outBuffBaseOffset = tempAlgParams.buffInfo.outBuffBaseOff;
    u64 inBuffBaseOffset = tempAlgParams.buffInfo.inBuffBaseOff;

    // templateRankSize_ == 1 时无通信步骤，PreCopy 已跳过，直接从 input 拷贝到 output
    // 数据未分片，只需 channelIdx==0 拷贝一次
    // 注意：此分支必须在 output==cclBuff 跳过判断之前，因为单 rank 场景下数据在 input 而非 cclBuff
    if (reduceScatterSteps_.empty()) {
        if (channelIdx != 0) {
            return HcclResult::HCCL_SUCCESS;
        }
        DataSlice copySrcSlice(localInBuffPtr, inBuffBaseOffset, processSize_, count_);
        DataSlice copyDstSlice(localOutBuffPtr, outBuffBaseOffset, processSize_, count_);
        CHK_RET(LocalCopy(threads.at(channelIdx), copySrcSlice, copyDstSlice));
        return HcclResult::HCCL_SUCCESS;
    }

    // output 与 cclBuff 基地址相同且 base 偏移相同（物理同一块内存的同一区域）时无需拷贝
    // 仅判地址不够：parallel executor 中 outputPtr==cclMem.addr 但 outBuffBaseOff != hcclBuffBaseOff
    // 仅多 rank 路径生效：单 rank 路径数据在 input，必须走 input→output 拷贝
    if (localOutBuffPtr == localHcclBuffPtr && outBuffBaseOffset == hcclBuffBaseOffset) {
        HCCL_INFO("[InsTempAllReduceHD] PostCopy skip because output is scratch");
        return HcclResult::HCCL_SUCCESS;
    }

    // 按 channel 分片将 cclBuff → output
    // readLastStepToOutput_ 启用时，最后一步已直接读入 output 的块跳过拷贝
    // lastStepReadSliceIdxs_ 为 std::set，IsLastStepReadSlice 为 O(logN) 查找
    for (u32 algRank = 0; algRank < templateRankSize_; ++algRank) {
        if (readLastStepToOutput_ && IsLastStepReadSlice(algRank)) {
            continue;
        }
        u64 rxSz = (algRank == templateRankSize_ - 1) ? dataSplitTail_[channelIdx] : dataSplit_[channelIdx];
        u64 rxOff = (algRank == templateRankSize_ - 1) ? dataOffsetTail_[channelIdx] : dataOffset_[channelIdx];
        u64 scratchOff = hcclBuffBaseOffset + algRank * sliceSize_ + rxOff;
        u64 outOff = outBuffBaseOffset + algRank * sliceSize_ + rxOff;
        DataSlice srcSlice(localHcclBuffPtr, scratchOff, rxSz, rxSz / dataTypeSize_);
        DataSlice dstSlice(localOutBuffPtr, outOff, rxSz, rxSz / dataTypeSize_);
        CHK_RET(LocalCopy(threads.at(channelIdx), srcSlice, dstSlice));
    }
    return HcclResult::HCCL_SUCCESS;
}

// 计算ReduceScatter每步的对端以及块编号：距离递减折半，伙伴 = algRank XOR D，
// 发送补半区段[sentStart, sentStart+D)，保留含本 rank 的半区段[keepStart, keepStart+D)
HcclResult InsTempAllReduceHD::GetReduceScatterStepInfoList(std::vector<AicpuNHRStepInfo>& stepInfoList) const
{
    stepInfoList.clear();

    u32 nSteps = GetHDStepNum();
    stepInfoList.resize(nSteps);

    for (u32 step = 0; step < nSteps; step++) {
        u32 dist = templateRankSize_ >> (step + 1);
        u32 peerIdx = myRankIdx_ ^ dist;
        u32 keepStart = myRankIdx_ & ~(dist - 1);
        u32 sentStart = peerIdx & ~(dist - 1);

        AicpuNHRStepInfo& currStepInfo = stepInfoList[step];
        currStepInfo.step = step;
        currStepInfo.myRank = myRankIdx_;
        currStepInfo.nSlices = dist;
        // HD 收发同伙伴
        currStepInfo.toRank = peerIdx;
        currStepInfo.fromRank = peerIdx;

        currStepInfo.txSliceIdxs.reserve(dist);
        currStepInfo.rxSliceIdxs.reserve(dist);
        for (u32 i = 0; i < dist; i++) {
            currStepInfo.txSliceIdxs.push_back(sentStart + i);
            currStepInfo.rxSliceIdxs.push_back(keepStart + i);
            HCCL_DEBUG(
                "[InsTempAllReduceHD][GetReduceScatterStepInfoList] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i,
                sentStart + i, keepStart + i);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

// 计算AllGather每步的对端以及块编号：距离递增倍增，伙伴 = algRank XOR D，
// 整段发送持有区段[heldStart, heldStart+D)，接收补半区段[heldStart^D, heldStart^D+D)
HcclResult InsTempAllReduceHD::GetAllGatherStepInfoList(std::vector<AicpuNHRStepInfo>& stepInfoList) const
{
    stepInfoList.clear();

    u32 nSteps = GetHDStepNum();
    stepInfoList.resize(nSteps);
    for (u32 step = 0; step < nSteps; step++) {
        u32 dist = 1u << step;
        u32 peerIdx = myRankIdx_ ^ dist;
        u32 heldStart = myRankIdx_ & ~(dist - 1);

        AicpuNHRStepInfo& currStepInfo = stepInfoList[step];
        currStepInfo.step = step;
        currStepInfo.toRank = peerIdx;
        currStepInfo.fromRank = peerIdx;
        currStepInfo.myRank = myRankIdx_;
        currStepInfo.nSlices = dist;

        currStepInfo.txSliceIdxs.reserve(dist);
        currStepInfo.rxSliceIdxs.reserve(dist);
        for (u32 i = 0; i < dist; i++) {
            currStepInfo.txSliceIdxs.push_back(heldStart + i);
            currStepInfo.rxSliceIdxs.push_back((heldStart ^ dist) + i);
            HCCL_DEBUG(
                "[InsTempAllReduceHD][GetAllGatherStepInfoList] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, heldStart + i,
                (heldStart ^ dist) + i);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempAllReduceHD::GetHDStepNum() const
{
    u32 nSteps = 0;
    for (u32 tmp = templateRankSize_; tmp > 1; tmp >>= 1, nSteps++) {
    }
    HCCL_DEBUG("[InsTempAllReduceHD][GetHDStepNum] rankSize[%u] nSteps[%u]", templateRankSize_, nSteps);

    return nSteps;
}

} // namespace ops_hccl
