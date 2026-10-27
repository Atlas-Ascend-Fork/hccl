/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_to_all_v_meta_stage.h"
#include "channel.h"
#include "executor_base.h"

namespace ops_hccl {

InsTempAlltoAllVHierMetaStage::InsTempAlltoAllVHierMetaStage(
    const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks, u32 stageIndex)
    : InsTempAlltoAllHierStageBase(param, rankId, subCommRanks, stageIndex)
{
    // subCommRanks = {intraRanks, interRanks}：基类构造已得templateRankSize_ = serverSize * serverNum = N
    serverSize_ = (!subCommRanks_.empty() && !subCommRanks_[0].empty()) ? static_cast<u32>(subCommRanks_[0].size()) : 0;
    serverNum_
        = (subCommRanks_.size() > 1 && !subCommRanks_[1].empty()) ? static_cast<u32>(subCommRanks_[1].size()) : 0;
    // 组不存在时不解引用(空/单组传入时下标访问是UB)，交由ValidateAndInit统一拦截
    mySlotIdx_ = !subCommRanks_.empty() ? FindIdxInGroup(subCommRanks_[0], rankId) : INVALID_ALG_RANK;
    myServerIdx_ = (subCommRanks_.size() > 1) ? FindIdxInGroup(subCommRanks_[1], rankId) : INVALID_ALG_RANK;
}

HcclResult InsTempAlltoAllVHierMetaStage::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // MetaStage由executor直接构造，线程/notify/channel资源与数据stage复用，无需单独申请
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)resourceRequest;
    return HCCL_SUCCESS;
}

void InsTempAlltoAllVHierMetaStage::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = std::min(ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE, templateRankSize_);
    u32 slaveThreadNum = (threadNum > 1) ? (threadNum - 1) : 0;
    for (u32 i = 0; i < slaveThreadNum; i++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempAlltoAllVHierMetaStage::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = std::min(ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE, templateRankSize_);
    u32 slaveThreadNum = (threadNum > 1) ? (threadNum - 1) : 0;
    for (u32 i = 0; i < slaveThreadNum; i++) {
        notifyIdxSubToMain.push_back(i);
    }
}

u32 InsTempAlltoAllVHierMetaStage::FindIdxInGroup(const std::vector<u32>& group, u32 rankId)
{
    for (u32 i = 0; i < group.size(); i++) {
        if (group[i] == rankId) {
            return i;
        }
    }
    return INVALID_ALG_RANK;
}

HcclResult InsTempAlltoAllVHierMetaStage::ValidateAndInit()
{
    CHK_PRT_RET(
        subCommRanks_.size() < 2 || serverSize_ == 0 || serverNum_ == 0,
        HCCL_ERROR(
            "[InsTempAlltoAllVHierMetaStage] expect dual groups {intra, inter}, groupNum[%zu], "
            "serverSize[%u], serverNum[%u]",
            subCommRanks_.size(), serverSize_, serverNum_),
        HCCL_E_PARA);
    CHK_PRT_RET(
        mySlotIdx_ == INVALID_ALG_RANK || myServerIdx_ == INVALID_ALG_RANK,
        HCCL_ERROR(
            "[InsTempAlltoAllVHierMetaStage] myRank[%u] not found in groups, mySlotIdx[%u], "
            "myServerIdx[%u]",
            myRank_, mySlotIdx_, myServerIdx_),
        HCCL_E_INTERNAL);
    // rank分解校验：inter组第s个成员即rank(s,b)，全局编号应为s*serverSize + b
    u32 decomposedRank = myServerIdx_ * serverSize_ + mySlotIdx_;
    CHK_PRT_RET(
        subCommRanks_[1][myServerIdx_] != decomposedRank,
        HCCL_ERROR(
            "[InsTempAlltoAllVHierMetaStage] rank decompose mismatch, interGroupRank[%u], expect[%u]",
            subCommRanks_[1][myServerIdx_], decomposedRank),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        static_cast<u64>(serverSize_) * serverNum_ != templateRankSize_,
        HCCL_ERROR(
            "[InsTempAlltoAllVHierMetaStage] size mismatch, serverSize[%u] * serverNum[%u] != "
            "templateRankSize[%u]",
            serverSize_, serverNum_, templateRankSize_),
        HCCL_E_INTERNAL);

    HCCL_INFO(
        "[InsTempAlltoAllVHierMetaStage] myRank[%u], s[%u], b[%u], serverSize[%u], serverNum[%u], "
        "rankSize[%u]",
        myRank_, myServerIdx_, mySlotIdx_, serverSize_, serverNum_, templateRankSize_);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVHierMetaStage::LocalCopyMatrixRow(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    const u64 rowSize = static_cast<u64>(templateRankSize_) * sizeof(u64); // N * sizeof(u64)
    u64 rowOffset = (static_cast<u64>(mySlotIdx_) * serverNum_ + myServerIdx_) * rowSize;

    // AICPU直读本卡sendCounts行(N个u64)，直写本卡cclbuff头部矩阵对应行
    DataSlice srcSlice(param.all2AllVDataDes.sendCounts, 0, rowSize, rowSize / sizeof(u64));
    DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, rowOffset, rowSize, rowSize / sizeof(u64));
    CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));

    HCCL_DEBUG(
        "[InsTempAlltoAllVHierMetaStage][LocalCopyMatrixRow] rowOffset[%llu], rowSize[%llu]", rowOffset, rowSize);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVHierMetaStage::RunInterExchange(
    const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    const u64 rowSize = static_cast<u64>(templateRankSize_) * sizeof(u64);
    u64 myRowOffset = (static_cast<u64>(mySlotIdx_) * serverNum_ + myServerIdx_) * rowSize;
    u32 threadNum = static_cast<u32>(templateResource.threads.size());

    for (u32 peerServerIdx = 0; peerServerIdx < serverNum_; peerServerIdx++) {
        if (peerServerIdx == myServerIdx_) {
            continue;
        }
        u32 remoteRank = subCommRanks_[1][peerServerIdx];
        auto chanIt = templateResource.channels.find(remoteRank);
        if (chanIt == templateResource.channels.end() || chanIt->second.empty()) {
            HCCL_ERROR("[InsTempAlltoAllVHierMetaStage] channel not found for remoteRank[%u]", remoteRank);
            return HCCL_E_INTERNAL;
        }
        const ChannelInfo& linkSend = chanIt->second[0];
        const ChannelInfo& linkRecv = chanIt->second[0];
        void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;

        u32 queIdx = (peerServerIdx < myServerIdx_) ? peerServerIdx : (peerServerIdx - 1);
        if (queIdx >= threadNum) {
            HCCL_WARNING(
                "[InsTempAlltoAllVHierMetaStage] queIdx[%u] >= threadNum[%u], fallback to main thread, "
                "peerServerIdx[%u], myServerIdx[%u]",
                queIdx, threadNum, peerServerIdx, myServerIdx_);
            queIdx = 0;
        }
        const ThreadHandle& thread = templateResource.threads[queIdx];

        // tx为live侧：本卡矩阵行 -> 对端cclbuff同偏移；rx为死代码侧恒占位，保证wrapper被调用
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, myRowOffset, rowSize, rowSize / sizeof(u64));
        txDstSlices.emplace_back(remoteCclBuffAddr, myRowOffset, rowSize, rowSize / sizeof(u64));
        rxSrcSlices.emplace_back(remoteCclBuffAddr, 0, 0, 0);
        rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, 0, 0, 0);

        SendRecvInfo sendRecvInfo{
            {linkSend, linkRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, HCCL_DATA_TYPE_UINT64};
        CHK_PRT_RET(
            SendRecvWrite(sendRecvInfo, thread),
            HCCL_ERROR("[InsTempAlltoAllVHierMetaStage] SendRecvWrite failed, remoteRank[%u]", remoteRank),
            HCCL_E_INTERNAL);
    }

    HCCL_DEBUG(
        "[InsTempAlltoAllVHierMetaStage][RunInterExchange] done, myRowOffset[%llu], rowSize[%llu]", myRowOffset,
        rowSize);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVHierMetaStage::RunIntraGather(
    const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    const u64 rowSize = static_cast<u64>(templateRankSize_) * sizeof(u64);
    u64 columnSize = static_cast<u64>(serverNum_) * rowSize; // peer的0a结果：连续serverNum行
    u32 threadNum = static_cast<u32>(templateResource.threads.size());

    for (u32 peerSlotIdx = 0; peerSlotIdx < serverSize_; peerSlotIdx++) {
        if (peerSlotIdx == mySlotIdx_) {
            continue;
        }
        u32 remoteRank = subCommRanks_[0][peerSlotIdx];
        auto chanIt = templateResource.channels.find(remoteRank);
        if (chanIt == templateResource.channels.end() || chanIt->second.empty()) {
            HCCL_ERROR("[InsTempAlltoAllVHierMetaStage] channel not found for remoteRank[%u]", remoteRank);
            return HCCL_E_INTERNAL;
        }
        const ChannelInfo& linkSend = chanIt->second[0];
        const ChannelInfo& linkRecv = chanIt->second[0];
        void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;

        u32 queIdx = (peerSlotIdx < mySlotIdx_) ? peerSlotIdx : (peerSlotIdx - 1);
        if (queIdx >= threadNum) {
            HCCL_WARNING(
                "[InsTempAlltoAllVHierMetaStage] queIdx[%u] >= threadNum[%u], fallback to main thread, "
                "peerSlotIdx[%u], mySlotIdx[%u]",
                queIdx, threadNum, peerSlotIdx, mySlotIdx_);
            queIdx = 0;
        }
        const ThreadHandle& thread = templateResource.threads[queIdx];

        // rx为live侧：对端0a结果(第peerSlotIdx*serverNum行起连续serverNum行) -> 本卡同偏移；
        // tx为死代码侧恒占位，保证wrapper被调用(0b的ACK-wait依赖两侧都进入wrapper)
        u64 columnOffset = static_cast<u64>(peerSlotIdx) * serverNum_ * rowSize;
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        rxSrcSlices.emplace_back(remoteCclBuffAddr, columnOffset, columnSize, columnSize / sizeof(u64));
        rxDstSlices.emplace_back(
            tempAlgParams.buffInfo.hcclBuff.addr, columnOffset, columnSize, columnSize / sizeof(u64));
        txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, 0, 0, 0);
        txDstSlices.emplace_back(remoteCclBuffAddr, 0, 0, 0);

        SendRecvInfo sendRecvInfo{
            {linkSend, linkRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, HCCL_DATA_TYPE_UINT64};
        CHK_PRT_RET(
            SendRecvRead(sendRecvInfo, thread),
            HCCL_ERROR("[InsTempAlltoAllVHierMetaStage] SendRecvRead failed, remoteRank[%u]", remoteRank),
            HCCL_E_INTERNAL);
    }

    HCCL_DEBUG("[InsTempAlltoAllVHierMetaStage][RunIntraGather] done, columnSize[%llu]", columnSize);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVHierMetaStage::PreSyncThreads(const TemplateResource& templateResource)
{
    u32 threadNum = static_cast<u32>(templateResource.threads.size());
    if (threadNum <= 1) {
        return HCCL_SUCCESS;
    }
    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    notifyIdxMainToSub_.resize(subThreads.size(), 0);
    return PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_);
}

HcclResult InsTempAlltoAllVHierMetaStage::PostSyncThreads(const TemplateResource& templateResource)
{
    u32 threadNum = static_cast<u32>(templateResource.threads.size());
    if (threadNum <= 1) {
        return HCCL_SUCCESS;
    }
    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    notifyIdxSubToMain_.clear();
    for (u32 i = 0; i < subThreads.size(); i++) {
        notifyIdxSubToMain_.push_back(i);
    }
    return PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_);
}

HcclResult InsTempAlltoAllVHierMetaStage::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    CHK_PRT_RET(
        param.all2AllVDataDes.sendCounts == nullptr,
        HCCL_ERROR("[InsTempAlltoAllVHierMetaStage][KernelRun] sendCounts ptr is null"), HCCL_E_PARA);
    // overlay前提：全矩阵(N*N*u64)需完整放入cclbuff头部
    const u64 matrixSize = static_cast<u64>(templateRankSize_) * templateRankSize_ * sizeof(u64);
    CHK_PRT_RET(
        tempAlgParams.buffInfo.hcclBuffSize < matrixSize,
        HCCL_ERROR(
            "[InsTempAlltoAllVHierMetaStage][KernelRun] cclBuffSize[%llu] < matrixSize[%llu], "
            "cannot overlay metadata matrix on cclbuff",
            tempAlgParams.buffInfo.hcclBuffSize, matrixSize),
        HCCL_E_PARA);
    CHK_RET(ValidateAndInit());

    // 行拷贝先入队主线程流再record：从线程tx读cclbuff本卡行(myRowOffset)，
    // record须排在LocalCopy之后，否则wait通过后tx可能与行拷贝并发(跨流依赖)
    CHK_RET(LocalCopyMatrixRow(param, tempAlgParams, templateResource));
    CHK_RET(PreSyncThreads(templateResource));
    CHK_RET(RunInterExchange(tempAlgParams, templateResource));
    CHK_RET(PostSyncThreads(templateResource)); // 本rank的0a全部完成
    CHK_RET(PreSyncThreads(templateResource));
    CHK_RET(RunIntraGather(tempAlgParams, templateResource));
    CHK_RET(PostSyncThreads(templateResource));

    HCCL_INFO("[InsTempAlltoAllVHierMetaStage][KernelRun] done, matrixSize[%llu]", matrixSize);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
