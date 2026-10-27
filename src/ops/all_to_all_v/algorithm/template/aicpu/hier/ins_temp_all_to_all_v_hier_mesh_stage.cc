/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_to_all_v_hier_mesh_stage.h"
#include "alltoall_stage_template_registry.h"
#include "channel.h"
#include "executor_base.h"

namespace ops_hccl {

InsTempAlltoAllVHierMeshStage::InsTempAlltoAllVHierMeshStage(
    const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks, u32 stageIndex)
    : InsTempAlltoAllHierStageBase(param, rankId, subCommRanks, stageIndex)
{}

HcclResult InsTempAlltoAllVHierMeshStage::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    u32 threadNum = std::min(ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE, templateRankSize_);
    resourceRequest.slaveThreadNum = (threadNum > 1) ? (threadNum - 1) : 0;
    for (u32 index = 0; index < resourceRequest.slaveThreadNum; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = resourceRequest.slaveThreadNum;

    std::vector<HcclChannelDesc> channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channels));
    resourceRequest.channels.push_back(channels);

    HCCL_INFO(
        "[InsTempAlltoAllVHierMeshStage][CalcRes] stageIndex[%u], rankSize[%u], threadNum[%u], "
        "slaveThreadNum[%u], channelNum[%u]",
        stageIndex_, templateRankSize_, threadNum, resourceRequest.slaveThreadNum, static_cast<u32>(channels.size()));
    return HCCL_SUCCESS;
}

void InsTempAlltoAllVHierMeshStage::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = std::min(ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE, templateRankSize_);
    u32 slaveThreadNum = (threadNum > 1) ? (threadNum - 1) : 0;
    for (u32 i = 0; i < slaveThreadNum; i++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempAlltoAllVHierMeshStage::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = std::min(ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE, templateRankSize_);
    u32 slaveThreadNum = (threadNum > 1) ? (threadNum - 1) : 0;
    for (u32 i = 0; i < slaveThreadNum; i++) {
        notifyIdxSubToMain.push_back(i);
    }
}

u32 InsTempAlltoAllVHierMeshStage::FindMyAlgRank() const
{
    if (subCommRanks_.empty() || subCommRanks_[0].empty()) {
        return INVALID_ALG_RANK;
    }
    for (u32 i = 0; i < subCommRanks_[0].size(); i++) {
        if (subCommRanks_[0][i] == myRank_) {
            return i;
        }
    }
    return INVALID_ALG_RANK;
}

void InsTempAlltoAllVHierMeshStage::InitStageParams(const TemplateDataParams& tempAlgParams, StageParams& params)
{
    dataType_ = tempAlgParams.dataType;
    dataTypeSize_ = HCCL_SIZE_TABLE[dataType_];
    params.otherDimRankSize = static_cast<u32>(tempAlgParams.outputSliceStride);
    cclBufferCountPerRank_ = (params.otherDimRankSize > 0) ?
                                 (tempAlgParams.inputSliceStride / dataTypeSize_ / params.otherDimRankSize) :
                                 0;
    params.stageRankSize = static_cast<u32>(subCommRanks_[0].size());
    params.srcGroupSize = params.otherDimRankSize;
    params.targetGroupSize = isFirstStage_ ? params.otherDimRankSize : params.stageRankSize;
    params.subSlotCount = isFirstStage_ ? templateRankSize_ : params.otherDimRankSize;
}

HcclResult InsTempAlltoAllVHierMeshStage::ValidateAndInit(
    const TemplateDataParams& tempAlgParams, u32& myAlgRank, StageParams& params)
{
    if (subCommRanks_.empty() || subCommRanks_[0].empty()) {
        HCCL_ERROR("[InsTempAlltoAllVHierMeshStage][KernelRun] subCommRanks is empty, stageIndex[%u]", stageIndex_);
        return HCCL_E_PARA;
    }
    // 防御性校验：基类单组推导约定templateRankSize_ == subCommRanks_[0].size()，
    // peer循环上界与变长数组下标均依赖该一致性
    CHK_PRT_RET(
        static_cast<u64>(subCommRanks_[0].size()) != templateRankSize_,
        HCCL_ERROR(
            "[InsTempAlltoAllVHierMeshStage][KernelRun] subCommRanks size[%zu] != templateRankSize[%u], "
            "stageIndex[%u]",
            subCommRanks_[0].size(), templateRankSize_, stageIndex_),
        HCCL_E_PARA);
    myAlgRank = FindMyAlgRank();
    if (myAlgRank == INVALID_ALG_RANK) {
        HCCL_ERROR("[InsTempAlltoAllVHierMeshStage][KernelRun] myRank[%u] not found in subCommRanks", myRank_);
        return HCCL_E_INTERNAL;
    }
    // 防御性校验：入口层已校验dataType，此处避免非法值越界查表/作除数
    CHK_PRT_RET(
        tempAlgParams.dataType >= HCCL_DATA_TYPE_RESERVED || HCCL_SIZE_TABLE[tempAlgParams.dataType] == 0,
        HCCL_ERROR(
            "[InsTempAlltoAllVHierMeshStage][KernelRun] invalid dataType[%u], stageIndex[%u]",
            static_cast<u32>(tempAlgParams.dataType), stageIndex_),
        HCCL_E_PARA);
    InitStageParams(tempAlgParams, params);
    if (cclBufferCountPerRank_ == 0) {
        HCCL_ERROR(
            "[InsTempAlltoAllVHierMeshStage][KernelRun] cclBufferCountPerRank_ is 0, "
            "inputSliceStride[%llu], dataTypeSize[%llu], otherDimRankSize[%u]",
            tempAlgParams.inputSliceStride, dataTypeSize_, params.otherDimRankSize);
        return HCCL_E_INTERNAL;
    }
    // 块布局容量校验：全矩阵块布局最大末端不得越过cclbuff(远端与本端同构同尺寸，本端校验即覆盖)
    u64 maxBlockBytes
        = static_cast<u64>(templateRankSize_) * params.otherDimRankSize * cclBufferCountPerRank_ * dataTypeSize_
          + tempAlgParams.buffInfo.hcclBuffBaseOff;
    CHK_PRT_RET(
        maxBlockBytes > tempAlgParams.buffInfo.hcclBuffSize,
        HCCL_ERROR(
            "[InsTempAlltoAllVHierMeshStage][KernelRun] block layout exceeds cclbuff, maxBlockBytes[%llu], "
            "hcclBuffSize[%llu], hcclBuffBaseOff[%llu], stageIndex[%u]",
            maxBlockBytes, tempAlgParams.buffInfo.hcclBuffSize, tempAlgParams.buffInfo.hcclBuffBaseOff, stageIndex_),
        HCCL_E_PARA);
    // 变长4数组的下标均基于全局rank(server-major)，需要覆盖本stage视角下的全部rank
    u64 matrixRankSize = static_cast<u64>(templateRankSize_) * params.otherDimRankSize;
    CHK_PRT_RET(
        tempAlgParams.sendCounts.size() < matrixRankSize || tempAlgParams.recvCounts.size() < matrixRankSize
            || tempAlgParams.sdispls.size() < matrixRankSize || tempAlgParams.rdispls.size() < matrixRankSize,
        HCCL_ERROR(
            "[InsTempAlltoAllVHierMeshStage][KernelRun] variadic arrays not enough, stageIndex[%u], "
            "sendCounts[%zu], recvCounts[%zu], sdispls[%zu], rdispls[%zu], matrixRankSize[%llu]",
            stageIndex_, tempAlgParams.sendCounts.size(), tempAlgParams.recvCounts.size(), tempAlgParams.sdispls.size(),
            tempAlgParams.rdispls.size(), matrixRankSize),
        HCCL_E_PARA);

    HCCL_INFO(
        "[InsTempAlltoAllVHierMeshStage][KernelRun] stageIndex[%u], myRank[%u], myAlgRank[%u], "
        "stageRankSize[%u], isFirst[%d], isLast[%d], isMiddle[%d], "
        "cclBufferCountPerRank_[%llu], srcGroupSize[%u], targetGroupSize[%u], subSlotCount[%u], "
        "processedDataCount[%llu]",
        stageIndex_, myRank_, myAlgRank, params.stageRankSize, isFirstStage_, isLastStage_, isMiddleStage_,
        cclBufferCountPerRank_, params.srcGroupSize, params.targetGroupSize, params.subSlotCount,
        tempAlgParams.processedDataCount);
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllVHierMeshStage::CalcSlotCount(u64 count, u64 processedDataCount) const
{
    return (count > processedDataCount) ? std::min(count - processedDataCount, cclBufferCountPerRank_) : 0;
}

HcclResult InsTempAlltoAllVHierMeshStage::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    (void)param;
    u32 myAlgRank = 0;
    StageParams params;
    CHK_RET(ValidateAndInit(tempAlgParams, myAlgRank, params));

    CHK_RET(PreSyncThreads(templateResource));

    if (isFirstStage_) {
        CHK_RET(LocalCopyForMyGroup(
            tempAlgParams, templateResource, myAlgRank, params.targetGroupSize, params.subSlotCount));
    }
    if (isLastStage_) {
        CHK_RET(
            LocalCopyForMyRank(tempAlgParams, templateResource, myAlgRank, params.srcGroupSize, params.subSlotCount));
    }
    CHK_RET(RunSendRecv(param, tempAlgParams, templateResource, myAlgRank, params));

    CHK_RET(PostSyncThreads(templateResource));

    HCCL_INFO("[InsTempAlltoAllVHierMeshStage][KernelRun] stageIndex[%u] done", stageIndex_);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVHierMeshStage::PreSyncThreads(const TemplateResource& templateResource)
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

HcclResult InsTempAlltoAllVHierMeshStage::PostSyncThreads(const TemplateResource& templateResource)
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

HcclResult InsTempAlltoAllVHierMeshStage::LocalCopyForMyGroup(
    const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource, u32 myAlgRank,
    u32 targetGroupSize, u32 subSlotCount)
{
    // 首级(inter)本地部分：本卡input中发往本server内各slot的数据，直写本卡cclbuff对应block
    const u64 processed = tempAlgParams.processedDataCount;
    for (u32 targetIdx = 0; targetIdx < targetGroupSize; targetIdx++) {
        u64 dataIdx = static_cast<u64>(myAlgRank) * targetGroupSize + targetIdx;
        u64 slotCount = CalcSlotCount(tempAlgParams.sendCounts[dataIdx], processed);
        if (slotCount == 0) {
            continue;
        }
        u64 srcOffset = tempAlgParams.sdispls[dataIdx] * dataTypeSize_ + tempAlgParams.buffInfo.inBuffBaseOff;
        u64 dstOffset
            = (static_cast<u64>(targetIdx) * subSlotCount + myAlgRank) * cclBufferCountPerRank_ * dataTypeSize_
              + tempAlgParams.buffInfo.hcclBuffBaseOff;

        DataSlice srcSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, slotCount * dataTypeSize_, slotCount);
        DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, dstOffset, slotCount * dataTypeSize_, slotCount);

        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
    }

    HCCL_DEBUG(
        "[InsTempAlltoAllVHierMeshStage][LocalCopyForMyGroup] myAlgRank[%u], targetGroupSize[%u], "
        "processedDataCount[%llu]",
        myAlgRank, targetGroupSize, processed);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVHierMeshStage::LocalCopyForMyRank(
    const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource, u32 myAlgRank, u32 srcGroupSize,
    u32 subSlotCount)
{
    // 末级(intra)本地部分：本卡cclbuff中属于本卡的block，直写本卡output(rdispls定位)
    const u64 processed = tempAlgParams.processedDataCount;
    for (u32 srcIdx = 0; srcIdx < srcGroupSize; srcIdx++) {
        u64 dataIdx = static_cast<u64>(srcIdx) * templateRankSize_ + myAlgRank;
        u64 slotCount = CalcSlotCount(tempAlgParams.recvCounts[dataIdx], processed);
        if (slotCount == 0) {
            continue;
        }
        u64 srcOffset = (static_cast<u64>(myAlgRank) * subSlotCount + srcIdx) * cclBufferCountPerRank_ * dataTypeSize_
                        + tempAlgParams.buffInfo.hcclBuffBaseOff;
        u64 dstOffset = tempAlgParams.rdispls[dataIdx] * dataTypeSize_ + tempAlgParams.buffInfo.outBuffBaseOff;

        DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcOffset, slotCount * dataTypeSize_, slotCount);
        DataSlice dstSlice(tempAlgParams.buffInfo.outputPtr, dstOffset, slotCount * dataTypeSize_, slotCount);

        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
    }

    HCCL_DEBUG(
        "[InsTempAlltoAllVHierMeshStage][LocalCopyForMyRank] myAlgRank[%u], srcGroupSize[%u], "
        "processedDataCount[%llu]",
        myAlgRank, srcGroupSize, processed);
    return HCCL_SUCCESS;
}

void InsTempAlltoAllVHierMeshStage::BuildSendSlices(
    const TemplateDataParams& tempAlgParams, u32 myAlgRank, u32 peerAlgRank, void* remoteCclBuffAddr,
    u32 targetGroupSize, u32 subSlotCount, std::vector<DataSlice>& txSrcSlices, std::vector<DataSlice>& txDstSlices)
{
    if (isFirstStage_) {
        // 首级tx为live侧：本卡input中发往对端server(peerAlgRank)各slot的段 -> 对端cclbuff block[t][s]
        const u64 processed = tempAlgParams.processedDataCount;
        for (u32 targetIdx = 0; targetIdx < targetGroupSize; targetIdx++) {
            u64 dataIdx = static_cast<u64>(peerAlgRank) * targetGroupSize + targetIdx;
            u64 slotCount = CalcSlotCount(tempAlgParams.sendCounts[dataIdx], processed);
            if (slotCount == 0) {
                continue;
            }
            u64 srcOffset = tempAlgParams.sdispls[dataIdx] * dataTypeSize_ + tempAlgParams.buffInfo.inBuffBaseOff;
            u64 dstOffset
                = (static_cast<u64>(targetIdx) * subSlotCount + myAlgRank) * cclBufferCountPerRank_ * dataTypeSize_
                  + tempAlgParams.buffInfo.hcclBuffBaseOff;
            txSrcSlices.emplace_back(tempAlgParams.buffInfo.inputPtr, srcOffset, slotCount * dataTypeSize_, slotCount);
            txDstSlices.emplace_back(remoteCclBuffAddr, dstOffset, slotCount * dataTypeSize_, slotCount);
        }
    }
    // 末级tx为死代码侧；live侧全零也需保证wrapper被调用(对端可能有数据发往本卡)，统一补占位slice
    if (txSrcSlices.empty()) {
        txSrcSlices.emplace_back(remoteCclBuffAddr, 0, 0, 0);
        txDstSlices.emplace_back(remoteCclBuffAddr, 0, 0, 0);
    }
}

void InsTempAlltoAllVHierMeshStage::BuildRecvSlices(
    const TemplateDataParams& tempAlgParams, u32 myAlgRank, u32 peerAlgRank, void* remoteCclBuffAddr,
    const StageParams& params, std::vector<DataSlice>& rxSrcSlices, std::vector<DataSlice>& rxDstSlices)
{
    if (isLastStage_) {
        // 末级rx为live侧：对端cclbuff中属于本卡的block[b'][i] -> 本卡output直写(rdispls定位)
        const u64 processed = tempAlgParams.processedDataCount;
        for (u32 srcIdx = 0; srcIdx < params.srcGroupSize; srcIdx++) {
            u64 dataIdx = static_cast<u64>(srcIdx) * templateRankSize_ + peerAlgRank;
            u64 slotCount = CalcSlotCount(tempAlgParams.recvCounts[dataIdx], processed);
            if (slotCount == 0) {
                continue;
            }
            u64 srcOffset
                = (static_cast<u64>(myAlgRank) * params.subSlotCount + srcIdx) * cclBufferCountPerRank_ * dataTypeSize_
                  + tempAlgParams.buffInfo.hcclBuffBaseOff;
            u64 dstOffset = tempAlgParams.rdispls[dataIdx] * dataTypeSize_ + tempAlgParams.buffInfo.outBuffBaseOff;
            rxSrcSlices.emplace_back(remoteCclBuffAddr, srcOffset, slotCount * dataTypeSize_, slotCount);
            rxDstSlices.emplace_back(tempAlgParams.buffInfo.outputPtr, dstOffset, slotCount * dataTypeSize_, slotCount);
        }
    }
    // 首级rx为死代码侧；live侧全零也需保证wrapper被调用，统一补占位slice
    if (rxSrcSlices.empty()) {
        rxSrcSlices.emplace_back(remoteCclBuffAddr, 0, 0, 0);
        rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, 0, 0, 0);
    }
}

HcclResult InsTempAlltoAllVHierMeshStage::ExecuteSendRecv(
    const ChannelInfo& linkSend, const ChannelInfo& linkRecv, const ThreadHandle& thread,
    const std::vector<DataSlice>& txSrcSlices, const std::vector<DataSlice>& txDstSlices,
    const std::vector<DataSlice>& rxSrcSlices, const std::vector<DataSlice>& rxDstSlices, u32 remoteRank)
{
    // 恒定wrapper：占位slice保证双侧非空，对每个peer必然调用一次，notify两侧配对
    SendRecvInfo sendRecvInfo{
        {linkSend, linkRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_};
    if (isLastStage_) {
        CHK_PRT_RET(
            SendRecvRead(sendRecvInfo, thread),
            HCCL_ERROR("[InsTempAlltoAllVHierMeshStage] SendRecvRead failed, remoteRank[%u]", remoteRank),
            HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(
            SendRecvWrite(sendRecvInfo, thread),
            HCCL_ERROR("[InsTempAlltoAllVHierMeshStage] SendRecvWrite failed, remoteRank[%u]", remoteRank),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVHierMeshStage::RunSendRecv(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource, u32 myAlgRank,
    const StageParams& params)
{
    (void)param;
    u32 threadNum = static_cast<u32>(templateResource.threads.size());

    for (u32 peerAlgRank = 0; peerAlgRank < templateRankSize_; peerAlgRank++) {
        if (peerAlgRank == myAlgRank) {
            continue;
        }

        u32 remoteRank = subCommRanks_[0][peerAlgRank];
        auto chanIt = templateResource.channels.find(remoteRank);
        if (chanIt == templateResource.channels.end() || chanIt->second.empty()) {
            HCCL_ERROR("[InsTempAlltoAllVHierMeshStage] channel not found for remoteRank[%u]", remoteRank);
            return HCCL_E_INTERNAL;
        }
        const ChannelInfo& linkSend = chanIt->second[0];
        const ChannelInfo& linkRecv = chanIt->second[0];
        void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;

        HCCL_DEBUG(
            "[InsTempAlltoAllVHierMeshStage][RunSendRecv] stageIndex[%u], myAlgRank[%u], peerAlgRank[%u], "
            "remoteRank[%u], isValid[%d], protocol[%u], locationType[%u], "
            "remoteCclBuffAddr[%p], remoteCclMemSize[%llu]",
            stageIndex_, myAlgRank, peerAlgRank, remoteRank, static_cast<int>(linkSend.isValid),
            static_cast<u32>(linkSend.protocol), static_cast<u32>(linkSend.locationType), remoteCclBuffAddr,
            linkSend.remoteCclMem.size);

        u32 queIdx = (peerAlgRank < myAlgRank) ? peerAlgRank : (peerAlgRank - 1);
        if (queIdx >= threadNum) {
            HCCL_WARNING(
                "[InsTempAlltoAllVHierMeshStage] queIdx[%u] >= threadNum[%u], fallback to main thread, "
                "peerAlgRank[%u], myAlgRank[%u], stageIndex[%u]",
                queIdx, threadNum, peerAlgRank, myAlgRank, stageIndex_);
            queIdx = 0;
        }
        const ThreadHandle& thread = templateResource.threads[queIdx];

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        BuildSendSlices(
            tempAlgParams, myAlgRank, peerAlgRank, remoteCclBuffAddr, params.targetGroupSize, params.subSlotCount,
            txSrcSlices, txDstSlices);
        BuildRecvSlices(tempAlgParams, myAlgRank, peerAlgRank, remoteCclBuffAddr, params, rxSrcSlices, rxDstSlices);

        HCCL_DEBUG(
            "[InsTempAlltoAllVHierMeshStage][RunSendRecv] after build slices, txSrcSlices[%zu], "
            "rxSrcSlices[%zu], processedDataCount[%llu]",
            txSrcSlices.size(), rxSrcSlices.size(), tempAlgParams.processedDataCount);

        CHK_RET(ExecuteSendRecv(
            linkSend, linkRecv, thread, txSrcSlices, txDstSlices, rxSrcSlices, rxDstSlices, remoteRank));
    }

    HCCL_DEBUG("[InsTempAlltoAllVHierMeshStage][RunSendRecv] done, myAlgRank[%u]", myAlgRank);
    return HCCL_SUCCESS;
}

REGISTER_A2A_STAGE_TEMPLATE(MeshVStage, InsTempAlltoAllVHierMeshStage);

} // namespace ops_hccl
