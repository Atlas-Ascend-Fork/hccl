/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_v_mesh_1D.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
namespace ops_hccl {
std::vector<CostModelParam> InsTempAllGatherVMesh1D::CalcCostCoeff(CalcCostCoeffParam param)
{
    int portNum = param.portNum[0];
    int taskNum
        = CostModelManager::CalcTransTaskNum(param.rankSize) + CostModelManager::CalcSyncTaskNum(param.rankSize) * 2;
    float A = 0;
    float B = 0;
    float C = 0;
    float D = 0;

    CostModelManager::Global()->CalcMeshParam(param.dataRatio, param.netType, portNum, param.rankSize, A, param.isPod);
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B);
    }
    CostModelManager::Global()->CalcLatencyParams(1, EngineType::AICPU, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AICPU, D);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

InsTempAllGatherVMesh1D::InsTempAllGatherVMesh1D(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
    supportSymmetricMemory_ = param.supportSymmetricMemory;
}
InsTempAllGatherVMesh1D::~InsTempAllGatherVMesh1D() {}

HcclResult InsTempAllGatherVMesh1D::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // mesh 算法只做level 0 层级的
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);

    // 多通道: 通道数即 die 拓扑的运行时体现, 线程按 (peer 数 × 通道数) 申请
    channelsPerRank_ = CalcChannelsPerRank(level0Channels);
    u32 peers = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 threadNum = peers * channelsPerRank_;
    resourceRequest.slaveThreadNum = threadNum - 1; // 主thread可以通过接口传入的stream来做转换
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    HCCL_INFO(
        "[InsTempAllGatherVMesh1D][CalcRes] channelsPerRank[%u] level0Channels[%zu] slaveThreadNum[%u]",
        channelsPerRank_, level0Channels.size(), resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherVMesh1D::GetThreadNum() const
{
    // 多通道: 总线程数 = peer 数 × 每 rank 通道数(与 CalcRes 申请同口径)
    u32 peers = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    return peers * channelsPerRank_;
}

u64 InsTempAllGatherVMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    if (supportSymmetricMemory_) {
        return 0;
    }
    u64 scratchMultiple = 0;
    if (opMode_ == OpMode::OPBASE) {
        scratchMultiple = templateRankSize_;
    }
    return scratchMultiple;
}

HcclResult InsTempAllGatherVMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    HCCL_INFO("[InsTempAllGatherVMesh1D] Run start");
    if (tempAlgParams.sliceSize == 0) {
        HCCL_INFO("[InsTempAllGatherVMesh1D] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }
    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.vDataDes.dataType;
    // 多通道: 运行期按实际批到的通道刷新
    if (!templateResource.channels.empty()) {
        channelsPerRank_ = CalcChannelsPerRank(templateResource.channels);
    }
    u32 peers = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    if (threadNum_ < peers * channelsPerRank_) {
        u32 shrunk = std::max(1U, threadNum_ / peers);
        HCCL_WARNING(
            "[InsTempAllGatherVMesh1D] threadNum_[%u] < peers[%u] * channelsPerRank_[%u], shrink channels to [%u]",
            threadNum_, peers, channelsPerRank_, shrunk);
        channelsPerRank_ = shrunk;
    }
    HCCL_DEBUG("[InsTempAllGatherVMesh1D] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);
    HCCL_INFO("[InsTempAllGatherVMesh1D] Run Start channelsPerRank_[%u] threadNum_[%u]", channelsPerRank_, threadNum_);
    CHK_RET(LocalDataCopy(templateResource.threads));
    if (templateRankSize_ == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    CHK_RET(RunAllGatherVMesh(templateResource.threads, templateResource.channels));

    if (threadNum_ > 1) {
        // 定义在alg_v2_template_base.h中的基类
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    if (!enableRemoteMemAccess_) {
        CHK_RET(PostLocalCopy(templateResource.threads));
    }
    HCCL_INFO("[InsTempAllGatherVMesh1D] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherVMesh1D::RunAllGatherVMesh(
    const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    HCCL_INFO("[InsTempAllGatherVMesh1D] RunAllGatherVMesh RankIDs[%d].", myRank_);
    u64 dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataTypeSize_ == 0, HCCL_ERROR("[InsTempAllGatherVMesh1D] dataTypeSize is 0"), HCCL_E_INTERNAL);
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    // 多通道: peer 数 = rankSize-1
    u32 peers = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * dataTypeSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (u32 peerIdx = 0; peerIdx < peers; peerIdx++) {
            u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + peerIdx) % subCommRanks_[0].size()];

            u32 connectedAlgRank = 0;
            CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));
            HCCL_INFO(
                "[InsTempAllGatherVMesh1D] RunAllGatherVMesh RankIDs[%d], connectedRank[%d], connectedAlgRank[%d].",
                myRank_, connectedRank, connectedAlgRank);

            // 异常检查
            CHK_PRT_RET(
                channels.count(connectedRank) == 0 || channels.at(connectedRank).empty(),
                HCCL_ERROR(
                    "[InsTempAllGatherVMesh1D][RankID]=%u peerIdx=%u, connectedRank=%d, channels.size=%zu", myRank_,
                    peerIdx, connectedRank, channels.size()),
                HcclResult::HCCL_E_INTERNAL);
            const std::vector<ChannelInfo>& curChannels = channels.at(connectedRank);

            const u64 mySliceSize = tempAlgParams_.allRankSliceSize[myAlgRank];
            const u64 peerSliceSize = tempAlgParams_.allRankSliceSize[connectedAlgRank];
            const u64 mySliceCount = mySliceSize / dataTypeSize_;
            const u64 peerSliceCount = peerSliceSize / dataTypeSize_;

            u64 txOutOffset = tempAlgParams_.allRankDispls[myAlgRank] * dataTypeSize_ + outBaseOff;
            u64 txScratchOffset = scratchBase + tempAlgParams_.sliceSize * myAlgRank;

            u64 rxOutOffset = tempAlgParams_.allRankDispls[connectedAlgRank] * dataTypeSize_ + outBaseOff;
            u64 rxScratchOffset = scratchBase + tempAlgParams_.sliceSize * connectedAlgRank;

            if (enableRemoteMemAccess_) {
                // 空槽（本端/对端本轮无数据）不做落点校验：多轮 loop 且各 rank counts 不等时，
                // 已传完一侧的槽位无实际写行为，按 displs+本轮进度校验会误报越界。
                const u64 outputSize = tempAlgParams_.buffInfo.outputSize;
                if ((mySliceSize > 0 && txOutOffset + mySliceSize > outputSize)
                    || (peerSliceSize > 0 && rxOutOffset + peerSliceSize > outputSize)) {
                    HCCL_ERROR(
                        "[InsTempAllGatherVMesh1D] output offset out of range, localRank[%u], remoteRank[%u], "
                        "txOffset[%llu], txSize[%llu], rxOffset[%llu], rxSize[%llu], outputSize[%llu]",
                        myRank_, connectedRank, txOutOffset, mySliceSize, rxOutOffset, peerSliceSize, outputSize);
                    return HcclResult::HCCL_E_INTERNAL;
                }
            }

            // 多通道: 把数据块按该 peer 链路的 portGroupSize 权重切到各通道;
            // 接收方向用同一批链路的权重切对端块, 保证逐通道收发配对。
            // 空 count 的 rank 两方向均为 0 尺寸切片, 不跳过调用以保握手对称。
            CHK_RET(CalcDataSplitByPortGroup(
                mySliceCount, dataTypeSize_, curChannels, elemCountOut_, sizeOut_, elemOffset_));
            CHK_RET(CalcDataSplitByPortGroup(
                peerSliceCount, dataTypeSize_, curChannels, rxElemCountOut_, rxSizeOut_, rxElemOffset_));
            u32 validChannels = std::min(channelsPerRank_, static_cast<u32>(curChannels.size()));
            for (u32 channelIdx = 0; channelIdx < validChannels; channelIdx++) {
                // 预留兼容offload模式
                const ChannelInfo& linkRemote = curChannels[channelIdx];
                void* remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
                void* remoteOutBuffAddr = linkRemote.remoteOutputGraphMode.addr;
                if (enableRemoteMemAccess_ && remoteOutBuffAddr == nullptr) {
                    HCCL_ERROR(
                        "[InsTempAllGatherVMesh1D] remote output addr is null, localRank[%u], remoteRank[%u]", myRank_,
                        connectedRank);
                    return HcclResult::HCCL_E_INTERNAL;
                }

                void* txSrcPtr = tempAlgParams_.buffInfo.outputPtr;
                void* txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : remoteOutBuffAddr;
                void* rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : remoteOutBuffAddr;
                void* rxDstPtr = (!enableRemoteMemAccess_) ? tempAlgParams_.buffInfo.hcclBuff.addr :
                                                             tempAlgParams_.buffInfo.outputPtr;
                // 多通道: 本通道子区偏移叠加到收发两侧(切分纯发送端本地, 接收侧不感知)
                u64 txSrcOffset = txOutOffset + elemOffset_[channelIdx];
                u64 txDstOffset = (!enableRemoteMemAccess_) ? txScratchOffset + elemOffset_[channelIdx] : txSrcOffset;
                // write 模式下 rx 地址不参与寻址；read 模式下源与目的同为对端/本地 output（或 CCL）布局，偏移一致
                u64 rxOffset = (!enableRemoteMemAccess_) ? rxScratchOffset + rxElemOffset_[channelIdx] :
                                                           rxOutOffset + rxElemOffset_[channelIdx];
                // write模式使用tx,rx地址不生效，仅使用对端link做Post/Wait
                // read 模式使用rx, tx地址不生效，仅使用对端link做Post/Wait
                std::vector<DataSlice> txSrcSlices{
                    DataSlice(txSrcPtr, txSrcOffset, sizeOut_[channelIdx], elemCountOut_[channelIdx])}; // 本地(send)
                std::vector<DataSlice> txDstSlices{
                    DataSlice(txDstPtr, txDstOffset, sizeOut_[channelIdx], elemCountOut_[channelIdx])}; // 远程(send)
                // read模式使用rx
                std::vector<DataSlice> rxDstSlices{
                    DataSlice(rxDstPtr, rxOffset, rxSizeOut_[channelIdx], rxElemCountOut_[channelIdx])}; // 本地(recv)
                std::vector<DataSlice> rxSrcSlices{
                    DataSlice(rxSrcPtr, rxOffset, rxSizeOut_[channelIdx], rxElemCountOut_[channelIdx])}; // 远程(recv)

                HCCL_DEBUG(
                    "[InsTempAllGatherVMesh1D][RunAllGatherVMesh] rankId [%d] connectedRank [%d] txSrcSlices: "
                    "offset[%d] sliceSize[%d] count[%d].",
                    myRank_, connectedRank, txSrcOffset, sizeOut_[channelIdx], elemCountOut_[channelIdx]);

                HCCL_DEBUG(
                    "[InsTempAllGatherVMesh1D][RunAllGatherVMesh] rankId [%d] connectedRank [%d] txDstSlices: "
                    "offset[%d] sliceSize[%d] count[%d].",
                    myRank_, connectedRank, txDstOffset, sizeOut_[channelIdx], elemCountOut_[channelIdx]);

                HCCL_DEBUG(
                    "[InsTempAllGatherVMesh1D][RunAllGatherVMesh] rankId [%d] connectedRank [%d] rxSrcSlices: "
                    "offset[%d] sliceSize[%d] count[%d].",
                    myRank_, connectedRank, rxOffset, rxSizeOut_[channelIdx], rxElemCountOut_[channelIdx]);

                HCCL_DEBUG(
                    "[InsTempAllGatherVMesh1D][RunAllGatherVMesh] rankId [%d] connectedRank [%d] rxDrcSlices: "
                    "offset[%d] sliceSize[%d] count[%d].",
                    myRank_, connectedRank, rxOffset, rxSizeOut_[channelIdx], rxElemCountOut_[channelIdx]);

                TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
                TxRxChannels sendRecvChannels(linkRemote, linkRemote);
                SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

                // 每个 (peer, channel) 工作项独占一条队列: peerIdx * channelsPerRank_ + channelIdx
                u32 queIdx = peerIdx * channelsPerRank_ + channelIdx;
                CHK_PRT_RET(
                    queIdx >= threads.size(),
                    HCCL_ERROR(
                        "[InsTempAllGatherVMesh1D][RankID]=%u queIdx=%u out of range, threads.size=%zu", myRank_,
                        queIdx, threads.size()),
                    HcclResult::HCCL_E_INTERNAL);
                CHK_PRT_RET(
                    SendRecvWrite(sendRecvInfo, threads[queIdx]),
                    HCCL_ERROR("[InsTempAllGatherVMesh1D] RunAllGatherV Send failed"), HcclResult::HCCL_E_INTERNAL);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherVMesh1D::LocalDataCopy(const std::vector<ThreadHandle>& threads)
{
    HCCL_INFO("[InsTempAllGatherVMesh1D] LocalDataCopy.");
    if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr) {
        return HcclResult::HCCL_SUCCESS;
    }
    u32 myAlgRank;
    u64 dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataTypeSize_ == 0, HCCL_ERROR("[InsTempAllGatherVMesh1D] dataTypeSize is 0"), HCCL_E_INTERNAL);
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        // repeat 造成的偏移
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        // 数据块rank编号造成的偏移
        const u64 inOff = inBaseOff;
        const u64 outOff = tempAlgParams_.allRankDispls[myAlgRank] * dataTypeSize_ + outBaseOff;
        const u64 sliceSize = tempAlgParams_.allRankSliceSize[myAlgRank];
        const u64 sliceCount = sliceSize / dataTypeSize_;

        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOff, sliceSize, sliceCount);

        HCCL_DEBUG(
            "[InsTempAllGatherVMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] srcSlice: inBaseOff[%d] inOff[%d] "
            "sliceSize[%d] count[%d].",
            myRank_, myAlgRank, inBaseOff, inOff, sliceSize, sliceCount);
        HCCL_DEBUG(
            "[InsTempAllGatherVMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] dstSlice: outBaseoff[%d] "
            "outOff[%d] sliceSize[%d] count[%d].",
            myRank_, myAlgRank, outBaseOff, outOff, sliceSize, sliceCount);

        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherVMesh1D::PostLocalCopy(const std::vector<ThreadHandle>& threads)
{
    HCCL_INFO("[InsTempAllGatherVMesh1D] PostLocalCopy.");
    u64 dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataTypeSize_ == 0, HCCL_ERROR("[InsTempAllGatherVMesh1D] dataTypeSize is 0"), HCCL_E_INTERNAL);
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (auto rank : subCommRanks_[0]) {
            if (rank == myRank_) {
                continue;
            }
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
            u64 scratchOffset = tempAlgParams_.sliceSize * algRank + scratchBase;
            u64 outBaseOff = tempAlgParams_.allRankProcessedDataCount[algRank] * dataTypeSize_
                             + rpt * tempAlgParams_.outputRepeatStride;
            u64 outOffset = tempAlgParams_.allRankDispls[algRank] * dataTypeSize_ + outBaseOff;
            u64 processSize = tempAlgParams_.allRankSliceSize[algRank];
            const u64 processCount = processSize / dataTypeSize_;
            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, processSize, processCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset, processSize, processCount);
            HCCL_DEBUG(
                "[InsTempAllGatherVMesh1D] PostLocalCopy RankID [%d] dataRank [%d] dataAlgRank[%d] "
                "scratchBase[%d] outBaseOff[%d] scratchOffset[%d] outOffset[%d].",
                myRank_, rank, algRank, scratchBase, outBaseOff, scratchOffset, outOffset);
            CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherVMesh1D::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    // 多通道: 通知数量按实际线程数(peers × 通道数), 与 CalcRes 申请同口径
    u32 slaveThreadNum = threadNum_ > 1 ? threadNum_ - 1 : 0;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempAllGatherVMesh1D::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    // 多通道: 主线程等待全部子线程完成, 通知数 = 实际线程数 - 1
    u32 notifyNum = threadNum_ > 1 ? threadNum_ - 1 : 0;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

} // namespace ops_hccl
