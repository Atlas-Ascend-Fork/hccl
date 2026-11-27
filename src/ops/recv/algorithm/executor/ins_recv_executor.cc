/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_recv_executor.h"
#include "alg_data_trans_wrapper.h"
#include "cost_model.h"
#include "alg_attrs_registry.h"
#include "p2p_channel_threads.h"
#include "template_utils.h"

namespace ops_hccl {
constexpr u32 P2P_CHANNEL_REPEAT_NUM = 2;
std::string InsRecvExecutor::Describe() const { return "Instruction based Recv Executor."; }

HcclResult InsRecvExecutor::InitRecvInfo(
    const HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void)comm;
    (void)algHierarchyInfo;
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    remoteRank_ = param.sendRecvRemoteRank;
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = static_cast<u64>(DATATYPE_SIZE_TABLE[dataType_]);

    HCCL_INFO(
        "[InsRecvExecutor][InitRecvInfo] myRank [%u], remoteRank [%u], rankSize [%u], devType [%u], "
        "dataType [%u] dataTypeSize [%u]",
        myRank_, remoteRank_, rankSize_, devType_, dataType_, dataTypeSize_);

    return HcclResult::HCCL_SUCCESS;
}
HcclResult InsRecvExecutor::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    (void)algAttrs;
    myRank_ = topoInfo->userRank;
    HCCL_DEBUG("[InsRecvExecutor][CalcAlgHierarchyInfoV2][%d] Start.", myRank_);
    CHK_PRT_RET(
        topoInfo->userRankSize == 0,
        HCCL_ERROR("[InsRecvExecutor][CalcAlgHierarchyInfoV2] Rank [%d], rankSize is 0.", myRank_),
        HcclResult::HCCL_E_PARA);

    algHierarchyInfo.infos.resize(1);
    algHierarchyInfo.infos[0].resize(1);
    algHierarchyInfo.infos[0][0].clear();
    for (uint32_t rankId = 0; rankId < topoInfo->userRankSize; rankId++) {
        algHierarchyInfo.infos[0][0].push_back(rankId);
    }

    HCCL_DEBUG("[InsRecvExecutor][CalcAlgHierarchyInfoV2][%d] Success.", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsRecvExecutor::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
#ifndef AICPU_COMPILE
    // 初始化一些基本成员变量
    InitRecvInfo(comm, param, topoInfo, algHierarchyInfo);
    HCCL_DEBUG("[InsRecvExecutor][CalcRes][%d]<-[%d] Start.", myRank_, remoteRank_);

    std::vector<HcclChannelDesc> level0Channels;
    bool isGroupEnabled = false;
    if (HcommIsSupportHcclGroupStatusGet()) {
        CHK_RET(HcclGroupStatusGet(&isGroupEnabled));
    }
    u32 channelNum = 1;
    if (isGroupEnabled) {
        CHK_RET(
            CreateChannelRequestByRankId(comm, param, myRank_, remoteRank_, level0Channels, P2P_CHANNEL_REPEAT_NUM));
    } else {
        CHK_RET(CreateChannelRequestByRankIdMultiChannel(comm, param, myRank_, remoteRank_, level0Channels));
        channelNum = static_cast<u32>(level0Channels.size());
    }
    resourceRequest.channels.push_back(level0Channels);

    resourceRequest.slaveThreadNum = (channelNum > 1) ? (channelNum - 1) : 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = resourceRequest.slaveThreadNum;

    HCCL_DEBUG("[InsRecvExecutor][CalcRes][%d]<-[%d] Success.", myRank_, remoteRank_);
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsRecvExecutor::RunDataPlane(const OpParam& param, const AlgResourceCtxSerializable& resCtx, ThreadHandle mainThread)
{
    opMode_ = param.opMode;
    myRank_ = resCtx.topoInfo.userRank;
    remoteRank_ = param.sendRecvRemoteRank;
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    // 防御性校验：非法 dataType 会越界查表且 dataTypeSize_ 可能为 0，
    // 后续 dataSize_ 计算与 maxLoopTransSize_ / dataTypeSize_ 除法均不可信。
    CHK_PRT_RET(
        !IsValidHcclDataType(dataType_),
        HCCL_ERROR("[InsRecvExecutor][%s] invalid dataType[%d]", __func__, static_cast<int>(dataType_)),
        HcclResult::HCCL_E_PARA);
    dataTypeSize_ = static_cast<u64>(DATATYPE_SIZE_TABLE[dataType_]);
    dataSize_ = dataCount_ * dataTypeSize_;

    HCCL_DEBUG("[InsRecvExecutor][RunDataPlane][%d]<-[%d] Start.", myRank_, remoteRank_);
    CHK_PRT_RET(
        (resCtx.threads.empty() || resCtx.channels.empty() || resCtx.channels[0].empty()),
        HCCL_ERROR("[InsRecvExecutor][RunDataPlane] Rank [%d], resCtx threads or channels is empty.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    std::vector<ChannelInfo> channels;
    std::vector<ThreadHandle> dataThreads;
    CHK_RET(
        P2pAssembleChannelsAndThreads(param, resCtx, myRank_, remoteRank_, false, mainThread, channels, dataThreads));

    // 使用的是PUT DMAMode，所以无论图模式还是单算子，数据都是从对端的input buffer来
    // channel.remoteInput不是对端input buffer
    // 但因为使用的是PUT模式srcBufferPtr无用，且无法获取对端input buffer地址，仅示意作用
    if (opMode_ == OpMode::OFFLOAD) {
        CHK_RET(OrchestrateOffload(param, resCtx, dataThreads, channels));
    } else {
        CHK_RET(OrchestrateOpbase(param, resCtx, dataThreads, channels));
    }
    CHK_RET(P2pFinishDataThreads(dataThreads));
    HCCL_DEBUG("[InsRecvExecutor][RunDataPlane][%d]<-[%d] Success.", myRank_, remoteRank_);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsRecvExecutor::Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    // dataType 校验先于资源校验（守卫顺序由 UT 锁定）；空 threads 传空句柄，资源校验在 RunDataPlane 内兜底
    ThreadHandle mainThread{};
    if (!resCtx.threads.empty()) {
        mainThread = resCtx.threads[0];
    }
    return RunDataPlane(param, resCtx, mainThread);
}

HcclResult InsRecvExecutor::OrchestrateWithThread(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, ThreadHandle sendRecvThread)
{
    return RunDataPlane(param, resCtx, sendRecvThread);
}

HcclResult InsRecvExecutor::OrchestrateOffload(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, const std::vector<ThreadHandle>& dataThreads,
    const std::vector<ChannelInfo>& channels)
{
    (void)resCtx;
    // 图模式直接到本端output buffer
    void* dstBufferPtr = static_cast<void*>(param.outputPtr);
    // UB传输最大数据量
    maxLoopTransSize_ = UB_MAX_DATA_SIZE;
    // 一次搬运最大数据个数
    maxLoopTransCount_ = maxLoopTransSize_ / dataTypeSize_;

    u32 channelNum = static_cast<u32>(channels.size());
    HCCL_INFO(
        "[InsRecvExecutor][Orchestrate][%d]<-[%d] OFFLOAD channelNum[%u], threadNum[%zu], Generating tasks.", myRank_,
        remoteRank_, channelNum, dataThreads.size());

    u64 dataCountToRecv = dataCount_;
    u64 currentOffset = 0;
    std::vector<std::vector<DataSlice>> srcSlicesByChannel(channelNum);
    std::vector<std::vector<DataSlice>> dstSlicesByChannel(channelNum);
    std::vector<bool> channelUsed(channelNum, false);
    std::vector<bool> isDmaReadByChannel(channelNum, false);
    while (dataCountToRecv > 0) {
        u64 transferCount = dataCountToRecv > maxLoopTransCount_ ? maxLoopTransCount_ : dataCountToRecv;
        u64 transferSize = transferCount * dataTypeSize_;
        std::vector<u64> elemCountOut;
        std::vector<u64> sizeOut;
        std::vector<u64> elemOffset;
        CHK_RET(CalcDataSplitByChannel(transferCount, dataTypeSize_, channels, elemCountOut, sizeOut, elemOffset));
        for (u32 channelIdx = 0; channelIdx < channelNum; channelIdx++) {
            if (elemCountOut[channelIdx] == 0) {
                continue;
            }
            channelUsed[channelIdx] = true;
            isDmaReadByChannel[channelIdx] = (channels[channelIdx].protocol == CommProtocol::COMM_PROTOCOL_PCIE);
            if (!isDmaReadByChannel[channelIdx]) {
                continue;
            }
            void* srcBufferPtr = static_cast<void*>(channels[channelIdx].remoteInputGraphMode.addr);
            srcSlicesByChannel[channelIdx].emplace_back(
                srcBufferPtr, currentOffset + elemOffset[channelIdx], sizeOut[channelIdx], elemCountOut[channelIdx]);
            dstSlicesByChannel[channelIdx].emplace_back(
                dstBufferPtr, currentOffset + elemOffset[channelIdx], sizeOut[channelIdx], elemCountOut[channelIdx]);
        }
        currentOffset = currentOffset + transferSize;
        dataCountToRecv = dataCountToRecv - transferCount;
    }
    for (u32 channelIdx = 0; channelIdx < channelNum; channelIdx++) {
        if (!channelUsed[channelIdx]) {
            continue;
        }
        if (isDmaReadByChannel[channelIdx]) {
            SlicesList recvSlicesList{srcSlicesByChannel[channelIdx], dstSlicesByChannel[channelIdx]};
            DataInfo recvInfo{channels[channelIdx], recvSlicesList};
            CHK_RET(RecvRead(recvInfo, dataThreads[channelIdx]));
        } else {
            std::vector<DataSlice> emptySlices;
            SlicesList recvSlicesList{emptySlices, emptySlices};
            DataInfo recvInfo{channels[channelIdx], recvSlicesList};
            CHK_RET(RecvWrite(recvInfo, dataThreads[channelIdx]));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsRecvExecutor::OrchestrateOpbase(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, const std::vector<ThreadHandle>& dataThreads,
    const std::vector<ChannelInfo>& channels)
{
    maxLoopTransSize_ = std::min<u64>(UB_MAX_DATA_SIZE, resCtx.cclMem.size);
    for (const ChannelInfo& channel : channels) {
        maxLoopTransSize_ = std::min<u64>(maxLoopTransSize_, channel.remoteCclMem.size);
    }
    maxLoopTransCount_ = maxLoopTransSize_ / dataTypeSize_;
    CHK_PRT_RET(
        maxLoopTransCount_ == 0,
        HCCL_ERROR(
            "[InsRecvExecutor][Orchestrate] maxLoopTransCount_[%llu] is 0, maxLoopTransSize_[%llu], "
            "dataTypeSize_[%llu].",
            maxLoopTransCount_, maxLoopTransSize_, dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);

    u32 channelNum = static_cast<u32>(channels.size());
    HCCL_INFO(
        "[InsRecvExecutor][Orchestrate][%d]<-[%d] OPBASE channelNum[%u], threadNum[%zu], Generating tasks.", myRank_,
        remoteRank_, channelNum, dataThreads.size());

    u64 dataCountToRecv = dataCount_;
    u64 currentOffset = 0;
    // 根据UB和ccl buffer大小限制，对数据进行切分
    while (dataCountToRecv > 0) {
        u64 transferCount = dataCountToRecv > maxLoopTransCount_ ? maxLoopTransCount_ : dataCountToRecv;
        u64 transferSize = transferCount * dataTypeSize_;
        std::vector<u64> elemCountOut;
        std::vector<u64> sizeOut;
        std::vector<u64> elemOffset;
        CHK_RET(CalcDataSplitByChannel(transferCount, dataTypeSize_, channels, elemCountOut, sizeOut, elemOffset));
        for (u32 channelIdx = 0; channelIdx < channelNum; channelIdx++) {
            if (elemCountOut[channelIdx] == 0) {
                continue;
            }
            bool isDmaRead = (channels[channelIdx].protocol == CommProtocol::COMM_PROTOCOL_PCIE);
            // 因ccl buffer大小限制，每次往ccl buffer写一片数据
            DataSlice remoteCclSlice{
                channels[channelIdx].remoteCclMem.addr, elemOffset[channelIdx], sizeOut[channelIdx],
                elemCountOut[channelIdx]};
            DataSlice outputSlice{
                param.outputPtr, currentOffset + elemOffset[channelIdx], sizeOut[channelIdx], elemCountOut[channelIdx]};
            if (isDmaRead) {
                // Read模式下，等待对端通知后，从对端cclBuffer中读取数据
                SlicesList recvSlicesList{{remoteCclSlice}, {outputSlice}};
                DataInfo recvInfo{channels[channelIdx], recvSlicesList};
                CHK_RET(RecvRead(recvInfo, dataThreads[channelIdx]));
            } else {
                // Write模式下，先等待对端数据写入本端cclBuffer，然后将数据拷贝到output上
                DataSlice cclSlice{
                    resCtx.cclMem.addr, elemOffset[channelIdx], sizeOut[channelIdx], elemCountOut[channelIdx]};
                SlicesList recvSlicesList{{remoteCclSlice}, {cclSlice}};
                DataInfo recvInfo{channels[channelIdx], recvSlicesList};
                CHK_RET(RecvWrite(recvInfo, dataThreads[channelIdx]));
                CHK_RET(LocalCopy(dataThreads[channelIdx], cclSlice, outputSlice));
            }
        }
        currentOffset = currentOffset + transferSize;
        dataCountToRecv = dataCountToRecv - transferCount;
    }

    return HcclResult::HCCL_SUCCESS;
}

std::vector<CostModelParam> InsRecvExecutor::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)topoInfo;
    (void)algName;
    (void)param;
    return {{0.0f, 0.0f, 3.0f, 0.0f}};
}

AlgNetMeta InsRecvExecutor::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)param;
    (void)algName;
    return BuildDefaultMeshNetMeta(topoInfo);
}

REGISTER_EXECUTOR_IMPL(HcclCMDType::HCCL_CMD_RECEIVE, AicpuRecvSoleMesh, InsRecvExecutor);
REGISTER_ALG_ATTRS(AicpuRecvSoleMesh, topo.supportLevel0Topos = LEVEL0_TOPO_ANY; topo.isSupportLevel0PcieMix = true;);
} // namespace ops_hccl
