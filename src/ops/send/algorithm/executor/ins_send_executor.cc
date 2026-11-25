/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_send_executor.h"
#include "alg_data_trans_wrapper.h"
#include "cost_model.h"
#include "alg_attrs_registry.h"
#include "auto_selector_base.h"
#include "hccl_rank_graph.h"
#include "p2p_channel_threads.h"
#include "template_utils.h"

namespace ops_hccl {
constexpr u32 P2P_CHANNEL_REPEAT_NUM = 2;

std::string InsSendExecutor::Describe() const { return "Instruction based Send Executor."; }

HcclResult
InsSendExecutor::InitSendInfo(const HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo)
{
    (void)comm;
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    remoteRank_ = param.sendRecvRemoteRank;
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = static_cast<u64>(DATATYPE_SIZE_TABLE[dataType_]);

    HCCL_INFO(
        "[InsSendExecutor][InitSendInfo] myRank [%u], remoteRank [%u], rankSize [%u], devType [%u], "
        "dataType [%u] dataTypeSize [%u]",
        myRank_, remoteRank_, rankSize_, devType_, dataType_, dataTypeSize_);

    return HcclResult::HCCL_SUCCESS;
}
HcclResult InsSendExecutor::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    (void)algAttrs;
    myRank_ = topoInfo->userRank;
    HCCL_DEBUG("[InsSendExecutor][CalcAlgHierarchyInfoV2][%d] Start.", myRank_);
    CHK_PRT_RET(
        (topoInfo->userRankSize == 0),
        HCCL_ERROR("[InsSendExecutor][CalcAlgHierarchyInfoV2] Rank [%d], rankSize is 0.", myRank_),
        HcclResult::HCCL_E_PARA);

    algHierarchyInfo.infos.resize(1);
    algHierarchyInfo.infos[0].resize(1);
    algHierarchyInfo.infos[0][0].clear();
    for (uint32_t rankId = 0; rankId < topoInfo->userRankSize; rankId++) {
        algHierarchyInfo.infos[0][0].push_back(rankId);
    }

    HCCL_DEBUG("[InsSendExecutor][CalcAlgHierarchyInfoV2][%d] Success.", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsSendExecutor::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    (void)algHierarchyInfo;
#ifndef AICPU_COMPILE
    // 初始化一些基本成员变量
    InitSendInfo(comm, param, topoInfo);
    HCCL_DEBUG("[InsSendExecutor][CalcRes][%d]->[%d] Start.", myRank_, remoteRank_);

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

    HCCL_DEBUG("[InsSendExecutor][CalcRes][%d]->[%d] Success.", myRank_, remoteRank_);
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsSendExecutor::RunDataPlane(const OpParam& param, const AlgResourceCtxSerializable& resCtx, ThreadHandle mainThread)
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
        HCCL_ERROR("[InsSendExecutor][%s] invalid dataType[%d]", __func__, static_cast<int>(dataType_)),
        HcclResult::HCCL_E_PARA);
    dataTypeSize_ = static_cast<u64>(DATATYPE_SIZE_TABLE[dataType_]);
    dataSize_ = dataCount_ * dataTypeSize_;

    HCCL_DEBUG("[InsSendExecutor][RunDataPlane][%d]->[%d] Start.", myRank_, remoteRank_);
    CHK_PRT_RET(
        (resCtx.threads.empty() || resCtx.channels.empty() || resCtx.channels[0].empty()),
        HCCL_ERROR("[InsSendExecutor][RunDataPlane] Rank [%d], resCtx threads or channels is empty.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    std::vector<ChannelInfo> channels;
    std::vector<ThreadHandle> dataThreads;
    CHK_RET(
        P2pAssembleChannelsAndThreads(param, resCtx, myRank_, remoteRank_, true, mainThread, channels, dataThreads));

    if (opMode_ == OpMode::OFFLOAD) {
        CHK_RET(OrchestrateOffload(param, resCtx, dataThreads, channels));
    } else {
        CHK_RET(OrchestrateOpbase(param, resCtx, dataThreads, channels));
    }
    CHK_RET(P2pFinishDataThreads(dataThreads));
    HCCL_DEBUG("[InsSendExecutor][RunDataPlane][%d]->[%d] Success.", myRank_, remoteRank_);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsSendExecutor::Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    // dataType 校验先于资源校验（守卫顺序由 UT 锁定）；空 threads 传空句柄，资源校验在 RunDataPlane 内兜底
    ThreadHandle mainThread{};
    if (!resCtx.threads.empty()) {
        mainThread = resCtx.threads[0];
    }
    return RunDataPlane(param, resCtx, mainThread);
}

HcclResult InsSendExecutor::OrchestrateWithThread(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, ThreadHandle sendRecvThread)
{
    return RunDataPlane(param, resCtx, sendRecvThread);
}

HcclResult InsSendExecutor::OrchestrateOffload(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, const std::vector<ThreadHandle>& dataThreads,
    const std::vector<ChannelInfo>& channels)
{
    (void)resCtx;
    // UB传输最大数据量
    maxLoopTransSize_ = UB_MAX_DATA_SIZE;
    // 一次搬运最大数据个数
    maxLoopTransCount_ = maxLoopTransSize_ / dataTypeSize_;

    u32 channelNum = static_cast<u32>(channels.size());
    HCCL_INFO(
        "[InsSendExecutor][Orchestrate][%d]->[%d] OFFLOAD channelNum[%u], threadNum[%zu], Generating tasks.", myRank_,
        remoteRank_, channelNum, dataThreads.size());
    // 图模式本端可拿到对端output buffer地址，所以直接从本端input buffer到对端output buffer
    std::vector<std::vector<DataSlice>> srcSlicesByChannel(channelNum);
    std::vector<std::vector<DataSlice>> dstSlicesByChannel(channelNum);
    u64 dataCountToSend = dataCount_;
    u64 currentOffset = 0;
    // 根据UB大小限制，对数据进行切分
    while (dataCountToSend > 0) {
        u64 transferCount = dataCountToSend > maxLoopTransCount_ ? maxLoopTransCount_ : dataCountToSend;
        u64 transferSize = transferCount * dataTypeSize_;
        std::vector<u64> elemCountOut;
        std::vector<u64> sizeOut;
        std::vector<u64> elemOffset;
        CHK_RET(CalcDataSplitByChannel(transferCount, dataTypeSize_, channels, elemCountOut, sizeOut, elemOffset));
        for (u32 channelIdx = 0; channelIdx < channelNum; channelIdx++) {
            if (elemCountOut[channelIdx] == 0) {
                continue;
            }
            void* dstBufferPtr = static_cast<void*>(channels[channelIdx].remoteOutputGraphMode.addr);
            srcSlicesByChannel[channelIdx].emplace_back(
                param.inputPtr, currentOffset + elemOffset[channelIdx], sizeOut[channelIdx], elemCountOut[channelIdx]);
            dstSlicesByChannel[channelIdx].emplace_back(
                dstBufferPtr, currentOffset + elemOffset[channelIdx], sizeOut[channelIdx], elemCountOut[channelIdx]);
        }
        currentOffset = currentOffset + transferSize;
        dataCountToSend = dataCountToSend - transferCount;
    }
    for (u32 channelIdx = 0; channelIdx < channelNum; channelIdx++) {
        if (srcSlicesByChannel[channelIdx].empty()) {
            continue;
        }
        SlicesList sendSlicesList{srcSlicesByChannel[channelIdx], dstSlicesByChannel[channelIdx]};
        DataInfo sendInfo{channels[channelIdx], sendSlicesList};
        bool isDmaRead = (channels[channelIdx].protocol == CommProtocol::COMM_PROTOCOL_PCIE);
        if (isDmaRead) {
            CHK_RET(SendRead(sendInfo, dataThreads[channelIdx]));
        } else {
            CHK_RET(SendWrite(sendInfo, dataThreads[channelIdx]));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsSendExecutor::OrchestrateOpbase(
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
            "[InsSendExecutor][Orchestrate] maxLoopTransCount_[%llu] is 0, maxLoopTransSize_[%llu], "
            "dataTypeSize_[%llu].",
            maxLoopTransCount_, maxLoopTransSize_, dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);

    u32 channelNum = static_cast<u32>(channels.size());
    HCCL_INFO(
        "[InsSendExecutor][Orchestrate][%d]->[%d] OPBASE channelNum[%u], threadNum[%zu], Generating tasks.", myRank_,
        remoteRank_, channelNum, dataThreads.size());

    u64 dataCountToSend = dataCount_;
    u64 currentOffset = 0;
    // 根据UB和ccl buffer大小限制，对数据进行切分
    while (dataCountToSend > 0) {
        u64 transferCount = dataCountToSend > maxLoopTransCount_ ? maxLoopTransCount_ : dataCountToSend;
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
            DataSlice inputSlice{
                param.inputPtr, currentOffset + elemOffset[channelIdx], sizeOut[channelIdx], elemCountOut[channelIdx]};
            // 因ccl buffer大小限制，每次往ccl buffer写一片数据
            DataSlice remoteCclSlice{
                channels[channelIdx].remoteCclMem.addr, elemOffset[channelIdx], sizeOut[channelIdx],
                elemCountOut[channelIdx]};
            if (isDmaRead) {
                // Read模式下，先拷贝到自己的hcclBuffer上，再通知对端来读取数据
                DataSlice cclSlice{
                    resCtx.cclMem.addr, elemOffset[channelIdx], sizeOut[channelIdx], elemCountOut[channelIdx]};
                CHK_RET(LocalCopy(dataThreads[channelIdx], inputSlice, cclSlice));
                SlicesList sendSlicesList{{cclSlice}, {remoteCclSlice}};
                DataInfo sendInfo{channels[channelIdx], sendSlicesList};
                CHK_RET(SendRead(sendInfo, dataThreads[channelIdx]));
            } else {
                // Write模式下，将数据直接写入对端cclBuffer
                SlicesList sendSlicesList{{inputSlice}, {remoteCclSlice}};
                DataInfo sendInfo{channels[channelIdx], sendSlicesList};
                CHK_RET(SendWrite(sendInfo, dataThreads[channelIdx]));
            }
        }
        currentOffset = currentOffset + transferSize;
        dataCountToSend = dataCountToSend - transferCount;
    }

    return HcclResult::HCCL_SUCCESS;
}

std::vector<CostModelParam> InsSendExecutor::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)topoInfo;
    (void)algName;
    (void)param;
    return {{0.0f, 0.0f, 3.0f, 0.0f}};
}

AlgNetMeta InsSendExecutor::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)param;
    (void)algName;
    return BuildDefaultMeshNetMeta(topoInfo);
}

REGISTER_EXECUTOR_IMPL(HcclCMDType::HCCL_CMD_SEND, AicpuSendSoleMesh, InsSendExecutor);
REGISTER_ALG_ATTRS(AicpuSendSoleMesh, topo.supportLevel0Topos = LEVEL0_TOPO_ANY; topo.isSupportLevel0PcieMix = true;);
} // namespace ops_hccl
