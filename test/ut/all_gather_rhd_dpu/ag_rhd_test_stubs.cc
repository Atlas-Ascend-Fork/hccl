/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ag_rhd_test_access.h"

#include "alg_data_trans_wrapper.h"
#include "channel.h"

namespace {

std::vector<ag_rhd_ut::CopyCall> g_copyCalls;
ag_rhd_ut::BatchModeStats g_batchModeStats;

s32 g_batchModeStartResult = 0;
s32 g_batchModeEndResult = 0;
s32 g_threadSyncResult = 0;
s32 g_sendRequestResult = 0;
s32 g_waitResponseResult = 0;
u32 g_sendMsgId = 1;
u32 g_waitResponseMsgId = 1;

} // namespace

namespace ag_rhd_ut {

std::vector<CopyCall>& CopyCalls() { return g_copyCalls; }

BatchModeStats& BatchModeCounts() { return g_batchModeStats; }

void SetBatchModeStartResult(s32 result) { g_batchModeStartResult = result; }

void SetBatchModeEndResult(s32 result) { g_batchModeEndResult = result; }

void SetThreadSyncResult(s32 result) { g_threadSyncResult = result; }

void SetSendRequestResult(s32 result) { g_sendRequestResult = result; }

void SetWaitResponseResult(s32 result) { g_waitResponseResult = result; }

void SetSendMsgId(u32 msgId) { g_sendMsgId = msgId; }

void SetWaitResponseMsgId(u32 msgId) { g_waitResponseMsgId = msgId; }

void ResetStubState()
{
    g_copyCalls.clear();
    g_batchModeStats = BatchModeStats{};
    g_batchModeStartResult = 0;
    g_batchModeEndResult = 0;
    g_threadSyncResult = 0;
    g_sendRequestResult = 0;
    g_waitResponseResult = 0;
    g_sendMsgId = 1;
    g_waitResponseMsgId = 1;
}

} // namespace ag_rhd_ut

namespace ops_hccl {

// 记录 host 侧拷贝指令（KernelRun 的 LocalDataCopy/PostLocalCopy 消费），供到达性断言使用；
// 返回值恒成功（本 UT 覆盖 batch 守卫窗口，不在 LocalCopy 上注入失败）
HcclResult LocalCopy(const ThreadHandle& thread, const DataSlice& srcSlice, const DataSlice& dstSlice)
{
    (void)thread;
    ag_rhd_ut::CopyCall call;
    call.srcAddr = srcSlice.addr_;
    call.srcOffset = srcSlice.offset_;
    call.srcSize = srcSlice.size_;
    call.dstAddr = dstSlice.addr_;
    call.dstOffset = dstSlice.offset_;
    call.dstSize = dstSlice.size_;
    ag_rhd_ut::CopyCalls().push_back(call);
    return HCCL_SUCCESS;
}

// 以下 6 个桩仅满足链接：InsTempAllGatherRHD/InsTempAllGatherRHDDPU 的 vtable 引用其
// CalcRes/KernelRun 等虚函数体，其内部对 channel 申请与传输原语的调用必须可解析；
// 本 UT 用例只驱动 DPU 版 KernelRun（host 侧装载/卸载与协议交互），不触达这些路径。
// 与 nhr_test_stubs.cc 的空桩口径一致（返回成功、无副作用）。
HcclResult CalcChannelRequestMesh1D(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMesh1DHighestHostRoce(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    return HCCL_SUCCESS;
}

HcclResult PreSyncInterThreads(
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxMainToSub)
{
    (void)mainThread;
    (void)subThreads;
    (void)notifyIdxMainToSub;
    return HCCL_SUCCESS;
}

HcclResult PostSyncInterThreads(
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxSubToMain)
{
    (void)mainThread;
    (void)subThreads;
    (void)notifyIdxSubToMain;
    return HCCL_SUCCESS;
}

HcclResult SendRecvRead(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread)
{
    (void)sendRecvInfo;
    (void)thread;
    return HCCL_SUCCESS;
}

HcclResult SendRecvBatchWrite(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread)
{
    (void)sendRecvInfo;
    (void)thread;
    return HCCL_SUCCESS;
}

} // namespace ops_hccl

// 与 hcomm_primitives.h（extern "C"）/hcomm_primitives_dl.h（extern "C" 弱声明）保持同签名；
// UT 强符号定义覆盖弱声明。Start/End 计数供守卫断言，其余按 g_*Result 透传注入值。
extern "C" {

int32_t HcommBatchModeStart(const char* batchTag)
{
    (void)batchTag;
    ag_rhd_ut::BatchModeCounts().startCount++;
    return g_batchModeStartResult;
}

int32_t HcommBatchModeEnd(const char* batchTag)
{
    (void)batchTag;
    ag_rhd_ut::BatchModeCounts().endCount++;
    return g_batchModeEndResult;
}

int32_t HcommThreadSynchronize(ThreadHandle thread)
{
    (void)thread;
    return g_threadSyncResult;
}

int32_t HcommSendRequest(uint64_t handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId)
{
    (void)handle;
    (void)msgTag;
    (void)src;
    (void)sizeByte;
    if (msgId != nullptr) {
        *msgId = g_sendMsgId;
    }
    return g_sendRequestResult;
}

int32_t HcommWaitResponse(uint64_t handle, void* dst, size_t sizeByte, uint32_t* msgId)
{
    (void)handle;
    (void)dst;
    (void)sizeByte;
    if (msgId != nullptr) {
        *msgId = g_waitResponseMsgId;
    }
    return g_waitResponseResult;
}

} // extern "C"
