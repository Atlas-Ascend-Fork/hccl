/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 算子层 UT (M2/M10) 桩: hccl_custom_op.cc / hcom_all_reduce_op.cc / hcom_reduce_scatter_op.cc
// 三个真实 TU 链接进 UT 后的全部外部依赖 — 算子层用例只直测纯布局函数,
// 引擎/通信/上下文入口一律 SUCCESS/空值桩化, 仅供链接 (用例不触达)
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "stub_comm_recorder.h"

#include "acl/acl_rt.h"
#include "hccl/hccl_comm.h"
#include "hccl/hcom.h"
#include "hccl_rt_utils.h"
#include "log.h"
#include "runtime/rt_external_model.h"

// ---- 引擎 ctx / 专用线程 (GetHcclUnfoldStream 链路) ----
// ThreadHandle/CommEngine 等内部类型来自 hccl.h 侧公共定义

// 记录桩向量定义 (stub_comm_recorder.h 持 extern 声明与结构体)
std::vector<StubCommCall> g_stubArCalls;
std::vector<StubCommCall> g_stubRsCalls;
std::vector<StubCommCall> g_stubSendCalls;
std::vector<StubCommCall> g_stubBcastCalls;
std::vector<StubCommCall> g_stubReduceCalls;
std::vector<StubCommCall> g_stubRecvCalls;

extern "C" {

// hcomm 通信域/CCL buffer 族
HcclResult HcomGetCommHandleByGroup(const char* group, HcclComm* comm)
{
    (void)group;
    *comm = reinterpret_cast<HcclComm>(0x30);
    return HCCL_SUCCESS;
}

// CCL buffer: 返回真实测试内存 (Execute 链路 D2D 拷贝真实落盘), 尺寸可配置 (Refresh 多迭代控制)
static uint8_t g_stubCclInBuf[1U << 20];
static uint8_t g_stubCclOutBuf[1U << 20];
uint64_t g_stubCclBufSize = 1U << 20;

HcclResult HcomGetInCCLbuffer(const char* group, void** ptr, u64* size)
{
    (void)group;
    *ptr = g_stubCclInBuf;
    *size = g_stubCclBufSize;
    return HCCL_SUCCESS;
}

HcclResult HcomGetOutCCLbuffer(const char* group, void** ptr, u64* size)
{
    (void)group;
    *ptr = g_stubCclOutBuf;
    *size = g_stubCclBufSize;
    return HCCL_SUCCESS;
}

HcclResult HcomCreateCommCCLbuffer(const char* group)
{
    (void)group;
    return HCCL_SUCCESS;
}

HcclResult HcomGetRankId(const char* group, u32* rankId)
{
    (void)group;
    *rankId = 0;
    return HCCL_SUCCESS;
}

// 通信入口 (算子层用例不触达, 供链接)
HcclResult HcclAllReduceGraphMode(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclReduceOp op, const char* group,
    aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,
    uint64_t scratchMemSize)
{
    (void)dataType;
    (void)op;
    (void)group;
    (void)stream;
    (void)streams;
    (void)streamCount;
    (void)scratchMemAddr;
    (void)scratchMemSize;
    g_stubArCalls.push_back({sendBuf, recvBuf, sendCount, tag != nullptr ? tag : ""});
    return HCCL_SUCCESS;
}

HcclResult HcclReduceScatterGraphMode(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, const char* group,
    aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,
    uint64_t scratchMemSize)
{
    (void)dataType;
    (void)op;
    (void)group;
    (void)stream;
    (void)streams;
    (void)streamCount;
    (void)scratchMemAddr;
    (void)scratchMemSize;
    g_stubRsCalls.push_back({sendBuf, recvBuf, recvCount, tag != nullptr ? tag : ""});
    return HCCL_SUCCESS;
}

HcclResult HcclSendGraphMode(
    void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, const char* group, aclrtStream stream,
    const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    (void)dataType;
    (void)destRank;
    (void)group;
    (void)stream;
    (void)streams;
    (void)streamCount;
    (void)scratchMemAddr;
    (void)scratchMemSize;
    g_stubSendCalls.push_back({sendBuf, nullptr, count, tag != nullptr ? tag : ""});
    return HCCL_SUCCESS;
}

HcclResult HcclBroadcastGraphMode(
    void* buf, uint64_t count, HcclDataType dataType, uint32_t root, const char* group, aclrtStream stream,
    const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    (void)dataType;
    (void)root;
    (void)group;
    (void)stream;
    (void)streams;
    (void)streamCount;
    (void)scratchMemAddr;
    (void)scratchMemSize;
    g_stubBcastCalls.push_back({buf, nullptr, count, tag != nullptr ? tag : ""});
    return HCCL_SUCCESS;
}
} // extern "C"

// ---- 引擎 ctx / 专用线程 (GetHcclUnfoldStream 链路, 用例不触达) ----
#include "hccl/hccl_res.h"

static uint64_t g_stubCtxSlot = 0x70;
static void* g_stubStream = nullptr;

HcclResult HcclEngineCtxGet(HcclComm comm, const char* ctxTag, CommEngine engine, void** ctx, uint64_t* size)
{
    (void)comm;
    (void)ctxTag;
    (void)engine;
    *ctx = &g_stubCtxSlot;
    *size = sizeof(g_stubCtxSlot);
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxCreate(HcclComm comm, const char* ctxTag, CommEngine engine, uint64_t size, void** ctx)
{
    (void)comm;
    (void)ctxTag;
    (void)engine;
    (void)size;
    *ctx = &g_stubCtxSlot;
    return HCCL_SUCCESS;
}

// 注册门控缺符号 UT: 隐藏宏按接口独立 — HCCL_UT_HIDE_DEDICATED_ACQUIRE /
// HCCL_UT_HIDE_THREAD_RES_INFO (部分缺符号矩阵: 两接口引入版本线不同, 只缺其一是真实
// 部署形态); 旧 HCCL_UT_HIDE_HCOMM_RES_APIS = 双隐藏等价别名。hccl_custom_op.cc 弱引用
// 对应解析为 null, 门控降级分支可被断言
#if !defined(HCCL_UT_HIDE_HCOMM_RES_APIS) && !defined(HCCL_UT_HIDE_DEDICATED_ACQUIRE)
HcclResult HcclDedicatedThreadAcquire(
    HcclComm comm, HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread)
{
    (void)comm;
    (void)useType;
    (void)notifyNumPerThread;
    *thread = 1;
    return HCCL_SUCCESS;
}
#endif

#if !defined(HCCL_UT_HIDE_HCOMM_RES_APIS) && !defined(HCCL_UT_HIDE_THREAD_RES_INFO)
HcclResult
HcclThreadResGetInfo(HcclComm comm, ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void** info)
{
    (void)comm;
    (void)thread;
    (void)resType;
    (void)infoLen;
    *info = &g_stubStream;
    return HCCL_SUCCESS;
}
#endif

extern "C" {
HcclResult HcclSetAivCoreLimitGraphMode(const char* group, uint32_t aivCoreLimit)
{
    (void)group;
    (void)aivCoreLimit;
    return HCCL_SUCCESS;
}

HcclResult HcclAllReduce(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    (void)dataType;
    (void)op;
    (void)comm;
    (void)stream;
    g_stubArCalls.push_back({sendBuf, recvBuf, count, ""});
    return HCCL_SUCCESS;
}

HcclResult HcclReduceScatter(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    (void)sendBuf;
    (void)recvBuf;
    (void)recvCount;
    (void)dataType;
    (void)op;
    (void)comm;
    (void)stream;
    return HCCL_SUCCESS;
}
} // extern "C"

HcclResult HcomSetAttachedStream(const char* group, u32 graphId, const rtStream_t* stream, s32 len)
{
    (void)group;
    (void)graphId;
    (void)stream;
    (void)len;
    return HCCL_SUCCESS;
}

// ---- aclrt 流查询 (RequestFollowStreams 链路, 用例不触达) ----
aclError aclrtStreamGetId(aclrtStream stream, int32_t* streamId)
{
    (void)stream;
    *streamId = 0;
    return ACL_SUCCESS;
}

// ---- gert 上下文成员 (GetStream / MallocReadOnlyDevArgs / RuntimeAttrs / UpdateArgsContext, 供链接) ----
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "exe_graph/runtime/update_args_context.h"

const void* g_stubAttrs[16] = {nullptr}; // 属性注册表 (hcut::CtxBuilder 按 index 写入)

namespace gert {
rtStream EagerOpExecutionContext::GetStream() const { return reinterpret_cast<rtStream>(0x20); }

const void* RuntimeAttrs::GetPointerByIndex(size_t index) const { return (index < 16) ? g_stubAttrs[index] : nullptr; }

const KernelArgs* UpdateArgsContext::GetKernelArgs(Placement placement, size_t index) const
{
    (void)placement;
    (void)index;
    return nullptr;
}
} // namespace gert

// ---- ge::AscendString (graph 库符号, 供链接) ----
#include <memory>
#include "graph/ascend_string.h"

namespace ge {
AscendString::AscendString(const char_t* const name)
    : name_(name == nullptr ? nullptr : std::make_shared<std::string>(name))
{}

AscendString::AscendString(const char_t* const name, size_t length)
    : name_(name == nullptr ? nullptr : std::make_shared<std::string>(name, length))
{}

const char_t* AscendString::GetString() const { return (name_ == nullptr) ? nullptr : name_->c_str(); }

size_t AscendString::GetLength() const { return (name_ == nullptr) ? 0 : name_->size(); }
} // namespace ge

// ---- Execute 链路可配置桩 (hcom_execute_test 消费, 默认值保持旧行为) ----
#include "exe_graph/runtime/eager_op_execution_context.h"
#include <string>
#include <vector>

// 上下文信号 (GetContextValue): 按 ContextParam 枚举值索引 (kNum=7, 8 槽足够)
bool g_stubCtxValRet[8] = {false};
uint64_t g_stubCtxVal[8] = {0};

bool gert::EagerOpExecutionContext::GetContextValue(gert::ContextParam key, uint64_t& value) const
{
    const int idx = static_cast<int>(key);
    if (idx < 0 || idx >= 8 || !g_stubCtxValRet[idx]) {
        return false; // 未注入契约: isDynamicShape=false / needRefresh=false
    }
    value = g_stubCtxVal[idx];
    return true;
}

ge::graphStatus gert::EagerOpExecutionContext::GetOption(const ge::AscendString& key, ge::AscendString& value) const
{
    (void)key;
    (void)value;
    return ge::GRAPH_FAILED; // 未注入 provider: atomicCleanPolicy/ge.hardwareInfo 回退不启用
}

gert::rtStream gert::EagerOpExecutionContext::RequestAttachedStream(const ge::AscendString& key, uint32_t flags)
{
    (void)key;
    (void)flags;
    return reinterpret_cast<gert::rtStream>(0x10000); // 非 null 哑流 (FollowStreams 申请成功)
}

// args struct 设备侧缓冲 (Refresh 路径 ADDR 二级指针经其字段地址间接寻址)
static uint8_t g_stubArgsBuf[256];
static gert::KernelArgs g_stubDevArgs = gert::KernelArgs();

const gert::KernelArgs* gert::EagerOpExecutionContext::MallocReadOnlyDevArgs(void* hostArgs, size_t argsSize) const
{
    // 真实拷贝: Refresh 路径经 args 字段 device 地址做二级指针间接寻址, 读到的必须是调用方写入的值
    if (hostArgs != nullptr && argsSize <= sizeof(g_stubArgsBuf)) {
        memcpy(g_stubArgsBuf, hostArgs, argsSize);
        g_stubDevArgs.args_size = argsSize;
    }
    g_stubDevArgs.args_data = g_stubArgsBuf;
    return &g_stubDevArgs;
}

// rtModel 反查/注册 (RegisterUnfoldStreamToModel): model 出参必须非空 (主流未绑模型按异常报错)
static void* g_stubRtModelSlot = nullptr;
rtError_t rtGetModelByStream(rtStream_t stream, rtModel_t* mdl)
{
    (void)stream;
    *mdl = reinterpret_cast<rtModel_t>(&g_stubRtModelSlot);
    return RT_ERROR_NONE;
}

rtError_t rtModelAddHcclStream(rtStream_t stream, rtModel_t mdl)
{
    (void)stream;
    (void)mdl;
    return RT_ERROR_NONE;
}

HcclResult HcclGetRankSize(HcclComm comm, uint32_t* rankSize)
{
    (void)comm;
    *rankSize = 8; // 与 HcomGetRankSize 桩值一致
    return HCCL_SUCCESS;
}

// ---- send/broadcast TU 链接补齐 (extern "C" 与 C++ 混合, 用例不触达) ----
extern "C" {
HcclResult
HcclSend(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream)
{
    (void)sendBuf;
    (void)count;
    (void)dataType;
    (void)destRank;
    (void)comm;
    (void)stream;
    return HCCL_SUCCESS;
}

HcclResult
HcclBroadcast(void* buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    (void)buf;
    (void)count;
    (void)dataType;
    (void)root;
    (void)comm;
    (void)stream;
    return HCCL_SUCCESS;
}

HcclWorkflowMode HcomGetWorkflowMode() { return HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE; }

HcclResult HcomSetWorkflowMode(HcclWorkflowMode mode)
{
    (void)mode;
    return HCCL_SUCCESS;
}

void HcomSetLaunchKernelMode(bool state) { (void)state; }
} // extern "C"

// aclrt 内存管理 (C++ 链接, 与 aclrtSynchronizeStream 桩同款)
aclError aclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    (void)policy;
    *devPtr = malloc(size);
    return (*devPtr != nullptr) ? ACL_SUCCESS : ACL_ERROR_RT_MEMORY_ALLOCATION;
}

aclError aclrtFree(void* devPtr)
{
    free(devPtr);
    return ACL_SUCCESS;
}

// broadcast 动态路径输出引用 (AllocRefOutputsDynamic, 用例不触达; 失败场景在其之前拦截)
gert::Tensor* gert::EagerOpExecutionContext::MakeOutputRefInput(size_t outputIndex, size_t inputIndex) const
{
    (void)outputIndex;
    (void)inputIndex;
    return nullptr;
}

// ---- alltoallv / all_gather_v / recv TU 链接补齐 ----
extern "C" {
HcclResult HcclAlltoAllGraphMode(
    const void* sendBuf, uint64_t sendCount, HcclDataType sendType, const void* recvBuf, uint64_t recvCount,
    HcclDataType recvType, const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount,
    void* scratchMemAddr, uint64_t scratchMemSize)
{
    (void)sendBuf;
    (void)sendCount;
    (void)sendType;
    (void)recvBuf;
    (void)recvCount;
    (void)recvType;
    (void)group;
    (void)stream;
    (void)tag;
    (void)streams;
    (void)streamCount;
    (void)scratchMemAddr;
    (void)scratchMemSize;
    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllVGraphMode(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, const char* group, aclrtStream stream,
    const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    (void)sendBuf;
    (void)sendCounts;
    (void)sdispls;
    (void)sendType;
    (void)recvBuf;
    (void)recvCounts;
    (void)rdispls;
    (void)recvType;
    (void)group;
    (void)stream;
    (void)tag;
    (void)streams;
    (void)streamCount;
    (void)scratchMemAddr;
    (void)scratchMemSize;
    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllVCGraphMode(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,
    uint64_t scratchMemSize)
{
    (void)sendBuf;
    (void)sendCountMatrix;
    (void)sendType;
    (void)recvBuf;
    (void)recvType;
    (void)group;
    (void)stream;
    (void)tag;
    (void)streams;
    (void)streamCount;
    (void)scratchMemAddr;
    (void)scratchMemSize;
    return HCCL_SUCCESS;
}

HcclResult HcclAllGatherV(
    void* sendBuf, uint64_t sendCount, void* recvBuf, const void* recvCounts, const void* recvDispls,
    HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    (void)sendBuf;
    (void)sendCount;
    (void)recvBuf;
    (void)recvCounts;
    (void)recvDispls;
    (void)dataType;
    (void)comm;
    (void)stream;
    return HCCL_SUCCESS;
}

HcclResult HcclAllGatherVGraphMode(
    void* sendBuf, void* recvBuf, uint64_t sendCount, const void* recvCounts, const void* recvDispls,
    HcclDataType dataType, const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount,
    void* scratchMemAddr, uint64_t scratchMemSize)
{
    (void)sendBuf;
    (void)recvBuf;
    (void)sendCount;
    (void)recvCounts;
    (void)recvDispls;
    (void)dataType;
    (void)group;
    (void)stream;
    (void)tag;
    (void)streams;
    (void)streamCount;
    (void)scratchMemAddr;
    (void)scratchMemSize;
    return HCCL_SUCCESS;
}

HcclResult
HcclRecv(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream)
{
    (void)dataType;
    (void)srcRank;
    (void)comm;
    (void)stream;
    g_stubRecvCalls.push_back({recvBuf, nullptr, count, ""});
    return HCCL_SUCCESS;
}

HcclResult HcclRecvGraphMode(
    void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, const char* group, aclrtStream stream,
    const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    (void)dataType;
    (void)srcRank;
    (void)group;
    (void)stream;
    (void)streams;
    (void)streamCount;
    (void)scratchMemAddr;
    (void)scratchMemSize;
    g_stubRecvCalls.push_back({recvBuf, nullptr, count, tag != nullptr ? tag : ""});
    return HCCL_SUCCESS;
}
} // extern "C"

extern "C" {
// comm 版 AlltoAll 三兄弟 (签名对照 src/ops/all_to_all_v/all_to_all_v.h)
HcclResult HcclAlltoAll(
    const void* sendBuf, uint64_t sendCount, HcclDataType sendType, const void* recvBuf, uint64_t recvCount,
    HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    (void)sendBuf;
    (void)sendCount;
    (void)sendType;
    (void)recvBuf;
    (void)recvCount;
    (void)recvType;
    (void)comm;
    (void)stream;
    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllV(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    (void)sendBuf;
    (void)sendCounts;
    (void)sdispls;
    (void)sendType;
    (void)recvBuf;
    (void)recvCounts;
    (void)rdispls;
    (void)recvType;
    (void)comm;
    (void)stream;
    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllVC(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    HcclComm comm, aclrtStream stream)
{
    (void)sendBuf;
    (void)sendCountMatrix;
    (void)sendType;
    (void)recvBuf;
    (void)recvType;
    (void)comm;
    (void)stream;
    return HCCL_SUCCESS;
}
} // extern "C"

// ---- Reduce 通信入口桩 (0 元素回归用例消费) ----
extern "C" {
HcclResult HcclReduce(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root, HcclComm comm,
    aclrtStream stream)
{
    (void)dataType;
    (void)op;
    (void)root;
    (void)comm;
    (void)stream;
    g_stubReduceCalls.push_back({sendBuf, recvBuf, count, ""});
    return HCCL_SUCCESS;
}

HcclResult HcclReduceGraphMode(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root,
    const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,
    uint64_t scratchMemSize)
{
    (void)dataType;
    (void)op;
    (void)root;
    (void)group;
    (void)stream;
    (void)streams;
    (void)streamCount;
    (void)scratchMemAddr;
    (void)scratchMemSize;
    g_stubReduceCalls.push_back({sendBuf, recvBuf, count, tag != nullptr ? tag : ""});
    return HCCL_SUCCESS;
}
} // extern "C"
