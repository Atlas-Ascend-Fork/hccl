/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// hccl_op_utils.cc 外部依赖桩: hcomm GraphMode 族 / HcomGenerateCclOpTag / HcclRt 拷贝封装 /
// gert MallocWorkSpace (CrackCleaner 链路, 测试用例不触达) — 签名与被测源码逐条对照

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "hccl/hcom.h"
#include "hccl_rt_utils.h"

namespace ops_hccl {
struct OpParamGraphMode;
} // namespace ops_hccl

using ops_hccl::OpParamGraphMode;

extern "C" {
HcclResult HcclCreateOpParamGraphMode(OpParamGraphMode** opParam)
{
    (void)opParam;
    return HCCL_SUCCESS;
}

HcclResult HcclDestroyOpParamGraphMode(OpParamGraphMode* opParam)
{
    (void)opParam;
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeOpType(OpParamGraphMode* opParam, const char* opType)
{
    (void)opParam;
    (void)opType;
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeDataCount(OpParamGraphMode* opParam, const uint64_t* dataCount)
{
    (void)opParam;
    (void)dataCount;
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeRankSize(OpParamGraphMode* opParam, const uint32_t* rankSize)
{
    (void)opParam;
    (void)rankSize;
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeHCCLBufferSize(OpParamGraphMode* opParam, const uint64_t* hcclBufferSize)
{
    (void)opParam;
    (void)hcclBufferSize;
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeDataType(OpParamGraphMode* opParam, const HcclDataType dataType)
{
    (void)opParam;
    (void)dataType;
    return HCCL_SUCCESS;
}

HcclResult HcclSetAivSelectOpParamGraphMode(OpParamGraphMode* opParam, uint32_t aivCoreLimit)
{
    (void)opParam;
    (void)aivCoreLimit;
    return HCCL_SUCCESS;
}

HcclResult HcclSelectAlgGraphMode(
    const char* group, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclCMDType opType,
    uint32_t aivCoreLimit, bool* ifAiv, char* algName)
{
    (void)group;
    (void)count;
    (void)dataType;
    (void)op;
    (void)opType;
    (void)aivCoreLimit;
    *ifAiv = false;
    algName[0] = '\0';
    return HCCL_SUCCESS;
}

HcclResult HcclCalcOpResOnlineGraphMode(
    OpParamGraphMode* opParam, uint64_t* opMemSize, uint32_t* streamNum, uint32_t* taskNum, uint32_t* aivCoreNum)
{
    (void)opParam;
    *opMemSize = 0;
    *streamNum = 0;
    *taskNum = 0;
    *aivCoreNum = 0;
    return HCCL_SUCCESS;
}

// BuildTagVec 链路: 确定性 tag, 供断言
HcclResult HcomGenerateCclOpTag(const char* opType, s64 hcomComm, const char* group, char* sTag)
{
    (void)opType;
    (void)hcomComm;
    (void)group;
    snprintf(sTag, 32, "UT_BASE_TAG");
    return HCCL_SUCCESS;
}

HcclResult HcomGetRankSize(const char* group, uint32_t* rankSize)
{
    (void)group;
    *rankSize = 8; // 约定桩值, 与测试常量保持一致
    return HCCL_SUCCESS;
}
} // extern "C"

// gert 上下文成员函数桩: AllocVOutputTensor/SetEmptyOutputTensor 链路 (纯函数用例不触达, 供链接)
#include "exe_graph/runtime/eager_op_execution_context.h"

// 可配置零缓冲 (全局域供测试 extern 引用)
void* g_stubMallocWsRet = nullptr;

// 输出分配队列 (hcut::CtxBuilder 注册, 顺序弹出; 空 → nullptr 保持旧行为) — 全局域供测试 extern
std::vector<gert::Tensor*> g_stubOutputTensors;
size_t g_stubOutputCursor = 0;

namespace gert {

void* EagerOpExecutionContext::MallocWorkSpace(size_t size)
{
    if (::g_stubMallocWsRet != nullptr) {
        return ::g_stubMallocWsRet; // 测试注入: 调用方控制返回值
    }
    (void)size;
    return nullptr; // 默认失败 (保持向后兼容)
}

Tensor* EagerOpExecutionContext::MallocOutputTensor(
    size_t outputIndex, const gert::StorageShape& shape, const gert::StorageFormat& format, ge::DataType dtype)
{
    (void)outputIndex;
    (void)shape;
    (void)format;
    (void)dtype;
    if (g_stubOutputCursor < g_stubOutputTensors.size()) {
        return g_stubOutputTensors[g_stubOutputCursor++];
    }
    return nullptr;
}
} // namespace gert

// aclrt 流同步桩 (SyncStreamBeforeD2H 链路, 纯函数用例不触达, 供链接)
#include "acl/acl_rt.h"
// 可配置失败注入: 测试用例可设置 g_stubSyncStreamRet 为非 ACL_SUCCESS 模拟同步失败
aclError g_stubSyncStreamRet = ACL_SUCCESS;

aclError aclrtSynchronizeStream(aclrtStream stream)
{
    (void)stream;
    return g_stubSyncStreamRet;
}

// ---- aclrt 拷贝桩 (hccl_rt_utils.cc 真源编译, UT 不链真实 ACL): 真实搬运 + 可配置返回码 ----
// 可配置返回码: 测试注入 ACL 失败 (→ HCCL_E_RUNTIME) / ACL_ERROR_RT_FEATURE_NOT_SUPPORT (→ HCCL_E_NOT_SUPPORT)
aclError g_stubAclMemcpyRet = ACL_SUCCESS;
aclError g_stubAclMemcpyAsyncRet = ACL_SUCCESS;
aclError g_stubAclAddrCopyRet = ACL_SUCCESS;

// 同步拷贝: 真实 memcpy (D2H 用例依赖), 保留 count>destMax 防御 (桩层语义, 生产由驱动保证)
aclError aclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind)
{
    (void)kind;
    if (g_stubAclMemcpyRet != ACL_SUCCESS) {
        return g_stubAclMemcpyRet;
    }
    if (dst == nullptr || src == nullptr || count > destMax) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (count > 0) {
        memcpy(dst, src, count);
    }
    return ACL_SUCCESS;
}

// 异步拷贝: 真实 memcpy — CrackCleaner D2D 清零用例依赖零缓冲→crack 地址的实际搬运
aclError
aclrtMemcpyAsync(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind, aclrtStream stream)
{
    (void)kind;
    (void)stream;
    if (g_stubAclMemcpyAsyncRet != ACL_SUCCESS) {
        return g_stubAclMemcpyAsyncRet;
    }
    if (dst == nullptr || src == nullptr || count > destMax) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (count > 0) {
        memcpy(dst, src, count);
    }
    return ACL_SUCCESS;
}

// ADDR 二级指针语义: 拷贝目标/源为地址槽, 实际搬运 (*dst)+dstOff ← (*src)+srcOff
aclError aclrtMemcpyAsyncWithOffset(
    void** dst, size_t destMax, size_t dstDataOffset, const void** src, size_t count, size_t srcDataOffset,
    aclrtMemcpyKind kind, aclrtStream stream)
{
    (void)kind;
    (void)stream;
    if (g_stubAclAddrCopyRet != ACL_SUCCESS) {
        return g_stubAclAddrCopyRet;
    }
    if (dst == nullptr || src == nullptr || *dst == nullptr || *src == nullptr || count > destMax) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (count > 0) {
        memcpy(static_cast<char*>(*dst) + dstDataOffset, static_cast<const char*>(*src) + srcDataOffset, count);
    }
    return ACL_SUCCESS;
}

// gert::Tensor 构造链的GetSizeInBytes桩 (尺寸校验辅助, 与 toolkit 实现同口径)
#include "graph/types.h"
namespace ge {
int64_t GetSizeInBytes(int64_t element_count, DataType data_type)
{
    switch (data_type) {
        case DT_BOOL:
        case DT_INT8:
        case DT_UINT8:
            return element_count * 1;
        case DT_INT16:
        case DT_UINT16:
        case DT_FLOAT16:
        case DT_BF16:
            return element_count * 2;
        case DT_INT32:
        case DT_UINT32:
        case DT_FLOAT:
            return element_count * 4;
        case DT_INT64:
        case DT_UINT64:
        case DT_DOUBLE:
            return element_count * 8;
        default:
            return 0;
    }
}
} // namespace ge
