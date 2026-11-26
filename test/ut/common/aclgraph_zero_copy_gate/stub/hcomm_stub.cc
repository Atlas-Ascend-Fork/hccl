/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// aclmdlRICaptureGetInfo 桩：op_common.cc IsStreamInCaptureMode 直接调用该 runtime API，
// 本 UT 二进制内 .o 定义优先于 libacl_rt 解析（与 order_launch UT 同模式），
// 用例通过 g_utCaptureStatus/g_utCaptureRet 控制捕获态与查询失败。

#include "hcomm_stub.h"

#include <gtest/gtest.h>

#include "acl_base.h"
#include "acl_rt.h"

namespace ut_stub {
int g_utCaptureStatus = 0;
int g_utCaptureRet = 0;
unsigned g_utCaptureGetCalls = 0;

void UtCaptureStubReset()
{
    g_utCaptureStatus = 1; // ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE
    g_utCaptureRet = 0;    // ACL_SUCCESS
    g_utCaptureGetCalls = 0;
}
} // namespace ut_stub

aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus* status, aclmdlRI* modelRI)
{
    (void)stream;
    ++ut_stub::g_utCaptureGetCalls;
    if (status == nullptr || modelRI == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (ut_stub::g_utCaptureRet != 0) {
        return ut_stub::g_utCaptureRet;
    }
    *status = static_cast<aclmdlRICaptureStatus>(ut_stub::g_utCaptureStatus);
    *modelRI = reinterpret_cast<aclmdlRI>(0x99U);
    return ACL_SUCCESS;
}

// GetHcommVersion 桩（真实实现定义在 dlhcomm_function.cc，本 UT 不链接）：
// 返回 9.1.0（aclgraph gate 前提的最低版本），本 UT 用例不依赖版本差异行为。
extern "C" int GetHcommVersion(void) { return CANN_VERSION(9, 1, 0); }

// HcclCommSymWinGet 桩（DECL_WEAK_FUNC 弱符号，此处强定义覆盖）：
// aclgraph gate 用例不应触发对称窗口探测（gate 对称条件命中即跳过），被调到即报失败。
extern "C" HcclResult
HcclCommSymWinGet(HcclComm comm, void* ptr, size_t size, HcclCommSymWindow* winHandle, size_t* offset)
{
    (void)comm;
    (void)ptr;
    (void)size;
    (void)winHandle;
    (void)offset;
    ADD_FAILURE() << "aclgraph gate用例不应调用HcclCommSymWinGet";
    return HCCL_E_NOT_SUPPORT;
}
