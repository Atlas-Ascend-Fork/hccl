/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base/err_mgr.h"

namespace error_message {
// UT桩：模拟liberror_manager的线程级上下文存取——Set记录入参、Get返回本线程最近Set的上下文
// （初值零）。线程级语义与真实SDK一致，使errCtx跨线程透传链路（发起线程采集→worker线程恢复）在UT可验证。
// 用例经error_message::SetErrMgrContext注入SDK侧上下文、经GetErrMgrContext读取Set入参完成断言
thread_local ErrorManagerContext g_utSdkCtx = {};

ErrorManagerContext GetErrMgrContext() { return g_utSdkCtx; }

void SetErrMgrContext(ErrorManagerContext error_context) { g_utSdkCtx = error_context; }
} // namespace error_message
