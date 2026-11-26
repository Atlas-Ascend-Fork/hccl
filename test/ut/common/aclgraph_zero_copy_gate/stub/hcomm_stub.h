/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLGRAPH_GATE_HCOMM_STUB_H
#define ACLGRAPH_GATE_HCOMM_STUB_H

#include "hccl.h"
#include "hccl_host_comm_dl.h"

namespace ut_stub {
// 流捕获态桩：aclmdlRICaptureGetInfo 的可控行为
extern int g_utCaptureStatus;        // aclmdlRICaptureStatus 出参（NONE/ACTIVE/INVALIDATED）
extern int g_utCaptureRet;           // aclmdlRICaptureGetInfo 返回值（模拟 API 失败）
extern unsigned g_utCaptureGetCalls; // 查询调用计数
void UtCaptureStubReset();           // 恢复默认桩行为（ACTIVE + 成功）
} // namespace ut_stub

#endif // ACLGRAPH_GATE_HCOMM_STUB_H
