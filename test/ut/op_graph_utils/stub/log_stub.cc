/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdarg>
#include <cstdio>

#include "log.h"

bool HcclCheckLogLevel(int logType, int moduleId)
{
    (void)moduleId;
    // 仅放行 ERROR/WARN 供失败定位; INFO/DEBUG 保持静默避免刷屏
    return (logType == HCCL_LOG_ERROR || logType == HCCL_LOG_WARN);
}

bool IsErrorToWarn() { return false; }

// 弱符号补齐 (真库未链接): stderr 打印, 供 Execute 链路失败定位
void DlogRecord(int32_t moduleId, int32_t level, const char* fmt, ...)
{
    (void)moduleId;
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[UT-LOG%d] ", level);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}
