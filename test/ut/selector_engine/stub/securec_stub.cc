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
#include <cstring>
#include <securec.h>

// UT桩：UT环境不链接libc_sec（既有UT零SDK库依赖，全桩化），按securec边界检查语义实现
extern "C" errno_t memcpy_s(void* dest, size_t numberOfElements, const void* src, size_t count)
{
    if (dest == nullptr || src == nullptr) {
        return EINVAL;
    }
    if (numberOfElements < count) {
        return ERANGE;
    }
    (void)memcpy(dest, src, count);
    return EOK;
}

extern "C" int sprintf_s(char* strDest, size_t destMax, const char* format, ...)
{
    if (strDest == nullptr || format == nullptr || destMax == 0) {
        return -1;
    }
    va_list args;
    va_start(args, format);
    int ret = vsnprintf(strDest, destMax, format, args);
    va_end(args);
    if (ret < 0 || static_cast<size_t>(ret) >= destMax) {
        return -1;
    }
    return ret;
}
