/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cerrno>
#include <cstring>
#include <securec.h>

// UT不链接SDK的libc_sec；为被测代码提供拷贝功能及基本边界检查。
extern "C" errno_t memcpy_s(void* dest, size_t destMax, const void* src, size_t count)
{
    if (dest == nullptr || src == nullptr) {
        return EINVAL;
    }
    if (count > destMax) {
        if (destMax > 0) {
            std::memset(dest, 0, destMax);
        }
        return ERANGE;
    }
    std::memcpy(dest, src, count);
    return EOK;
}

extern "C" errno_t strcpy_s(char* dest, size_t destMax, const char* src)
{
    if (dest == nullptr || src == nullptr || destMax == 0) {
        return EINVAL;
    }
    const size_t length = std::strlen(src);
    if (length >= destMax) {
        dest[0] = '\0';
        return ERANGE;
    }
    std::memcpy(dest, src, length + 1);
    return EOK;
}
