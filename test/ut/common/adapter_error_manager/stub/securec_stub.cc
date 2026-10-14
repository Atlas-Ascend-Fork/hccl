/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <errno.h>
#include <string.h>
#include <securec.h>

namespace {
// 用例注入的失败开关：置true后memcpy_s恒返回失败（ERANGE），用于触达被测代码的memcpy_s失败分支。
// 仅在用例主线程同步切换，worker线程仅在开关为false期间经被测代码调用，无并发访问
bool g_utMemcpySFail = false;
} // namespace

namespace ut_stub {
void SetMemcpySFail(bool enable) { g_utMemcpySFail = enable; }
} // namespace ut_stub

// UT桩：UT环境不链接libc_sec（既有UT零SDK库依赖，全桩化），按securec边界检查语义实现memcpy_s：
// 正常路径与真实实现行为一致（拷贝并返回EOK）；空指针/长度不足按真实语义返回EINVAL/ERANGE且不拷贝
extern "C" errno_t memcpy_s(void* dest, size_t numberOfElements, const void* src, size_t count)
{
    if (g_utMemcpySFail) {
        return ERANGE;
    }
    if (dest == nullptr || src == nullptr) {
        return EINVAL;
    }
    if (numberOfElements < count) {
        return ERANGE;
    }
    (void)memcpy(dest, src, count);
    return EOK;
}
