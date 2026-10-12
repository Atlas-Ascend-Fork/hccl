/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BIAR_UT_HCOMM_STUB_H
#define BIAR_UT_HCOMM_STUB_H

#include <cstdint>
#include <string>
#include <vector>

namespace biar_ut {

enum class StubApi {
    kLocalReduce,
    kLocalCopy,
    kWrite,
    kWriteReduce,
    kRead,
    kNotifyRecord,
    kNotifyWait,
    kPreSync,
    kPostSync,
    kAcquireComm,
    kReleaseComm,
    kBatchModeStart,
    kBatchModeEnd,
};

struct StubCall {
    StubApi api;
    uint64_t thread = 0;
    uint64_t channel = 0;
    void* dst = nullptr;
    const void* src = nullptr;
    /* count 语义随接口而异：reduce 类为元素个数，copy/write/read 类为字节数 */
    uint64_t count = 0;
    int dataType = -1;
    int reduceOp = -1;
    uint32_t notifyIdx = 0;
    std::string tag;
};

std::vector<StubCall>& StubCalls();
void ClearStubCalls();
void SetStubFailure(StubApi api, int32_t code);
void ClearStubFailure();
int32_t RecordCall(const StubCall& call);

} // namespace biar_ut

#endif // BIAR_UT_HCOMM_STUB_H
