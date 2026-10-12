/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_stub.h"

#include <map>
#include "hcomm_primitives.h"

namespace biar_ut {
namespace {
    std::map<StubApi, int32_t>& FailureTable()
    {
        static std::map<StubApi, int32_t> table;
        return table;
    }

    int32_t RecordImpl(const StubCall& call)
    {
        StubCalls().push_back(call);
        auto it = FailureTable().find(call.api);
        return it == FailureTable().end() ? 0 : it->second;
    }
} // namespace

int32_t RecordCall(const StubCall& call) { return RecordImpl(call); }

std::vector<StubCall>& StubCalls()
{
    static std::vector<StubCall> calls;
    return calls;
}

void ClearStubCalls() { StubCalls().clear(); }

void SetStubFailure(StubApi api, int32_t code) { FailureTable()[api] = code; }

void ClearStubFailure() { FailureTable().clear(); }
} // namespace biar_ut

extern "C" {

int32_t HcommLocalReduceOnThread(
    ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    return biar_ut::RecordImpl(
        {biar_ut::StubApi::kLocalReduce, thread, 0, dst, src, count, static_cast<int>(dataType),
         static_cast<int>(reduceOp), 0, ""});
}

int32_t HcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    return biar_ut::RecordImpl({biar_ut::StubApi::kLocalCopy, thread, 0, dst, src, len, -1, -1, 0, ""});
}

int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    return biar_ut::RecordImpl({biar_ut::StubApi::kWrite, thread, channel, dst, src, len, -1, -1, 0, ""});
}

int32_t HcommWriteReduceOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    return biar_ut::RecordImpl(
        {biar_ut::StubApi::kWriteReduce, thread, channel, dst, src, count, static_cast<int>(dataType),
         static_cast<int>(reduceOp), 0, ""});
}

int32_t HcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    return biar_ut::RecordImpl({biar_ut::StubApi::kRead, thread, channel, dst, src, len, -1, -1, 0, ""});
}

int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    return biar_ut::RecordImpl(
        {biar_ut::StubApi::kNotifyRecord, thread, channel, nullptr, nullptr, 0, -1, -1, remoteNotifyIdx, ""});
}

int32_t
HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
{
    (void)timeOut;
    return biar_ut::RecordImpl(
        {biar_ut::StubApi::kNotifyWait, thread, channel, nullptr, nullptr, 0, -1, -1, localNotifyIdx, ""});
}

int32_t HcommAcquireComm(const char* commId)
{
    return biar_ut::RecordImpl(
        {biar_ut::StubApi::kAcquireComm, 0, 0, nullptr, nullptr, 0, -1, -1, 0, commId == nullptr ? "" : commId});
}

int32_t HcommReleaseComm(const char* commId)
{
    return biar_ut::RecordImpl(
        {biar_ut::StubApi::kReleaseComm, 0, 0, nullptr, nullptr, 0, -1, -1, 0, commId == nullptr ? "" : commId});
}

int32_t HcommBatchModeStart(const char* batchTag)
{
    return biar_ut::RecordImpl(
        {biar_ut::StubApi::kBatchModeStart, 0, 0, nullptr, nullptr, 0, -1, -1, 0, batchTag == nullptr ? "" : batchTag});
}

int32_t HcommBatchModeEnd(const char* batchTag)
{
    return biar_ut::RecordImpl(
        {biar_ut::StubApi::kBatchModeEnd, 0, 0, nullptr, nullptr, 0, -1, -1, 0, batchTag == nullptr ? "" : batchTag});
}
}
