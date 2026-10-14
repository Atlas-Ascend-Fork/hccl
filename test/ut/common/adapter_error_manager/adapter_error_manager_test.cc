/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <thread>
#include "adapter_error_manager_pub.h"
#include "base/err_mgr.h"
#include "comm_worker.h"

using namespace ops_hccl;

// securec桩的失败注入开关（定义见stub/securec_stub.cc）
namespace ut_stub {
void SetMemcpySFail(bool enable);
}

namespace {
constexpr uint64_t RESERVED_BASE = 0x9ABCDEF012345678ULL;
constexpr size_t RESERVED_SIZE = sizeof(ErrContext::reserved) / sizeof(ErrContext::reserved[0]);

ErrContext MakeDistinctErrContext(uint64_t seed)
{
    ErrContext ctx;
    ctx.work_stream_id = 0x1234567890ABCDEFULL * seed + 7;
    for (size_t i = 0; i < RESERVED_SIZE; i++) {
        ctx.reserved[i] = RESERVED_BASE + seed * 0x100ULL + i;
    }
    return ctx;
}

error_message::ErrorManagerContext MakeDistinctSdkContext(uint64_t seed)
{
    error_message::ErrorManagerContext ctx;
    ctx.work_stream_id = 0x1234567890ABCDEFULL * seed + 7;
    for (size_t i = 0; i < RESERVED_SIZE; i++) {
        ctx.reserved[i] = RESERVED_BASE + seed * 0x100ULL + i;
    }
    return ctx;
}

// SDK上下文与本地ErrContext逐字段比对（两结构体字段一一对应：work_stream_id + reserved[7]）
bool SdkCtxEqualsErrCtx(const error_message::ErrorManagerContext& sdkCtx, const ErrContext& localCtx)
{
    if (sdkCtx.work_stream_id != localCtx.work_stream_id) {
        return false;
    }
    for (size_t i = 0; i < RESERVED_SIZE; i++) {
        if (sdkCtx.reserved[i] != localCtx.reserved[i]) {
            return false;
        }
    }
    return true;
}

bool IsSameErrContext(const ErrContext& lhs, const ErrContext& rhs)
{
    if (lhs.work_stream_id != rhs.work_stream_id) {
        return false;
    }
    for (size_t i = 0; i < RESERVED_SIZE; i++) {
        if (lhs.reserved[i] != rhs.reserved[i]) {
            return false;
        }
    }
    return true;
}
} // namespace

// 用例间复位memcpy_s失败注入开关，避免失败分支用例污染后续用例（含worker线程内的正常拷贝）
class AdapterErrorManagerTest : public ::testing::Test {
protected:
    void SetUp() override { ut_stub::SetMemcpySFail(false); }
    void TearDown() override { ut_stub::SetMemcpySFail(false); }
};

// 用例1：haclrtGetErrMgrContext对SDK上下文字段的完整拷贝——SDK侧注入非零work_stream_id与reserved，
// 断言转换后的ErrContext逐字段一致（覆盖work_stream_id直赋与reserved[7]经memcpy_s拷贝的映射正确性）
TEST_F(AdapterErrorManagerTest, HaclrtGetErrMgrContextCopiesSdkFieldsByField)
{
    error_message::ErrorManagerContext sdkCtx = MakeDistinctSdkContext(1);
    error_message::SetErrMgrContext(sdkCtx); // 桩记录SDK侧上下文，模拟真实ErrorMgr当前上下文

    ErrContext localCtx = haclrtGetErrMgrContext();
    EXPECT_TRUE(SdkCtxEqualsErrCtx(sdkCtx, localCtx)); // 逐字段一致，无遗漏无错位
}

// 用例2：haclrtSetErrMgrContext的Set入参逐字段透传到SDK侧
TEST_F(AdapterErrorManagerTest, HaclrtSetErrMgrContextPassesFieldsToSdk)
{
    ErrContext inputCtx = MakeDistinctErrContext(2);
    haclrtSetErrMgrContext(inputCtx);

    error_message::ErrorManagerContext recordedCtx = error_message::GetErrMgrContext(); // 桩记录的Set入参
    EXPECT_TRUE(SdkCtxEqualsErrCtx(recordedCtx, inputCtx));
}

// 用例3：Get的memcpy_s失败分支——work_stream_id为直赋不受影响，reserved保持初值零
TEST_F(AdapterErrorManagerTest, HaclrtGetErrMgrContextMemcpyFailKeepsReservedZero)
{
    error_message::ErrorManagerContext sdkCtx = MakeDistinctSdkContext(4);
    error_message::SetErrMgrContext(sdkCtx);
    ut_stub::SetMemcpySFail(true); // 注入memcpy_s失败

    ErrContext localCtx = haclrtGetErrMgrContext();
    EXPECT_TRUE(localCtx.work_stream_id == sdkCtx.work_stream_id); // 直赋字段保留
    for (size_t i = 0; i < RESERVED_SIZE; i++) {
        EXPECT_TRUE(localCtx.reserved[i] == 0); // 拷贝失败，reserved保持NSDMI初值零
    }
}

// 用例4：Set的memcpy_s失败分支——提前返回，SDK侧上下文保持基线未被覆盖
TEST_F(AdapterErrorManagerTest, HaclrtSetErrMgrContextMemcpyFailSkipsSdkSet)
{
    error_message::ErrorManagerContext baselineCtx = MakeDistinctSdkContext(5);
    error_message::SetErrMgrContext(baselineCtx); // 先建立SDK侧基线
    ut_stub::SetMemcpySFail(true);                // 注入memcpy_s失败

    haclrtSetErrMgrContext(MakeDistinctErrContext(6));
    error_message::ErrorManagerContext recordedCtx = error_message::GetErrMgrContext();
    EXPECT_TRUE(SdkCtxEqualsErrCtx(recordedCtx, MakeDistinctErrContext(5))); // SDK侧仍为基线
}

// 用例5：CommWorker::Submit→WorkerLoop间errCtx传递与恢复时序——发起线程采集errCtx随任务下发，
// WorkerLoop先于task执行恢复到worker线程，task内经haclrtGetErrMgrContext读回逐字段一致；
// 若WorkerLoop未先恢复，worker线程的线程级桩上下文为初值零，断言即失败（时序由断言隐式校验）
TEST(CommWorkerErrCtxTest, SubmitRelaysAndRestoresErrCtxBeforeTaskInWorkerThread)
{
    error_message::SetErrMgrContext(MakeDistinctSdkContext(3)); // 发起线程注入非零上下文
    ErrContext capturedCtx = haclrtGetErrMgrContext();          // 模拟CommWorkerMgr::Submit在发起线程采集

    CommWorker worker;
    ASSERT_TRUE(worker.Start() == HCCL_SUCCESS);
    ErrContext seenInWorkerCtx{};
    std::thread::id workerTid{};
    HcclResult submitRet = worker.Submit(nullptr, capturedCtx, [&]() -> HcclResult {
        workerTid = std::this_thread::get_id();
        seenInWorkerCtx = haclrtGetErrMgrContext();
        return HCCL_SUCCESS;
    });
    EXPECT_TRUE(submitRet == HCCL_SUCCESS);
    EXPECT_TRUE(workerTid != std::this_thread::get_id());        // task确在worker线程执行
    EXPECT_TRUE(IsSameErrContext(seenInWorkerCtx, capturedCtx)); // 透传恢复后worker线程读回与采集值一致
    worker.Shutdown();
}
