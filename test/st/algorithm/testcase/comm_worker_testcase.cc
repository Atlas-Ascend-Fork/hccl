/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include "gtest/gtest.h"
#include "acl/acl_rt.h"
#include "adapter_error_manager_pub.h"
#include "comm_worker.h"
#include "comm_worker_mgr.h"
#include "hccl.h"

using namespace ops_hccl;

// libhccl.so按符号白名单（src/hccl.map）隐藏CommWorker/CommWorkerMgr内部符号，无法从库直接链接，
// 本用例将生产源码（comm_worker.cc/comm_worker_mgr.cc）随用例编译驱动；日志级别与ErrMgr上下文桩
// 参照UT的log_stub口径在用例内提供，DlogRecord/aclrt*由hccl-model-proxy桩框架提供
bool HcclCheckLogLevel(int logType, int moduleId)
{
    (void)logType;
    (void)moduleId;
    return true;
}

bool IsErrorToWarn() { return false; }

ErrContext haclrtGetErrMgrContext(void) { return ErrContext{}; }

void haclrtSetErrMgrContext(ErrContext errorContext) { (void)errorContext; }

namespace {
HcclComm FakeComm(uintptr_t seed) { return reinterpret_cast<HcclComm>(seed); }
} // namespace

class ST_COMM_WORKER_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        // 发起线程绑定device：CommWorkerMgr按发起线程当前device分桶管理worker
        (void)aclrtSetDevice(0);
    }
    void TearDown() override {}
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// 用例1：首次Submit创建worker，后续Submit复用同一线程
TEST_F(ST_COMM_WORKER_TEST, st_comm_worker_submit_reuse_same_thread)
{
    HcclComm comm = FakeComm(1);
    std::thread::id firstTid;
    std::thread::id secondTid;
    HcclResult firstRet = CommWorkerMgr::GetInstance().Submit(comm, [&firstTid]() -> HcclResult {
        firstTid = std::this_thread::get_id();
        return HCCL_SUCCESS;
    });
    HcclResult secondRet = CommWorkerMgr::GetInstance().Submit(comm, [&secondTid]() -> HcclResult {
        secondTid = std::this_thread::get_id();
        return HCCL_SUCCESS;
    });
    EXPECT_TRUE(firstRet == HCCL_SUCCESS);
    EXPECT_TRUE(secondRet == HCCL_SUCCESS);
    EXPECT_TRUE(firstTid == secondTid);                  // 后续Submit复用同一线程
    EXPECT_TRUE(firstTid != std::this_thread::get_id()); // 任务在worker线程执行而非调用线程
    CommWorkerMgr::GetInstance().Remove(comm);
}

// 用例2：Remove（Shutdown）后线程已join，worker对象再Submit返回错误；Mgr对同comm懒重建新worker
TEST_F(ST_COMM_WORKER_TEST, st_comm_worker_remove_then_submit)
{
    // worker对象级：Shutdown即Mgr::Remove的回收动作，同步等待在途任务收尾并join线程后返回
    CommWorker worker;
    ASSERT_TRUE(worker.Start() == HCCL_SUCCESS);
    HcclResult runRet = worker.Submit(nullptr, ErrContext{}, []() -> HcclResult {
        return HCCL_SUCCESS;
    });
    EXPECT_TRUE(runRet == HCCL_SUCCESS);
    worker.Shutdown();
    HcclResult deadRet = worker.Submit(nullptr, ErrContext{}, []() -> HcclResult {
        return HCCL_SUCCESS;
    });
    EXPECT_TRUE(deadRet == HCCL_E_INTERNAL); // 已随comm销毁回收，提交被拒

    // Mgr级：Remove后同comm再Submit懒重建新worker，任务正常执行
    HcclComm comm = FakeComm(2);
    HcclResult createRet = CommWorkerMgr::GetInstance().Submit(comm, []() -> HcclResult {
        return HCCL_SUCCESS;
    });
    EXPECT_TRUE(createRet == HCCL_SUCCESS);
    CommWorkerMgr::GetInstance().Remove(comm);
    bool taskDone = false;
    HcclResult recreateRet = CommWorkerMgr::GetInstance().Submit(comm, [&taskDone]() -> HcclResult {
        taskDone = true;
        return HCCL_SUCCESS;
    });
    EXPECT_TRUE(recreateRet == HCCL_SUCCESS);
    EXPECT_TRUE(taskDone);
    CommWorkerMgr::GetInstance().Remove(comm);
}

// 用例3：Remove与在途Submit并发时，在途任务正常执行并返回结果（不丢结果语义）
TEST_F(ST_COMM_WORKER_TEST, st_comm_worker_remove_concurrent_with_submit)
{
    HcclComm comm = FakeComm(3);
    std::atomic<bool> taskStarted(false);
    std::atomic<bool> taskCanFinish(false);
    HcclResult inFlightRet = HCCL_E_INTERNAL;
    std::thread submitThread([&]() {
        // 发起线程须绑定device（生产契约：发起协商的线程必已绑定comm对应device）；ST桩的device绑定
        // 为thread_local且未绑定时aclrtGetDevice返回-1，不绑定会使worker落入-1桶、与主线程Submit命中不同worker
        (void)aclrtSetDevice(0);
        inFlightRet = CommWorkerMgr::GetInstance().Submit(comm, [&]() -> HcclResult {
            taskStarted = true;
            while (!taskCanFinish.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return HCCL_SUCCESS;
        });
    });
    while (!taskStarted.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // 在途任务执行期间并发Remove：Remove阻塞等待任务收尾并join，不丢在途结果
    std::thread removeThread([&]() {
        CommWorkerMgr::GetInstance().Remove(comm);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 让Remove先进入等待收尾
    taskCanFinish = true;

    submitThread.join();
    removeThread.join();
    EXPECT_TRUE(inFlightRet == HCCL_SUCCESS); // 在途任务结果不丢，正常回传
}

// 用例4：task内抛异常时Submit返回HCCL_E_INTERNAL，且worker可继续接任务
TEST_F(ST_COMM_WORKER_TEST, st_comm_worker_task_exception_recoverable)
{
    HcclComm comm = FakeComm(4);
    HcclResult stdExRet = CommWorkerMgr::GetInstance().Submit(comm, []() -> HcclResult {
        throw std::runtime_error("mock std exception");
    });
    EXPECT_TRUE(stdExRet == HCCL_E_INTERNAL);
    HcclResult unknownExRet = CommWorkerMgr::GetInstance().Submit(comm, []() -> HcclResult {
        throw 1; // 非std::exception，覆盖catch(...)分支
    });
    EXPECT_TRUE(unknownExRet == HCCL_E_INTERNAL);

    bool taskDone = false;
    HcclResult recoverRet = CommWorkerMgr::GetInstance().Submit(comm, [&taskDone]() -> HcclResult {
        taskDone = true;
        return HCCL_SUCCESS;
    });
    EXPECT_TRUE(recoverRet == HCCL_SUCCESS); // worker存活，可继续接任务
    EXPECT_TRUE(taskDone);
    CommWorkerMgr::GetInstance().Remove(comm);
}

// 用例5：单任务槽在途任务未收尾时，并发Submit被立即拒绝，后到任务不会静默覆盖在途任务
TEST_F(ST_COMM_WORKER_TEST, st_comm_worker_concurrent_submit_rejected)
{
    HcclComm comm = FakeComm(5);
    std::atomic<bool> taskStarted(false);
    std::atomic<bool> taskCanFinish(false);
    HcclResult inFlightRet = HCCL_E_INTERNAL;
    std::thread submitThread([&]() {
        // 发起线程须绑定device（生产契约：发起协商的线程必已绑定comm对应device）；ST桩的device绑定
        // 为thread_local且未绑定时aclrtGetDevice返回-1，不绑定会使worker落入-1桶、与主线程Submit命中不同worker
        (void)aclrtSetDevice(0);
        inFlightRet = CommWorkerMgr::GetInstance().Submit(comm, [&]() -> HcclResult {
            taskStarted = true;
            while (!taskCanFinish.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return HCCL_SUCCESS;
        });
    });
    while (!taskStarted.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    HcclResult rejectedRet = CommWorkerMgr::GetInstance().Submit(comm, []() -> HcclResult {
        return HCCL_SUCCESS;
    });
    EXPECT_TRUE(rejectedRet == HCCL_E_INTERNAL); // 后到任务被拒，不会静默覆盖

    taskCanFinish = true;
    submitThread.join();
    EXPECT_TRUE(inFlightRet == HCCL_SUCCESS); // 在途任务不受影响
    CommWorkerMgr::GetInstance().Remove(comm);
}

// 用例6：worker线程经context恢复发起线程的device绑定（ctx桩语义对齐真实ACL后的行为验证）
TEST_F(ST_COMM_WORKER_TEST, st_comm_worker_ctx_restores_device_binding)
{
    HcclComm comm = FakeComm(6);
    int32_t workerDevId = -1;
    HcclResult ret = CommWorkerMgr::GetInstance().Submit(comm, [&workerDevId]() -> HcclResult {
        aclError aclRet = aclrtGetDevice(&workerDevId);
        return (aclRet == ACL_SUCCESS) ? HCCL_SUCCESS : HCCL_E_INTERNAL;
    });
    EXPECT_TRUE(ret == HCCL_SUCCESS);
    // 发起线程（SetUp绑定device 0）的ctx随任务下发，worker线程经SetCurrentContext恢复同一device绑定；
    // 若ctx桩回退为恒返回null（跳过SetCurrent降级），worker线程curr_dev_id保持UINT32_MAX，此处将为-1
    EXPECT_TRUE(workerDevId == 0);
    CommWorkerMgr::GetInstance().Remove(comm);
}
