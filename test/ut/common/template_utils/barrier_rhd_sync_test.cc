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
#include "barrier_rhd_sync_stub.h"
#include "ins_temp_barrier_rhd_aicpu.h"

using namespace ops_hccl;

namespace {
constexpr ThreadHandle MAIN_THREAD = 10;

TemplateResource MakeResource(u32 rank, const std::vector<u32>& ranks)
{
    TemplateResource resource;
    resource.threads.push_back(MAIN_THREAD);
    for (u32 peer : ranks) {
        if (peer == rank) {
            continue;
        }
        ChannelInfo channel;
        channel.remoteRank = peer;
        channel.locationType = ENDPOINT_LOC_TYPE_DEVICE;
        resource.channels[peer] = {channel};
    }
    return resource;
}

HcclResult RunBarrier(u32 rank, const std::vector<u32>& ranks)
{
    OpParam param{};
    InsTempBarrierRhdAicpu barrier(param, rank, {ranks});
    TemplateResource resource = MakeResource(rank, ranks);
    return barrier.KernelRun(param, TemplateDataParams{}, resource);
}
} // namespace

TEST(BarrierRhdSync, EnqueuesAllStepsOnMainThread)
{
    barrier_rhd_sync_ut::Reset();
    EXPECT_EQ(RunBarrier(0, {0, 1, 2, 3}), HCCL_SUCCESS);
    EXPECT_EQ(barrier_rhd_sync_ut::calls, (std::vector<std::string>{"send:1", "send:2"}));
    EXPECT_EQ(barrier_rhd_sync_ut::sendThreads, (std::vector<uint64_t>{MAIN_THREAD, MAIN_THREAD}));
}

TEST(BarrierRhdSync, EnqueuesNonPowerOfTwoRolesOnMainThread)
{
    // 6 成员中 0/1 为代表，2/3 被折叠，4/5 仅参与主体。
    const std::vector<std::vector<std::string>> expected{
        {"send:2", "send:1", "send:4", "send:2"},
        {"send:3", "send:0", "send:5", "send:3"},
        {"send:0", "send:0"},
        {"send:1", "send:1"},
        {"send:5", "send:0"},
        {"send:4", "send:1"},
    };
    for (u32 rank = 0; rank < expected.size(); ++rank) {
        SCOPED_TRACE(rank);
        barrier_rhd_sync_ut::Reset();
        EXPECT_EQ(RunBarrier(rank, {0, 1, 2, 3, 4, 5}), HCCL_SUCCESS);
        EXPECT_EQ(barrier_rhd_sync_ut::calls, expected[rank]);
        EXPECT_EQ(barrier_rhd_sync_ut::sendThreads, std::vector<uint64_t>(expected[rank].size(), MAIN_THREAD));
    }
}

TEST(BarrierRhdSync, EmptyPlanDoesNotTouchCommunicationResources)
{
    barrier_rhd_sync_ut::Reset();
    OpParam param{};
    InsTempBarrierRhdAicpu barrier(param, 0, {{0}});
    TemplateResource resource;
    EXPECT_EQ(barrier.KernelRun(param, TemplateDataParams{}, resource), HCCL_SUCCESS);
    EXPECT_TRUE(barrier_rhd_sync_ut::calls.empty());
}

TEST(BarrierRhdSync, RejectsMissingChannelsBeforeSubmittingAnyStep)
{
    barrier_rhd_sync_ut::Reset();
    OpParam param{};
    InsTempBarrierRhdAicpu barrier(param, 0, {{0, 1, 2, 3}});
    TemplateResource resource = MakeResource(0, {0, 1, 2, 3});
    resource.channels.erase(2);
    EXPECT_EQ(barrier.KernelRun(param, TemplateDataParams{}, resource), HCCL_E_INTERNAL);
    EXPECT_TRUE(barrier_rhd_sync_ut::calls.empty());
}

TEST(BarrierRhdSync, StopsBeforeFollowingStepAfterSubmitFailure)
{
    barrier_rhd_sync_ut::Reset();
    barrier_rhd_sync_ut::failSendCall = 2;
    EXPECT_EQ(RunBarrier(0, {0, 1, 2, 3, 4, 5}), HCCL_E_INTERNAL);
    EXPECT_EQ(barrier_rhd_sync_ut::calls, (std::vector<std::string>{"send:2", "send:1"}));
    EXPECT_EQ(barrier_rhd_sync_ut::sendThreads, (std::vector<uint64_t>{MAIN_THREAD, MAIN_THREAD}));
}
