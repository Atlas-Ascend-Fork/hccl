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
#include <vector>
#include "hcomm_stub.h"
#include "hcomm_primitives.h"
#include "all_reduce_biar.h"
#include "all_reduce_executor_base.h"

namespace {
using namespace ops_hccl;
using namespace ops_hccl_experimental;
using biar_ut::StubApi;
using biar_ut::StubCall;

void* const kInputBase = reinterpret_cast<void*>(0x10000000000ULL);
void* const kOutputBase = reinterpret_cast<void*>(0x20000000000ULL);
void* const kScratchBase = reinterpret_cast<void*>(0x30000000000ULL);
void* const kRemoteBase = reinterpret_cast<void*>(0x40000000000ULL);
constexpr u64 kMemSize = 1ULL << 20;
constexpr u64 kRemoteStep = 1ULL << 24;
constexpr u64 kStride = 256;
constexpr u32 kUnit = 4;
constexpr ThreadHandle kMainThread = 100;
constexpr ThreadHandle kSub0 = 101;
constexpr ThreadHandle kSub1 = 102;
constexpr ChannelHandle kHandleBase = 0x1000;

void* PtrAt(void* base, u64 offset) { return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + offset); }

u64 OffsetFrom(const void* ptr, const void* base)
{
    return static_cast<u64>(reinterpret_cast<uintptr_t>(ptr) - reinterpret_cast<uintptr_t>(base));
}

std::vector<Slice> MakeSlices(const std::vector<u64>& sizes)
{
    std::vector<Slice> slices;
    u64 offset = 0;
    for (u64 size : sizes) {
        slices.push_back(Slice{offset, size});
        offset += size;
    }
    return slices;
}

ChannelInfo MakeChannel(u32 remoteRank)
{
    ChannelInfo channel;
    channel.isValid = true;
    channel.remoteRank = remoteRank;
    channel.handle = kHandleBase + remoteRank;
    channel.remoteOutput = HcclMem{HCCL_MEM_TYPE_DEVICE, PtrAt(kRemoteBase, remoteRank * kRemoteStep), kMemSize};
    return channel;
}

std::vector<ChannelInfo> MakeChannels(u32 rankSize)
{
    std::vector<ChannelInfo> channels;
    for (u32 i = 0; i < rankSize; i++) {
        channels.push_back(MakeChannel(i));
    }
    return channels;
}

class BiarExecutorUnderTest : public AllReduceExecutorBase {
public:
    using AllReduceExecutorBase::PrepareDataSlice;
};

class BiarTemplateUnderTest : public AllReduceBIAR {
public:
    using AllReduceBIAR::HCCSProcessAgLoop;
    using AllReduceBIAR::HCCSProcessRsLoop;
    using AllReduceBIAR::LocalCopyAgLoop;
    using AllReduceBIAR::LocalCopyRsLoop;
    using AllReduceBIAR::Preprocess;
    using AllReduceBIAR::SIOProcessAgLoop;
    using AllReduceBIAR::SIOProcessRsLoop;
    using AllReduceBIAR::TreeReduceRs;

    void SetupMem(const std::vector<Slice>& slices, u64 count, HcclReduceOp op = HCCL_REDUCE_SUM, u32 unitSize = kUnit)
    {
        HcclMem inputMem{HCCL_MEM_TYPE_DEVICE, kInputBase, kMemSize};
        HcclMem outputMem{HCCL_MEM_TYPE_DEVICE, kOutputBase, kMemSize};
        HcclMem scratchMem{HCCL_MEM_TYPE_DEVICE, kScratchBase, kMemSize};
        std::vector<ThreadHandle> slaveThreads{kSub0, kSub1};
        AllReduceBIAR::Prepare(
            inputMem, outputMem, scratchMem, count, HCCL_DATA_TYPE_INT32, kMainThread, slaveThreads, op, 0, slices, 0,
            false);
        unitSize_ = unitSize;
    }

    void SetupTopology(
        u32 sioRank, const std::vector<u32>& hccsRanks, const std::vector<u32>& neighbourRanks,
        const std::vector<ChannelInfo>& channels)
    {
        sio_rank_ = sioRank;
        sio_link = channels[sioRank];
        hccs_ranks = hccsRanks;
        hccs_neighbour_rank = neighbourRanks;
        for (u32 hccsRank : hccsRanks) {
            hccs_links.push_back(channels[hccsRank]);
        }
        hccs_ranks_reversed.assign(hccs_ranks.rbegin(), hccs_ranks.rend());
        hccs_links_reversed.assign(hccs_links.rbegin(), hccs_links.rend());
    }

    u32 UnitSize() const { return unitSize_; }
};

class BiarCoreTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        biar_ut::ClearStubCalls();
        biar_ut::ClearStubFailure();
    }
    void TearDown() override
    {
        biar_ut::ClearStubCalls();
        biar_ut::ClearStubFailure();
    }
};

// ---------------- AllReduceExecutorBase::PrepareDataSlice ----------------

TEST_F(BiarCoreTest, PrepareDataSliceUniformSplit)
{
    BiarExecutorUnderTest executor;
    std::vector<Slice> slices;
    ASSERT_EQ(executor.PrepareDataSlice(1600, kUnit, 16, slices), HCCL_SUCCESS);
    ASSERT_EQ(slices.size(), 16u);
    for (u32 i = 0; i < 16; i++) {
        EXPECT_EQ(slices[i].size, 400u);
        EXPECT_EQ(slices[i].offset, static_cast<u64>(i) * 400);
    }
}

TEST_F(BiarCoreTest, PrepareDataSliceRemainderGoesToFirstRanks)
{
    BiarExecutorUnderTest executor;
    std::vector<Slice> slices;
    ASSERT_EQ(executor.PrepareDataSlice(1603, kUnit, 16, slices), HCCL_SUCCESS);
    ASSERT_EQ(slices.size(), 16u);
    u64 expectOffset = 0;
    for (u32 i = 0; i < 16; i++) {
        u64 expectSize = (i < 3) ? 404 : 400;
        EXPECT_EQ(slices[i].size, expectSize) << "slice " << i;
        EXPECT_EQ(slices[i].offset, expectOffset) << "slice " << i;
        expectOffset += expectSize;
    }
    EXPECT_EQ(expectOffset, 1603ULL * kUnit);
}

TEST_F(BiarCoreTest, PrepareDataSliceCountSmallerThanSliceNum)
{
    BiarExecutorUnderTest executor;
    std::vector<Slice> slices;
    ASSERT_EQ(executor.PrepareDataSlice(3, 2, 8, slices), HCCL_SUCCESS);
    ASSERT_EQ(slices.size(), 8u);
    const u64 expectSizes[8] = {2, 2, 2, 0, 0, 0, 0, 0};
    const u64 expectOffsets[8] = {0, 2, 4, 6, 6, 6, 6, 6};
    for (u32 i = 0; i < 8; i++) {
        EXPECT_EQ(slices[i].size, expectSizes[i]) << "slice " << i;
        EXPECT_EQ(slices[i].offset, expectOffsets[i]) << "slice " << i;
    }
}

TEST_F(BiarCoreTest, PrepareDataSliceZeroCountYieldsEmptySlices)
{
    BiarExecutorUnderTest executor;
    std::vector<Slice> slices;
    ASSERT_EQ(executor.PrepareDataSlice(0, kUnit, 4, slices), HCCL_SUCCESS);
    ASSERT_EQ(slices.size(), 4u);
    for (u32 i = 0; i < 4; i++) {
        EXPECT_EQ(slices[i].size, 0u);
        EXPECT_EQ(slices[i].offset, 0u);
    }
}

TEST_F(BiarCoreTest, PrepareDataSliceZeroSliceNumIsRejected)
{
    BiarExecutorUnderTest executor;
    std::vector<Slice> slices;
    EXPECT_EQ(executor.PrepareDataSlice(64, kUnit, 0, slices), HCCL_E_PARA);
    EXPECT_TRUE(slices.empty());
}

TEST_F(BiarCoreTest, PrepareDataSliceOddCountTwoSlices)
{
    BiarExecutorUnderTest executor;
    std::vector<Slice> slices;
    ASSERT_EQ(executor.PrepareDataSlice(5, 8, 2, slices), HCCL_SUCCESS);
    ASSERT_EQ(slices.size(), 2u);
    EXPECT_EQ(slices[0].size, 24u);
    EXPECT_EQ(slices[0].offset, 0u);
    EXPECT_EQ(slices[1].size, 16u);
    EXPECT_EQ(slices[1].offset, 24u);
}

// ---------------- AllReduceBIAR::TreeReduceRs ----------------

TEST_F(BiarCoreTest, TreeReduceFourRanksGroup0ReducesIntoHomeSlot)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 40}), 40);
    ASSERT_EQ(tpl.TreeReduceRs(0, 4, 2, kStride), HCCL_SUCCESS);
    const auto& calls = biar_ut::StubCalls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].api, StubApi::kLocalReduce);
    EXPECT_EQ(calls[0].thread, kMainThread);
    EXPECT_EQ(calls[0].dst, PtrAt(kScratchBase, 0));
    EXPECT_EQ(calls[0].src, PtrAt(kScratchBase, 3 * kStride));
    EXPECT_EQ(calls[0].count, 10u);
}

TEST_F(BiarCoreTest, TreeReduceFourRanksGroup1ReducesThenCopiesHome)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 40}), 40);
    ASSERT_EQ(tpl.TreeReduceRs(3, 4, 2, kStride), HCCL_SUCCESS);
    const auto& calls = biar_ut::StubCalls();
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_EQ(calls[0].api, StubApi::kLocalReduce);
    EXPECT_EQ(calls[0].dst, PtrAt(kScratchBase, 2 * kStride));
    EXPECT_EQ(calls[0].src, PtrAt(kScratchBase, 1 * kStride));
    EXPECT_EQ(calls[1].api, StubApi::kLocalCopy);
    EXPECT_EQ(calls[1].dst, PtrAt(kScratchBase, 1 * kStride));
    EXPECT_EQ(calls[1].src, PtrAt(kScratchBase, 2 * kStride));
    EXPECT_EQ(calls[1].count, 40u);
}

TEST_F(BiarCoreTest, TreeReduceEightRanksTwoLevelTreeOrder)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 40, 40, 40, 40, 40}), 40);
    ASSERT_EQ(tpl.TreeReduceRs(5, 8, 2, kStride), HCCL_SUCCESS);
    const auto& calls = biar_ut::StubCalls();
    ASSERT_EQ(calls.size(), 4u);
    EXPECT_EQ(calls[0].api, StubApi::kLocalReduce);
    EXPECT_EQ(OffsetFrom(calls[0].src, kScratchBase), 5 * kStride);
    EXPECT_EQ(OffsetFrom(calls[0].dst, kScratchBase), 4 * kStride);
    EXPECT_EQ(calls[1].api, StubApi::kLocalReduce);
    EXPECT_EQ(OffsetFrom(calls[1].src, kScratchBase), 7 * kStride);
    EXPECT_EQ(OffsetFrom(calls[1].dst, kScratchBase), 2 * kStride);
    EXPECT_EQ(calls[2].api, StubApi::kLocalReduce);
    EXPECT_EQ(OffsetFrom(calls[2].src, kScratchBase), 2 * kStride);
    EXPECT_EQ(OffsetFrom(calls[2].dst, kScratchBase), 4 * kStride);
    EXPECT_EQ(calls[3].api, StubApi::kLocalCopy);
    EXPECT_EQ(OffsetFrom(calls[3].src, kScratchBase), 4 * kStride);
    EXPECT_EQ(OffsetFrom(calls[3].dst, kScratchBase), 2 * kStride);
}

TEST_F(BiarCoreTest, TreeReduceZeroSliceProducesNoTraffic)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 0}), 30);
    ASSERT_EQ(tpl.TreeReduceRs(3, 4, 2, kStride), HCCL_SUCCESS);
    EXPECT_TRUE(biar_ut::StubCalls().empty());
}

TEST_F(BiarCoreTest, TreeReduceSingleGroupHasNoWork)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40}), 20);
    ASSERT_EQ(tpl.TreeReduceRs(1, 2, 2, kStride), HCCL_SUCCESS);
    EXPECT_TRUE(biar_ut::StubCalls().empty());
}

TEST_F(BiarCoreTest, TreeReducePropagatesDataTypeAndReduceOp)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 40}), 40, HCCL_REDUCE_MAX);
    ASSERT_EQ(tpl.TreeReduceRs(0, 4, 2, kStride), HCCL_SUCCESS);
    const auto& calls = biar_ut::StubCalls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].reduceOp, static_cast<int>(static_cast<HcommReduceOp>(HCCL_REDUCE_MAX)));
    EXPECT_EQ(calls[0].dataType, static_cast<int>(static_cast<HcommDataType>(HCCL_DATA_TYPE_INT32)));
}

TEST_F(BiarCoreTest, TreeReducePropagatesStubFailure)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 40}), 40);
    biar_ut::SetStubFailure(StubApi::kLocalReduce, static_cast<int32_t>(HCCL_E_INTERNAL));
    EXPECT_EQ(tpl.TreeReduceRs(0, 4, 2, kStride), HCCL_E_INTERNAL);
}

// ---------------- ReduceScatter 阶段循环 ----------------

TEST_F(BiarCoreTest, HccsRsLoopRound0ProducesNoTraffic)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 40}), 40);
    tpl.SetupTopology(1, {2}, {3}, MakeChannels(4));
    ASSERT_EQ(tpl.HCCSProcessRsLoop(0, 0, 4, 2, kStride), HCCL_SUCCESS);
    EXPECT_TRUE(biar_ut::StubCalls().empty());
}

TEST_F(BiarCoreTest, HccsRsLoopWritesIntoPeerGroupSlot)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 40}), 40);
    tpl.SetupTopology(1, {2}, {3}, MakeChannels(4));
    ASSERT_EQ(tpl.HCCSProcessRsLoop(1, 0, 4, 2, kStride), HCCL_SUCCESS);
    const auto& calls = biar_ut::StubCalls();
    ASSERT_EQ(calls.size(), 5u);
    EXPECT_EQ(calls[0].api, StubApi::kNotifyRecord);
    EXPECT_EQ(calls[0].thread, kSub0);
    EXPECT_EQ(calls[0].channel, kHandleBase + 2);
    EXPECT_EQ(calls[0].notifyIdx, NOTIFY_IDX_ACK);
    EXPECT_EQ(calls[1].api, StubApi::kNotifyWait);
    EXPECT_EQ(calls[1].notifyIdx, NOTIFY_IDX_ACK);
    EXPECT_EQ(calls[2].api, StubApi::kWrite);
    EXPECT_EQ(calls[2].dst, PtrAt(kRemoteBase, 2 * kRemoteStep + 2 * kStride));
    EXPECT_EQ(calls[2].src, PtrAt(kScratchBase, 1 * kStride));
    EXPECT_EQ(calls[2].count, 40u);
    EXPECT_EQ(calls[3].api, StubApi::kNotifyRecord);
    EXPECT_EQ(calls[3].notifyIdx, NOTIFY_IDX_DATA_SIGNAL);
    EXPECT_EQ(calls[4].api, StubApi::kNotifyWait);
    EXPECT_EQ(calls[4].notifyIdx, NOTIFY_IDX_DATA_SIGNAL);
}

TEST_F(BiarCoreTest, HccsRsLoopZeroSliceSkipsWriteButKeepsNotifies)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 0, 40}), 30);
    tpl.SetupTopology(1, {2}, {3}, MakeChannels(4));
    ASSERT_EQ(tpl.HCCSProcessRsLoop(1, 0, 4, 2, kStride), HCCL_SUCCESS);
    const auto& calls = biar_ut::StubCalls();
    ASSERT_EQ(calls.size(), 4u);
    for (const auto& call : calls) {
        EXPECT_NE(call.api, StubApi::kWrite);
    }
}

TEST_F(BiarCoreTest, SioRsLoopIntermediateRoundUsesNeighbourSlice)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 44}), 41);
    tpl.SetupTopology(1, {2}, {3}, MakeChannels(4));
    ASSERT_EQ(tpl.SIOProcessRsLoop(0, 0, 4, 2, kStride), HCCL_SUCCESS);
    const auto& calls = biar_ut::StubCalls();
    ASSERT_EQ(calls.size(), 5u);
    EXPECT_EQ(calls[0].api, StubApi::kNotifyRecord);
    EXPECT_EQ(calls[0].thread, kMainThread);
    EXPECT_EQ(calls[0].channel, kHandleBase + 1);
    EXPECT_EQ(calls[1].api, StubApi::kNotifyWait);
    EXPECT_EQ(calls[2].api, StubApi::kWriteReduce);
    EXPECT_EQ(calls[2].dst, PtrAt(kRemoteBase, 1 * kRemoteStep + 1 * kStride));
    EXPECT_EQ(calls[2].src, PtrAt(kInputBase, 120));
    EXPECT_EQ(calls[2].count, 11u);
    EXPECT_EQ(calls[3].api, StubApi::kNotifyRecord);
    EXPECT_EQ(calls[4].api, StubApi::kNotifyWait);
}

TEST_F(BiarCoreTest, SioRsLoopLastRoundUsesSioRankSlice)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 36, 40, 40}), 39);
    tpl.SetupTopology(1, {2}, {3}, MakeChannels(4));
    ASSERT_EQ(tpl.SIOProcessRsLoop(1, 0, 4, 2, kStride), HCCL_SUCCESS);
    const auto& calls = biar_ut::StubCalls();
    ASSERT_EQ(calls.size(), 5u);
    EXPECT_EQ(calls[2].api, StubApi::kWriteReduce);
    EXPECT_EQ(calls[2].dst, PtrAt(kRemoteBase, 1 * kRemoteStep + 0));
    EXPECT_EQ(calls[2].src, PtrAt(kInputBase, 40));
    EXPECT_EQ(calls[2].count, 9u);
}

TEST_F(BiarCoreTest, SioRsLoopZeroSliceSkipsWriteReduce)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 0}), 30);
    tpl.SetupTopology(1, {2}, {3}, MakeChannels(4));
    ASSERT_EQ(tpl.SIOProcessRsLoop(0, 0, 4, 2, kStride), HCCL_SUCCESS);
    const auto& calls = biar_ut::StubCalls();
    ASSERT_EQ(calls.size(), 4u);
    for (const auto& call : calls) {
        EXPECT_NE(call.api, StubApi::kWriteReduce);
    }
}

TEST_F(BiarCoreTest, LocalCopyRsLoopRound0CopiesOwnSliceOnly)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 40}), 40);
    tpl.SetupTopology(1, {2}, {3}, MakeChannels(4));
    ASSERT_EQ(tpl.LocalCopyRsLoop(0, 0, 4, 2, kStride), HCCL_SUCCESS);
    const auto& calls = biar_ut::StubCalls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].api, StubApi::kLocalCopy);
    EXPECT_EQ(calls[0].thread, kSub1);
    EXPECT_EQ(calls[0].dst, PtrAt(kScratchBase, 0));
    EXPECT_EQ(calls[0].src, PtrAt(kInputBase, 0));
    EXPECT_EQ(calls[0].count, 40u);

    biar_ut::ClearStubCalls();
    ASSERT_EQ(tpl.LocalCopyRsLoop(1, 0, 4, 2, kStride), HCCL_SUCCESS);
    EXPECT_TRUE(biar_ut::StubCalls().empty());
}

// ---------------- AllGather 阶段循环 ----------------

TEST_F(BiarCoreTest, HccsAgLoopRound0WritesHomeSlotToPeer)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 40}), 40);
    tpl.SetupTopology(1, {2}, {3}, MakeChannels(4));
    ASSERT_EQ(tpl.HCCSProcessAgLoop(0, 0, 4, 2, kStride), HCCL_SUCCESS);
    const auto& calls = biar_ut::StubCalls();
    ASSERT_EQ(calls.size(), 5u);
    EXPECT_EQ(calls[2].api, StubApi::kWrite);
    EXPECT_EQ(calls[2].dst, PtrAt(kRemoteBase, 2 * kRemoteStep + 0));
    EXPECT_EQ(calls[2].src, PtrAt(kScratchBase, 0));
    EXPECT_EQ(calls[2].count, 40u);

    biar_ut::ClearStubCalls();
    ASSERT_EQ(tpl.HCCSProcessAgLoop(1, 0, 4, 2, kStride), HCCL_SUCCESS);
    EXPECT_TRUE(biar_ut::StubCalls().empty());
}

TEST_F(BiarCoreTest, SioAgLoopRound0ReadsSioRankSlice)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 36, 40, 40}), 39);
    tpl.SetupTopology(1, {2}, {3}, MakeChannels(4));
    ASSERT_EQ(tpl.SIOProcessAgLoop(0, 0, 4, 2, kStride), HCCL_SUCCESS);
    const auto& calls = biar_ut::StubCalls();
    ASSERT_EQ(calls.size(), 5u);
    EXPECT_EQ(calls[2].api, StubApi::kRead);
    EXPECT_EQ(calls[2].dst, PtrAt(kOutputBase, 40));
    EXPECT_EQ(calls[2].src, PtrAt(kRemoteBase, 1 * kRemoteStep + 0));
    EXPECT_EQ(calls[2].count, 36u);
}

TEST_F(BiarCoreTest, SioAgLoopLastRoundReadsPairedSlot)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 44}), 41);
    tpl.SetupTopology(1, {2}, {3}, MakeChannels(4));
    ASSERT_EQ(tpl.SIOProcessAgLoop(1, 0, 4, 2, kStride), HCCL_SUCCESS);
    const auto& calls = biar_ut::StubCalls();
    ASSERT_EQ(calls.size(), 5u);
    EXPECT_EQ(calls[2].api, StubApi::kRead);
    EXPECT_EQ(calls[2].dst, PtrAt(kOutputBase, 120));
    EXPECT_EQ(calls[2].src, PtrAt(kRemoteBase, 1 * kRemoteStep + 1 * kStride));
    EXPECT_EQ(calls[2].count, 44u);
}

TEST_F(BiarCoreTest, LocalCopyAgLoopCopiesFromRound1On)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 40}), 40);
    tpl.SetupTopology(1, {2}, {3}, MakeChannels(4));
    ASSERT_EQ(tpl.LocalCopyAgLoop(0, 0, 4, 2, kStride), HCCL_SUCCESS);
    EXPECT_TRUE(biar_ut::StubCalls().empty());

    ASSERT_EQ(tpl.LocalCopyAgLoop(1, 0, 4, 2, kStride), HCCL_SUCCESS);
    const auto& calls = biar_ut::StubCalls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].api, StubApi::kLocalCopy);
    EXPECT_EQ(calls[0].thread, kSub1);
    EXPECT_EQ(calls[0].dst, PtrAt(kOutputBase, 80));
    EXPECT_EQ(calls[0].src, PtrAt(kScratchBase, 1 * kStride));
    EXPECT_EQ(calls[0].count, 40u);
}

// ---------------- Preprocess ----------------

TEST_F(BiarCoreTest, PreprocessRank1FallsBackToLocalCopy)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40}), 10);
    std::vector<ChannelInfo> channels;
    ASSERT_EQ(tpl.Preprocess(0, 1, channels), HCCL_SUCCESS);
    const auto& calls = biar_ut::StubCalls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].api, StubApi::kLocalCopy);
    EXPECT_EQ(calls[0].dst, kOutputBase);
    EXPECT_EQ(calls[0].src, kInputBase);
    EXPECT_EQ(calls[0].count, kMemSize);
}

TEST_F(BiarCoreTest, PreprocessRejectsInsufficientChannels)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 40}), 40);
    std::vector<ChannelInfo> channels = MakeChannels(2);
    EXPECT_EQ(tpl.Preprocess(0, 4, channels), HCCL_E_INTERNAL);
    EXPECT_TRUE(biar_ut::StubCalls().empty());
}

TEST_F(BiarCoreTest, PreprocessSetsUnitSizeFromDataType)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({40, 40, 40, 40}), 40);
    std::vector<ChannelInfo> channels = MakeChannels(4);
    ASSERT_EQ(tpl.Preprocess(0, 4, channels), HCCL_SUCCESS);
    EXPECT_EQ(tpl.UnitSize(), kUnit);
}

// ---------------- RunAsync 编排确定性 ----------------

std::vector<StubApi> RecordedApis()
{
    std::vector<StubApi> apis;
    for (const auto& call : biar_ut::StubCalls()) {
        apis.push_back(call.api);
    }
    return apis;
}

TEST_F(BiarCoreTest, RunAsyncRank0Of4ProducesDeterministicSequence)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({100, 100, 100, 100}), 100);
    std::vector<ChannelInfo> channels = MakeChannels(4);
    ASSERT_EQ(tpl.RunAsync(0, 4, channels), HCCL_SUCCESS);

    using P = StubApi;
    const std::vector<P> expect{
        // 预处理后首拷贝: hccs_ranks[0] 切片 input -> scratch
        P::kPreSync,
        P::kLocalCopy,
        P::kPostSync,
        // RS round0: SIO WriteReduce(neighbour) + LocalCopy(own)
        P::kPreSync,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kWriteReduce,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kLocalCopy,
        P::kPostSync,
        // RS round1: HCCS Write + SIO WriteReduce(sio_rank)
        P::kPreSync,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kWrite,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kWriteReduce,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kPostSync,
        // TreeReduce: group0 归约到 home slot, 无末尾拷贝
        P::kPreSync,
        P::kLocalReduce,
        P::kPostSync,
        // AG round0: HCCS Write + SIO Read(sio_rank)
        P::kPreSync,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kWrite,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kRead,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kPostSync,
        // AG round1: SIO Read(paired) + LocalCopyAg
        P::kPreSync,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kRead,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kLocalCopy,
        P::kPostSync,
        // 末尾拷贝: scratch -> output 本 rank 切片
        P::kPreSync,
        P::kLocalCopy,
        P::kPostSync,
    };
    EXPECT_EQ(RecordedApis(), expect);

    const auto& calls = biar_ut::StubCalls();
    const u64 stride = HCCL_MIN_SLICE_ALIGN_910B;
    EXPECT_EQ(calls[1].dst, PtrAt(kScratchBase, 1 * stride));
    EXPECT_EQ(calls[1].src, PtrAt(kInputBase, 200));
    EXPECT_EQ(calls[1].count, 100u);
    const auto& treeReduce = calls[24];
    EXPECT_EQ(treeReduce.api, StubApi::kLocalReduce);
    EXPECT_EQ(treeReduce.dst, PtrAt(kScratchBase, 0));
    EXPECT_EQ(treeReduce.src, PtrAt(kScratchBase, 3 * stride));
    EXPECT_EQ(treeReduce.count, 25u);
    const auto& finalCopy = calls[calls.size() - 2];
    EXPECT_EQ(finalCopy.api, StubApi::kLocalCopy);
    EXPECT_EQ(finalCopy.dst, PtrAt(kOutputBase, 0));
    EXPECT_EQ(finalCopy.src, PtrAt(kScratchBase, 0));
    EXPECT_EQ(finalCopy.count, 100u);
}

TEST_F(BiarCoreTest, RunAsyncZeroSliceSkipsGuardedDataTraffic)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({4, 4, 4, 0}), 3);
    std::vector<ChannelInfo> channels = MakeChannels(4);
    ASSERT_EQ(tpl.RunAsync(0, 4, channels), HCCL_SUCCESS);

    using P = StubApi;
    const std::vector<P> expect{
        P::kPreSync,
        P::kLocalCopy,
        P::kPostSync,
        // RS round0: neighbour 切片为 0, WriteReduce 被跳过
        P::kPreSync,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kLocalCopy,
        P::kPostSync,
        P::kPreSync,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kWrite,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kWriteReduce,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kPostSync,
        P::kPreSync,
        P::kLocalReduce,
        P::kPostSync,
        P::kPreSync,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kWrite,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kRead,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kPostSync,
        // AG round1: paired 切片为 0, Read 被跳过
        P::kPreSync,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kNotifyRecord,
        P::kNotifyWait,
        P::kLocalCopy,
        P::kPostSync,
        P::kPreSync,
        P::kLocalCopy,
        P::kPostSync,
    };
    EXPECT_EQ(RecordedApis(), expect);
}

TEST_F(BiarCoreTest, RunAsyncPropagatesTransferFailure)
{
    BiarTemplateUnderTest tpl;
    tpl.SetupMem(MakeSlices({100, 100, 100, 100}), 100);
    std::vector<ChannelInfo> channels = MakeChannels(4);
    biar_ut::SetStubFailure(StubApi::kWriteReduce, static_cast<int32_t>(HCCL_E_INTERNAL));
    EXPECT_EQ(tpl.RunAsync(0, 4, channels), HCCL_E_INTERNAL);
}
} // namespace
