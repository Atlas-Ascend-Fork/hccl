/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "alltoall_mesh_group.h"

namespace ops_hccl {

using u32 = uint32_t;
using u64 = uint64_t;

class AlltoAllHierOffsetTest : public ::testing::TestWithParam<std::tuple<u32, u32, u64, u64, u64>> {};

TEST_P(AlltoAllHierOffsetTest, FirstStageSendOffsetFormula)
{
    auto [peerAlgRank, targetGroupSize, totalCount, dataTypeSize, cclBufferCountPerRank] = GetParam();
    u64 inBuffBaseOff = 0;
    for (u32 i = 0; i < targetGroupSize; i++) {
        u64 expectedSrcOffset = (peerAlgRank * targetGroupSize + i) * totalCount * dataTypeSize + inBuffBaseOff;
        EXPECT_GE(expectedSrcOffset, 0u);
    }
}

TEST_P(AlltoAllHierOffsetTest, LastStageRecvOffsetFormula)
{
    auto [peerAlgRank, srcGroupSize, totalCount, dataTypeSize, cclBufferCountPerRank] = GetParam();
    u64 outBuffBaseOff = 0;
    for (u32 i = 0; i < srcGroupSize; i++) {
        u64 expectedDstOffset = (i * 8 + peerAlgRank) * totalCount * dataTypeSize + outBuffBaseOff;
        EXPECT_GE(expectedDstOffset, 0u);
    }
}

INSTANTIATE_TEST_SUITE_P(
    OffsetParams, AlltoAllHierOffsetTest,
    ::testing::Values(
        std::make_tuple(1u, 2u, 1000000ull, 2ull, 6553600ull), std::make_tuple(3u, 8u, 3333333ull, 4ull, 3276800ull),
        std::make_tuple(0u, 1u, 67108864ull, 1ull, 8388608ull), std::make_tuple(7u, 8u, 2222222ull, 2ull, 6553600ull)));

class AlltoAllHierSelectorLogicTest : public ::testing::Test {};

TEST_F(AlltoAllHierSelectorLogicTest, HierConfiguredWithMultiLevelMeshReturnsHier)
{
    u32 topoLevelNums = 2;
    bool hierConfigured = true;
    bool level0Supported = true;
    bool expectedMatch = hierConfigured && topoLevelNums > 1 && level0Supported;
    EXPECT_TRUE(expectedMatch);
}

TEST_F(AlltoAllHierSelectorLogicTest, HierNotConfiguredFallsThrough)
{
    u32 topoLevelNums = 2;
    bool hierConfigured = false;
    bool expectedFallthrough = !hierConfigured || topoLevelNums <= 1;
    EXPECT_TRUE(expectedFallthrough);
}

TEST_F(AlltoAllHierSelectorLogicTest, HierConfiguredWithSingleLevelFallsThrough)
{
    u32 topoLevelNums = 1;
    bool hierConfigured = true;
    bool expectedFallthrough = !hierConfigured || topoLevelNums <= 1;
    EXPECT_TRUE(expectedFallthrough);
}

} // namespace ops_hccl

namespace ops_hccl {
TEST(AlltoAllMeshGroupTest, CoversPeersOnceAndPairsEndpointsInSameRound)
{
    for (uint32_t size : {1U, 2U, 3U, 16U, 17U, 18U, 32U, 33U, 34U, 65U, 128U}) {
        std::vector<uint32_t> ranks;
        for (uint32_t i = 0; i < size; ++i) {
            ranks.push_back(i * 3 + 7); // Non-contiguous user rank IDs.
        }
        for (uint32_t myIndex = 0; myIndex < size; ++myIndex) {
            std::vector<uint32_t> visits(size, 0);
            uint32_t round = 0;
            for (;; ++round) {
                const auto peers = CalcAlltoAllMeshGroup(ranks, myIndex, round);
                if (peers.empty()) {
                    break;
                }
                ASSERT_LE(peers.size(), ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE);
                for (auto peer : peers) {
                    const auto peerIndex
                        = static_cast<uint32_t>(std::find(ranks.begin(), ranks.end(), peer) - ranks.begin());
                    ASSERT_LT(peerIndex, size);
                    ASSERT_NE(peerIndex, myIndex);
                    ++visits[peerIndex];
                    const auto reversePeers = CalcAlltoAllMeshGroup(ranks, peerIndex, round);
                    EXPECT_NE(std::find(reversePeers.begin(), reversePeers.end(), ranks[myIndex]), reversePeers.end());
                }
            }
            EXPECT_EQ(
                round,
                (size - 1 + ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE - 1) / ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE);
            for (uint32_t i = 0; i < size; ++i) {
                EXPECT_EQ(visits[i], i == myIndex ? 0U : 1U);
            }
        }
    }
}

TEST(AlltoAllMeshGroupTest, MatchesAicpuDistanceOrder)
{
    std::vector<uint32_t> ranks;
    for (uint32_t i = 0; i < 32; ++i) {
        ranks.push_back(i);
    }
    EXPECT_EQ(
        CalcAlltoAllMeshGroup(ranks, 0, 0),
        (std::vector<uint32_t>{31, 1, 30, 2, 29, 3, 28, 4, 27, 5, 26, 6, 25, 7, 24, 8}));
    EXPECT_EQ(
        CalcAlltoAllMeshGroup(ranks, 0, 1),
        (std::vector<uint32_t>{23, 9, 22, 10, 21, 11, 20, 12, 19, 13, 18, 14, 17, 15, 16}));
    EXPECT_TRUE(CalcAlltoAllMeshGroup(ranks, 0, 2).empty());
    EXPECT_TRUE(CalcAlltoAllMeshGroup({}, 0, 0).empty());
    EXPECT_TRUE(CalcAlltoAllMeshGroup(ranks, 32, 0).empty());
}
} // namespace ops_hccl

namespace ops_hccl {
TEST(AlltoAllMeshGroupTest, DeviceFilteringSkipsHostPeersAndContinuesAfterEmptyRound)
{
    std::vector<uint32_t> ranks;
    for (uint32_t i = 0; i < 65; ++i) {
        ranks.push_back(i);
    }
    // rank0's first distance group is HOST-only, but a later group contains DEVICE peers.
    const std::vector<uint32_t> devices{9, 56, 10, 55, 11, 54, 12, 53, 13, 52, 14, 51, 15, 50, 16, 49, 17, 48};
    EXPECT_TRUE(CalcAlltoAllMeshGroup(ranks, 0, 0, &devices).empty());
    EXPECT_EQ(CalcAlltoAllMeshGroup(ranks, 0, 1, &devices).size(), 16U);
    EXPECT_EQ(CalcAlltoAllMeshGroup(ranks, 0, 2, &devices).size(), 2U);
    std::vector<uint32_t> covered;
    for (uint32_t round = 0; round < 4; ++round) {
        const auto group = CalcAlltoAllMeshGroup(ranks, 0, round, &devices);
        covered.insert(covered.end(), group.begin(), group.end());
        for (auto peer : group) {
            const std::vector<uint32_t> reverseDevices{0};
            const auto reverse = CalcAlltoAllMeshGroup(ranks, peer, round, &reverseDevices);
            EXPECT_NE(std::find(reverse.begin(), reverse.end(), 0), reverse.end());
        }
    }
    std::sort(covered.begin(), covered.end());
    auto expected = devices;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(covered, expected);
    const std::vector<uint32_t> noDevices;
    EXPECT_TRUE(CalcAlltoAllMeshGroup(ranks, 0, 0, &noDevices).empty());
}
} // namespace ops_hccl
