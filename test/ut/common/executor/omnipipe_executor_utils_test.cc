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
#include "executor_common_ops.h"
#include "omnipipe_executor_utils.h"

using namespace ops_hccl;

namespace {
TopoInfoWithNetLayerDetails MakeTopoForNetLayerPath(u32 serverNum, u32 serverNumPerSuperPod)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.serverNum = serverNum;
    topo.userRankSize = serverNum;
    topo.serverNumPerSuperPod = serverNumPerSuperPod;
    topo.superPodNum = 0;
    return topo;
}
} // namespace

class IsOnlyThirdAxisTopoTest : public ::testing::Test {};

// serverNum == 0（未初始化/空拓扑）直接判 false
TEST_F(IsOnlyThirdAxisTopoTest, ServerNumZero_ReturnsFalse)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.serverNum = 0;
    topo.userRankSize = 0;
    EXPECT_FALSE(IsOnlyThirdAxisTopo(&topo));
}

// serverNum != userRankSize（每 server 多卡，L0 != 1）→ false
TEST_F(IsOnlyThirdAxisTopoTest, ServerNumNotEqualUserRankSize_ReturnsFalse)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.serverNum = 2;
    topo.userRankSize = 4;         // 每 server 2 卡
    topo.serverNumPerSuperPod = 1; // 即便 L1 == 1 也不成立，先被 L0 拦下
    EXPECT_FALSE(IsOnlyThirdAxisTopo(&topo));
}

// 每 server 独占一个 L1 实例 → 只剩第三轴链路 → true
TEST_F(IsOnlyThirdAxisTopoTest, ServerNumPerSuperPodOne_ReturnsTrue)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.serverNum = 4;
    topo.userRankSize = 4;
    topo.serverNumPerSuperPod = 1;
    EXPECT_TRUE(IsOnlyThirdAxisTopo(&topo));
}

// layers[1] 指向的 L1 实例大小为 1（每 server 独占一个 L1）→ true
TEST_F(IsOnlyThirdAxisTopoTest, SecondLayerSizeOne_ReturnsTrue)
{
    auto topo = MakeTopoForNetLayerPath(4, 2);
    topo.netLayerDetails.netLayers = {0, 1};
    topo.netLayerDetails.localNetInsSizeOfLayer = {4, 1}; // L1 实例大小 == 1
    EXPECT_TRUE(IsOnlyThirdAxisTopo(&topo));
}

// netLayers 的取值不是 0/1 连续编号时，仍按 layers[1] 作为下标取 L1 大小
TEST_F(IsOnlyThirdAxisTopoTest, NonContiguousLayerIds_UseSecondLayerAsIndex)
{
    auto topo = MakeTopoForNetLayerPath(4, 2);
    topo.netLayerDetails.netLayers = {2, 5};                          // layers[1] == 5
    topo.netLayerDetails.localNetInsSizeOfLayer = {4, 4, 4, 4, 4, 1}; // index 5 == 1
    EXPECT_TRUE(IsOnlyThirdAxisTopo(&topo));
}

TEST(ConcurrentPortNumTest, ProductionParametersForTwoAndFourRanks)
{
    struct TestCase {
        const char* name;
        u32 rankSize;
        u32 ratio;
        u32 tune;
        u32 expectedMeshPort;
    };
    const TestCase testCases[] = {
        {"AllGather symmetric 2 ranks", 2, 1, 2, 20},     {"AllGather asymmetric 2 ranks", 2, 1, 2, 20},
        {"AllGather symmetric 4 ranks", 4, 1, 4, 60},     {"AllGather asymmetric 4 ranks", 4, 1, 2, 62},
        {"AllReduce symmetric 2 ranks", 2, 1, 2, 20},     {"AllReduce asymmetric 2 ranks", 2, 10, 2, 20},
        {"AllReduce symmetric 4 ranks", 4, 1, 4, 60},     {"AllReduce asymmetric 4 ranks", 4, 10, 2, 80},
        {"AllToAll symmetric 2 ranks", 2, 1, 1, 21},      {"AllToAll asymmetric 2 ranks", 2, 4, 1, 24},
        {"AllToAll symmetric 4 ranks", 4, 1, 3, 61},      {"AllToAll asymmetric 4 ranks", 4, 4, 3, 64},
        {"ReduceScatter symmetric 2 ranks", 2, 1, 2, 20}, {"ReduceScatter asymmetric 2 ranks", 2, 0, 2, 20},
        {"ReduceScatter symmetric 4 ranks", 4, 1, 4, 60}, {"ReduceScatter asymmetric 4 ranks", 4, 0, 4, 60},
    };

    for (const auto& testCase : testCases) {
        SCOPED_TRACE(testCase.name);
        u32 meshPort = 0;
        u32 closPort = 0;
        GetConcurrentPortNum(testCase.rankSize, testCase.ratio, testCase.tune, meshPort, closPort);
        EXPECT_EQ(meshPort, testCase.expectedMeshPort);
        EXPECT_EQ(closPort, 80U);
    }
}

TEST(ConcurrentPortNumTest, AppliesRatioAndTuneWithoutChoosingPolicy)
{
    u32 meshPort = 0;
    u32 closPort = 0;
    GetConcurrentPortNum(8, 1, 4, meshPort, closPort);
    EXPECT_EQ(meshPort, 144U);
    EXPECT_EQ(closPort, 80U);
    GetConcurrentPortNum(8, 4, 4, meshPort, closPort);
    EXPECT_EQ(meshPort, 156U);
    EXPECT_EQ(closPort, 80U);
    GetConcurrentPortNum(8, 4, 0, meshPort, closPort);
    EXPECT_EQ(meshPort, 172U);
    EXPECT_EQ(closPort, 80U);
}
