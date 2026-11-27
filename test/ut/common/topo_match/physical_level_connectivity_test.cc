/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the license.
 */

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "alg_attrs.h"
#include "hccl_rank_graph_dl.h"
#include "physical_level.h"
#include "topo_match_test_common.h"

namespace {
using ops_hccl::BuildPhysicalLevels;
using ops_hccl::TopoInfoWithNetLayerDetails;
using ops_hccl::ut_helper::Range;
using ops_hccl::ut_helper::RangeFrom;

using RankGroups = std::vector<std::vector<u32>>;

TopoInfoWithNetLayerDetails MakeTopology(u32 rank, u32 rankSize, const RankGroups& localRanks, RankGroups sizes)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = rank;
    topo.userRankSize = rankSize;
    auto& details = topo.netLayerDetails;
    details.netLayerNum = static_cast<u32>(sizes.size());
    details.netLayers = Range(details.netLayerNum);
    for (const auto& layer : localRanks) {
        details.localNetInsSizeOfLayer.push_back(static_cast<u32>(layer.size()));
    }
    for (const auto& layer : sizes) {
        details.netInstNumOfLayer.push_back(static_cast<u32>(layer.size()));
    }
    details.instSizeListOfLayer = std::move(sizes);
    return topo;
}
} // namespace

extern "C" HcclResult HcclRankGraphGetRanksByLayer(HcclComm comm, u32 netLayer, u32** ranks, u32* rankNum)
{
    auto& group = static_cast<RankGroups*>(comm)->at(netLayer);
    *ranks = group.data();
    *rankNum = static_cast<u32>(group.size());
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclRankGraphGetTopoInstsByLayer(HcclComm, u32, u32** topoInsts, u32* topoInstNum)
{
    *topoInsts = nullptr;
    *topoInstNum = 0;
    return HCCL_SUCCESS;
}

TEST(PhysicalLevelConnectivityTest, RejectsSingletonRailOnBothPods)
{
    std::vector<u32> railSizes(15, 2);
    railSizes.push_back(1);
    const RankGroups sizes{{7, 8, 8, 8}, {15, 16}, railSizes};
    RankGroups singletonRankGroups{RangeFrom(8, 8), Range(16), {15}};
    RankGroups peerRankGroups{{16, 17, 18, 19, 20, 21, 22, 30}, RangeFrom(16, 15), {0, 30}};
    auto singletonTopo = MakeTopology(15, 31, singletonRankGroups, sizes);
    auto peerTopo = MakeTopology(30, 31, peerRankGroups, sizes);

    EXPECT_EQ(BuildPhysicalLevels(&singletonRankGroups, &singletonTopo), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(BuildPhysicalLevels(&peerRankGroups, &peerTopo), HCCL_E_NOT_SUPPORT);
    EXPECT_TRUE(singletonTopo.physicalLevels.empty());
    EXPECT_TRUE(peerTopo.physicalLevels.empty());
}

TEST(PhysicalLevelConnectivityTest, AllowsSingletonServerGroupWithRegularRails)
{
    RankGroups localRanks{{0}, {0, 1, 2}, {0, 3}};
    auto topo = MakeTopology(0, 6, localRanks, {{1, 2, 1, 2}, {3, 3}, {2, 2, 2}});

    EXPECT_EQ(BuildPhysicalLevels(&localRanks, &topo), HCCL_SUCCESS);
    EXPECT_EQ(topo.physicalLevels.size(), 3U);
}
