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
#include "alltoall_dpu_topology.h"
#include "alltoall_mesh_group.h"

namespace ops_hccl {
namespace {
    PhysicalLevelInfo MakeLevel(std::vector<u32> ranks, u32 netLayer, EndpointLocType locType)
    {
        PhysicalLevelInfo level;
        level.localRanks = std::move(ranks);
        level.ref.netLayer = netLayer;
        level.locType = locType;
        level.hasTopoInst = true;
        return level;
    }
} // namespace

TEST(DpuAlltoAllLocalPodTest, SelectsPodRatherThanServerOrGlobalHostScope)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = 3;
    topo.physicalLevels
        = {MakeLevel({0, 3}, 0, ENDPOINT_LOC_TYPE_DEVICE),
           MakeLevel({0, 3, 7, 12}, POD_NET_LAYER, ENDPOINT_LOC_TYPE_DEVICE),
           MakeLevel({0, 3, 7, 12, 20, 21, 30, 40}, RAIL_NET_LAYER, ENDPOINT_LOC_TYPE_HOST)};
    std::vector<u32> pod;
    ASSERT_EQ(GetAlltoAllLocalPodRanks(topo, pod), HCCL_SUCCESS);
    EXPECT_EQ(pod, (std::vector<u32>{0, 3, 7, 12}));
    EXPECT_EQ(CalcAlltoAllMeshGroup(pod, 1, 0), (std::vector<u32>{0, 7, 12}));
    // The same membership is used by the peer regardless of a different server-level scope.
    topo.userRank = 12;
    topo.physicalLevels[0].localRanks = {7, 12};
    std::vector<u32> otherPod;
    ASSERT_EQ(GetAlltoAllLocalPodRanks(topo, otherPod), HCCL_SUCCESS);
    EXPECT_EQ(otherPod, pod);
}

TEST(DpuAlltoAllLocalPodTest, PodsHaveIndependentMembershipAndRoundCounts)
{
    for (u32 size : {1U, 8U, 17U, 18U, 33U}) {
        TopoInfoWithNetLayerDetails topo{};
        std::vector<u32> members;
        for (u32 i = 0; i < size; ++i) {
            members.push_back(i * 3 + 100);
        }
        topo.userRank = members[0];
        topo.physicalLevels = {MakeLevel(members, POD_NET_LAYER, ENDPOINT_LOC_TYPE_DEVICE)};
        std::vector<u32> pod;
        ASSERT_EQ(GetAlltoAllLocalPodRanks(topo, pod), HCCL_SUCCESS);
        std::vector<u32> covered;
        const u32 concurrentSize = ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE;
        const u32 rounds = (size - 1 + concurrentSize - 1) / concurrentSize;
        for (u32 round = 0; round < rounds; ++round) {
            const auto peers = CalcAlltoAllMeshGroup(pod, 0, round);
            EXPECT_LE(peers.size(), concurrentSize);
            covered.insert(covered.end(), peers.begin(), peers.end());
        }
        std::sort(covered.begin(), covered.end());
        EXPECT_EQ(covered, (std::vector<u32>(members.begin() + 1, members.end())));
        EXPECT_TRUE(CalcAlltoAllMeshGroup(pod, 0, rounds).empty());
    }
}

TEST(DpuAlltoAllLocalPodTest, IgnoresRailDeviceScopeAndHostScopeAtPodLayer)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = 0;
    topo.physicalLevels = {
        MakeLevel({0, 1}, 0, ENDPOINT_LOC_TYPE_DEVICE), MakeLevel({0, 1, 2, 3}, POD_NET_LAYER, ENDPOINT_LOC_TYPE_HOST),
        MakeLevel({0, 4, 8, 12}, RAIL_NET_LAYER, ENDPOINT_LOC_TYPE_DEVICE)};
    std::vector<u32> pod;
    ASSERT_EQ(GetAlltoAllLocalPodRanks(topo, pod), HCCL_SUCCESS);
    EXPECT_EQ(pod, (std::vector<u32>{0, 1}));
}

TEST(DpuAlltoAllLocalPodTest, HostOnlyUsesSingletonAndRejectsInvalidDeviceMembership)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = 0;
    topo.physicalLevels = {MakeLevel({0, 1, 2, 3}, POD_NET_LAYER, ENDPOINT_LOC_TYPE_HOST)};
    std::vector<u32> pod;
    ASSERT_EQ(GetAlltoAllLocalPodRanks(topo, pod), HCCL_SUCCESS);
    EXPECT_EQ(pod, (std::vector<u32>{0}));
    topo.physicalLevels = {MakeLevel({1, 2}, 0, ENDPOINT_LOC_TYPE_DEVICE)};
    EXPECT_EQ(GetAlltoAllLocalPodRanks(topo, pod), HCCL_E_PARA);
    topo.physicalLevels
        = {MakeLevel({0, 1}, 0, ENDPOINT_LOC_TYPE_DEVICE), MakeLevel({0, 2}, POD_NET_LAYER, ENDPOINT_LOC_TYPE_DEVICE)};
    EXPECT_EQ(GetAlltoAllLocalPodRanks(topo, pod), HCCL_E_NOT_SUPPORT);
}
} // namespace ops_hccl
