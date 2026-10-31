/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "reduce_scatter_operand_order.h"
#include "reduce_scatter_selector_utils.h"

using namespace ops_hccl;

namespace {

uint8_t inputBuffer[0x100];
uint8_t outputBuffer[0x100];

OpParam MakeOpParam()
{
    OpParam op{};
    op.opExecuteConfig = OpExecuteConfig::CCU_MS;
    op.reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
    op.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    op.inputPtr = inputBuffer;
    op.inputSize = sizeof(inputBuffer);
    op.outputPtr = outputBuffer;
    op.outputSize = sizeof(outputBuffer);
    return op;
}

TopoInfoWithNetLayerDetails MakeSingleDieMesh(u32 rankSize = 8)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRankSize = rankSize;
    topo.topoLevelNums = 1;
    topo.level0Topo = Level0Shape::MESH_1D;
    topo.level0MeshType = Level0MeshType::SINGLE_DIE;
    return topo;
}

} // namespace

TEST(ReduceScatterSelector, StrictCcuMsSelectsBatchInvariantSingleDieMesh)
{
    OpParam op = MakeOpParam();
    TopoInfoWithNetLayerDetails topo = MakeSingleDieMesh();

    EXPECT_TRUE(CanSelectBatchInvariantCcuMsReduceScatter(op, &topo));
}

TEST(ReduceScatterSelector, StrictCcuMsRejectsInPlace)
{
    OpParam op = MakeOpParam();
    op.outputPtr = op.inputPtr;
    TopoInfoWithNetLayerDetails topo = MakeSingleDieMesh();

    EXPECT_FALSE(CanSelectBatchInvariantCcuMsReduceScatter(op, &topo));
}

TEST(ReduceScatterSelector, StrictCcuMsRejectsTwoDieMesh)
{
    OpParam op = MakeOpParam();
    TopoInfoWithNetLayerDetails topo = MakeSingleDieMesh();
    topo.level0MeshType = Level0MeshType::TWO_DIE_REGULAR;

    EXPECT_FALSE(CanSelectBatchInvariantCcuMsReduceScatter(op, &topo));
}

TEST(ReduceScatterSelector, StrictCcuMsRejectsRankSizeAboveCcuLimit)
{
    OpParam op = MakeOpParam();
    TopoInfoWithNetLayerDetails topo = MakeSingleDieMesh(9);

    EXPECT_FALSE(CanSelectBatchInvariantCcuMsReduceScatter(op, &topo));
}

TEST(ReduceScatterSelector, StrictCcuMsRejectsIncompleteMeshClos)
{
    OpParam op = MakeOpParam();
    TopoInfoWithNetLayerDetails topo = MakeSingleDieMesh();
    topo.level0Topo = Level0Shape::MESH_1D_CLOS;
    topo.level0MeshType = Level0MeshType::NOT_MESH;
    topo.netLayerDetails.localNetInsSizeOfLayer = {8};
    topo.topoInstDetailsOfLayer.resize(1);
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[CommTopo::COMM_TOPO_1DMESH] = {4};

    EXPECT_FALSE(CanSelectBatchInvariantCcuMsReduceScatter(op, &topo));
}

TEST(ReduceScatterSelector, StrictCcuMsRejectsIncompleteMeshClosWithoutLocalInsSize)
{
    OpParam op = MakeOpParam();
    TopoInfoWithNetLayerDetails topo = MakeSingleDieMesh();
    topo.level0Topo = Level0Shape::MESH_1D_CLOS;
    topo.level0MeshType = Level0MeshType::NOT_MESH;
    topo.topoInstDetailsOfLayer.resize(1);
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[CommTopo::COMM_TOPO_1DMESH] = {8};

    EXPECT_FALSE(CanSelectBatchInvariantCcuMsReduceScatter(op, &topo));
}

TEST(ReduceScatterSelector, StrictCcuMsRejectsIncompleteMeshClosWithLayer0SizeMismatch)
{
    OpParam op = MakeOpParam();
    TopoInfoWithNetLayerDetails topo = MakeSingleDieMesh();
    topo.level0Topo = Level0Shape::MESH_1D_CLOS;
    topo.level0MeshType = Level0MeshType::NOT_MESH;
    topo.netLayerDetails.localNetInsSizeOfLayer = {4};
    topo.topoInstDetailsOfLayer.resize(1);
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[CommTopo::COMM_TOPO_1DMESH] = {8};

    EXPECT_FALSE(CanSelectBatchInvariantCcuMsReduceScatter(op, &topo));
}

TEST(ReduceScatterSelector, StrictCcuMsRejectsIncompleteMeshClosWithoutTopoInstDetails)
{
    OpParam op = MakeOpParam();
    TopoInfoWithNetLayerDetails topo = MakeSingleDieMesh();
    topo.level0Topo = Level0Shape::MESH_1D_CLOS;
    topo.level0MeshType = Level0MeshType::NOT_MESH;
    topo.netLayerDetails.localNetInsSizeOfLayer = {8};

    EXPECT_FALSE(CanSelectBatchInvariantCcuMsReduceScatter(op, &topo));
}

TEST(ReduceScatterSelector, StrictCcuMsSelectsFullMeshClos)
{
    OpParam op = MakeOpParam();
    TopoInfoWithNetLayerDetails topo = MakeSingleDieMesh();
    topo.level0Topo = Level0Shape::MESH_1D_CLOS;
    topo.level0MeshType = Level0MeshType::NOT_MESH;
    topo.netLayerDetails.localNetInsSizeOfLayer = {8};
    topo.topoInstDetailsOfLayer.resize(1);
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[CommTopo::COMM_TOPO_1DMESH] = {8};

    EXPECT_TRUE(CanSelectBatchInvariantCcuMsReduceScatter(op, &topo));
}

TEST(ReduceScatterSelector, StrictCcuMsRejectsUnsupportedOperation)
{
    OpParam op = MakeOpParam();
    TopoInfoWithNetLayerDetails topo = MakeSingleDieMesh();

    op.reduceType = HcclReduceOp::HCCL_REDUCE_PROD;
    EXPECT_FALSE(CanSelectBatchInvariantCcuMsReduceScatter(op, &topo));

    op.reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
    op.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP64;
    EXPECT_FALSE(CanSelectBatchInvariantCcuMsReduceScatter(op, &topo));
}

TEST(ReduceScatterSelector, StrictCcuMsRejectsNonCcuEngine)
{
    OpParam op = MakeOpParam();
    op.opExecuteConfig = OpExecuteConfig::AICPU_TS;
    TopoInfoWithNetLayerDetails topo = MakeSingleDieMesh();

    EXPECT_FALSE(CanSelectBatchInvariantCcuMsReduceScatter(op, &topo));
}

TEST(ReduceScatterOperandOrder, StrictPlanBuildsAscendingRemoteRanksAndRankSlot)
{
    for (uint32_t rankSize = 2; rankSize <= 8; ++rankSize) {
        for (uint32_t rankId = 0; rankId < rankSize; ++rankId) {
            const ReduceScatterOperandPlan plan = BuildReduceScatterOperandPlan(rankSize, rankId, true);

            ASSERT_EQ(plan.localRankSlot, rankId);
            ASSERT_EQ(plan.remoteRanks.size(), rankSize - 1);
            for (uint32_t i = 0; i < plan.remoteRanks.size(); ++i) {
                const uint32_t expectedRank = (i < rankId) ? i : i + 1;
                EXPECT_EQ(plan.remoteRanks[i], expectedRank);
            }
        }
    }
}

TEST(ReduceScatterOperandOrder, NonStrictPlanKeepsLegacyLocalSlot)
{
    const ReduceScatterOperandPlan plan = BuildReduceScatterOperandPlan(8, 3, false);

    EXPECT_EQ(plan.localRankSlot, 7);
    EXPECT_EQ(plan.remoteRanks, (std::vector<uint32_t>{0, 1, 2, 4, 5, 6, 7}));
}
