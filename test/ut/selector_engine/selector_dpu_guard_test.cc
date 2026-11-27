/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// IsDataTypeOrReduceTypeSpecial 的 6 处选路守护:
//   1. all_reduce_auto_selector.cc     SelectDPUAlgo -> DpuAllReducePipeLineMeshNHRNHR
//   2. all_reduce_auto_selector.cc     SelectNdaAlgo -> DpuAllReducePipeLineMeshNHRNHRNDA
//   3. reduce_auto_selector.cc         SelectDPUAlgo -> DpuReducePipeLineMeshNHRNHR
//   4. reduce_auto_selector.cc         SelectNdaAlgo -> DpuReducePipeLineMeshNHRNHRNDA
//   5. reduce_scatter_auto_selector.cc SelectDPUAlgo -> DpuReduceScatterPipeLineMeshNHRMesh
//   6. reduce_scatter_auto_selector.cc SelectNdaAlgo -> DpuReduceScatterPipeLineMeshNHRMeshNDA
// 4 种特殊入参(INT64/UINT64/FP64/PROD) × 3 个算子都必须挡住 pipeline 算法;
// 每个守护点另配一条非特殊入参对照, 证明拓扑与数据量前置条件成立(pipeline 可达)。

#include <string>

#include <gtest/gtest.h>

#include "all_reduce_auto_selector.h"
#include "auto_selector_base.h"
#include "reduce_auto_selector.h"
#include "reduce_scatter_auto_selector.h"
#include "selector_deps_stub.h"

namespace {
using ops_hccl::AllReduceAutoSelector;
using ops_hccl::AutoSelectorBase;
using ops_hccl::Level0Shape;
using ops_hccl::OpParam;
using ops_hccl::ReduceAutoSelector;
using ops_hccl::ReduceScatterAutoSelector;
using ops_hccl::SelectorStatus;
using ops_hccl::TopoInfoWithNetLayerDetails;

// DPU pipeline 分支的拓扑前置条件: 两级 MESH_1D_CLOS 且非 PCIE 混合
TopoInfoWithNetLayerDetails MakeTwoLevelMeshClosNoPcieTopo()
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRankSize = 16;
    topo.topoLevelNums = 2;
    topo.deviceNumPerModule = 8;
    topo.level0Topo = Level0Shape::MESH_1D_CLOS;
    topo.level0PcieMix = false;
    topo.netLayerDetails.localNetInsSizeOfLayer = {8, 2};
    return topo;
}

// 特殊入参: 64bit 数据类型或 PROD 归约; 4M 元素越过 NDA pipeline 的 4MB 数据量门槛
OpParam MakeSpecialInputParam(HcclCMDType opType, HcclDataType dataType, HcclReduceOp reduceType)
{
    OpParam param{};
    param.opType = opType;
    param.DataDes.count = 4 * 1024 * 1024;
    param.DataDes.dataType = dataType;
    param.reduceType = reduceType;
    return param;
}

// 非特殊入参: FP16 + SUM, 作为 pipeline 分支可达的正向对照
OpParam MakeNormalInputParam(HcclCMDType opType)
{
    return MakeSpecialInputParam(opType, HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_SUM);
}
} // namespace

// ---------- 守护点 1: AllReduce SelectDPUAlgo ----------
TEST(AllReduceDpuPipelineGuardTest, SpecialInputsFallBackToSequenceAlgo)
{
    // given
    TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshClosNoPcieTopo();
    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    OpParam int64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam uint64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_UINT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam fp64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam prodParam = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_PROD);
    AllReduceAutoSelector selector;
    AutoSelectorBase& selectorBase = selector;

    // when
    std::string int64Alg;
    SelectorStatus int64Ret = selectorBase.SelectDPUAlgo(&topo, int64Param, {}, int64Alg);
    std::string uint64Alg;
    SelectorStatus uint64Ret = selectorBase.SelectDPUAlgo(&topo, uint64Param, {}, uint64Alg);
    std::string fp64Alg;
    SelectorStatus fp64Ret = selectorBase.SelectDPUAlgo(&topo, fp64Param, {}, fp64Alg);
    std::string prodAlg;
    SelectorStatus prodRet = selectorBase.SelectDPUAlgo(&topo, prodParam, {}, prodAlg);

    // then
    EXPECT_EQ(int64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(int64Alg, "DpuAllReduceSequenceMeshMesh");
    EXPECT_EQ(uint64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(uint64Alg, "DpuAllReduceSequenceMeshMesh");
    EXPECT_EQ(fp64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(fp64Alg, "DpuAllReduceSequenceMeshMesh");
    EXPECT_EQ(prodRet, SelectorStatus::MATCH);
    EXPECT_EQ(prodAlg, "DpuAllReduceSequenceMeshMesh");
}

TEST(AllReduceDpuPipelineGuardTest, NonSpecialInputSelectsPipelineAlgo)
{
    // given
    TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshClosNoPcieTopo();
    OpParam param = MakeNormalInputParam(HcclCMDType::HCCL_CMD_ALLREDUCE);
    AllReduceAutoSelector selector;
    AutoSelectorBase& selectorBase = selector;

    // when
    std::string algName;
    SelectorStatus ret = selectorBase.SelectDPUAlgo(&topo, param, {}, algName);

    // then
    EXPECT_EQ(ret, SelectorStatus::MATCH);
    EXPECT_EQ(algName, "DpuAllReducePipeLineMeshNHRNHR");
}

// ---------- 守护点 2: AllReduce SelectNdaAlgo ----------
TEST(AllReduceNdaPipelineGuardTest, SpecialInputsFallBackToSequenceAlgo)
{
    // given
    TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshClosNoPcieTopo();
    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    OpParam int64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam uint64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_UINT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam fp64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam prodParam = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_PROD);
    AllReduceAutoSelector selector;
    AutoSelectorBase& selectorBase = selector;

    // when
    std::string int64Alg;
    SelectorStatus int64Ret = selectorBase.SelectNdaAlgo(&topo, int64Param, {}, int64Alg);
    std::string uint64Alg;
    SelectorStatus uint64Ret = selectorBase.SelectNdaAlgo(&topo, uint64Param, {}, uint64Alg);
    std::string fp64Alg;
    SelectorStatus fp64Ret = selectorBase.SelectNdaAlgo(&topo, fp64Param, {}, fp64Alg);
    std::string prodAlg;
    SelectorStatus prodRet = selectorBase.SelectNdaAlgo(&topo, prodParam, {}, prodAlg);

    // then
    EXPECT_EQ(int64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(int64Alg, "DpuAllReduceSequenceMeshNHRNDA");
    EXPECT_EQ(uint64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(uint64Alg, "DpuAllReduceSequenceMeshNHRNDA");
    EXPECT_EQ(fp64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(fp64Alg, "DpuAllReduceSequenceMeshNHRNDA");
    EXPECT_EQ(prodRet, SelectorStatus::MATCH);
    EXPECT_EQ(prodAlg, "DpuAllReduceSequenceMeshNHRNDA");
}

TEST(AllReduceNdaPipelineGuardTest, NonSpecialInputSelectsPipelineAlgo)
{
    // given
    TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshClosNoPcieTopo();
    OpParam param = MakeNormalInputParam(HcclCMDType::HCCL_CMD_ALLREDUCE);
    AllReduceAutoSelector selector;
    AutoSelectorBase& selectorBase = selector;

    // when
    std::string algName;
    SelectorStatus ret = selectorBase.SelectNdaAlgo(&topo, param, {}, algName);

    // then
    EXPECT_EQ(ret, SelectorStatus::MATCH);
    EXPECT_EQ(algName, "DpuAllReducePipeLineMeshNHRNHRNDA");
}

// ---------- 守护点 3: Reduce SelectDPUAlgo ----------
TEST(ReduceDpuPipelineGuardTest, SpecialInputsFallBackToSequenceAlgo)
{
    // given
    TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshClosNoPcieTopo();
    HcclCMDType opType = HcclCMDType::HCCL_CMD_REDUCE;
    OpParam int64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam uint64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_UINT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam fp64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam prodParam = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_PROD);
    ReduceAutoSelector selector;
    AutoSelectorBase& selectorBase = selector;

    // when
    std::string int64Alg;
    SelectorStatus int64Ret = selectorBase.SelectDPUAlgo(&topo, int64Param, {}, int64Alg);
    std::string uint64Alg;
    SelectorStatus uint64Ret = selectorBase.SelectDPUAlgo(&topo, uint64Param, {}, uint64Alg);
    std::string fp64Alg;
    SelectorStatus fp64Ret = selectorBase.SelectDPUAlgo(&topo, fp64Param, {}, fp64Alg);
    std::string prodAlg;
    SelectorStatus prodRet = selectorBase.SelectDPUAlgo(&topo, prodParam, {}, prodAlg);

    // then
    EXPECT_EQ(int64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(int64Alg, "DpuReduceSequenceMeshMesh");
    EXPECT_EQ(uint64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(uint64Alg, "DpuReduceSequenceMeshMesh");
    EXPECT_EQ(fp64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(fp64Alg, "DpuReduceSequenceMeshMesh");
    EXPECT_EQ(prodRet, SelectorStatus::MATCH);
    EXPECT_EQ(prodAlg, "DpuReduceSequenceMeshMesh");
}

TEST(ReduceDpuPipelineGuardTest, NonSpecialInputSelectsPipelineAlgo)
{
    // given
    TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshClosNoPcieTopo();
    OpParam param = MakeNormalInputParam(HcclCMDType::HCCL_CMD_REDUCE);
    ReduceAutoSelector selector;
    AutoSelectorBase& selectorBase = selector;

    // when
    std::string algName;
    SelectorStatus ret = selectorBase.SelectDPUAlgo(&topo, param, {}, algName);

    // then
    EXPECT_EQ(ret, SelectorStatus::MATCH);
    EXPECT_EQ(algName, "DpuReducePipeLineMeshNHRNHR");
}

// ---------- 守护点 4: ReduceScatter SelectDPUAlgo ----------
TEST(ReduceScatterDpuPipelineGuardTest, SpecialInputsFallBackToSequenceAlgo)
{
    // given
    TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshClosNoPcieTopo();
    HcclCMDType opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    OpParam int64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam uint64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_UINT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam fp64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam prodParam = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_PROD);
    ReduceScatterAutoSelector selector;
    AutoSelectorBase& selectorBase = selector;

    // when
    std::string int64Alg;
    SelectorStatus int64Ret = selectorBase.SelectDPUAlgo(&topo, int64Param, {}, int64Alg);
    std::string uint64Alg;
    SelectorStatus uint64Ret = selectorBase.SelectDPUAlgo(&topo, uint64Param, {}, uint64Alg);
    std::string fp64Alg;
    SelectorStatus fp64Ret = selectorBase.SelectDPUAlgo(&topo, fp64Param, {}, fp64Alg);
    std::string prodAlg;
    SelectorStatus prodRet = selectorBase.SelectDPUAlgo(&topo, prodParam, {}, prodAlg);

    // then
    EXPECT_EQ(int64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(int64Alg, "DpuReduceScatterSequenceMeshMesh");
    EXPECT_EQ(uint64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(uint64Alg, "DpuReduceScatterSequenceMeshMesh");
    EXPECT_EQ(fp64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(fp64Alg, "DpuReduceScatterSequenceMeshMesh");
    EXPECT_EQ(prodRet, SelectorStatus::MATCH);
    EXPECT_EQ(prodAlg, "DpuReduceScatterSequenceMeshMesh");
}

TEST(ReduceScatterDpuPipelineGuardTest, NonSpecialInputSelectsPipelineAlgo)
{
    // given
    TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshClosNoPcieTopo();
    OpParam param = MakeNormalInputParam(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    ReduceScatterAutoSelector selector;
    AutoSelectorBase& selectorBase = selector;

    // when
    std::string algName;
    SelectorStatus ret = selectorBase.SelectDPUAlgo(&topo, param, {}, algName);

    // then
    EXPECT_EQ(ret, SelectorStatus::MATCH);
    EXPECT_EQ(algName, "DpuReduceScatterPipeLineMeshNHRMesh");
}

// ---------- 守护点 5: ReduceScatter SelectNdaAlgo ----------
TEST(ReduceScatterNdaPipelineGuardTest, SpecialInputsFallBackToSequenceAlgo)
{
    // given
    TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshClosNoPcieTopo();
    HcclCMDType opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    OpParam int64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam uint64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_UINT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam fp64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam prodParam = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_PROD);
    ReduceScatterAutoSelector selector;
    AutoSelectorBase& selectorBase = selector;

    // when
    std::string int64Alg;
    SelectorStatus int64Ret = selectorBase.SelectNdaAlgo(&topo, int64Param, {}, int64Alg);
    std::string uint64Alg;
    SelectorStatus uint64Ret = selectorBase.SelectNdaAlgo(&topo, uint64Param, {}, uint64Alg);
    std::string fp64Alg;
    SelectorStatus fp64Ret = selectorBase.SelectNdaAlgo(&topo, fp64Param, {}, fp64Alg);
    std::string prodAlg;
    SelectorStatus prodRet = selectorBase.SelectNdaAlgo(&topo, prodParam, {}, prodAlg);

    // then
    EXPECT_EQ(int64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(int64Alg, "DpuReduceScatterSequenceMeshMeshNDA");
    EXPECT_EQ(uint64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(uint64Alg, "DpuReduceScatterSequenceMeshMeshNDA");
    EXPECT_EQ(fp64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(fp64Alg, "DpuReduceScatterSequenceMeshMeshNDA");
    EXPECT_EQ(prodRet, SelectorStatus::MATCH);
    EXPECT_EQ(prodAlg, "DpuReduceScatterSequenceMeshMeshNDA");
}

TEST(ReduceScatterNdaPipelineGuardTest, NonSpecialInputSelectsPipelineAlgo)
{
    // given
    TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshClosNoPcieTopo();
    OpParam param = MakeNormalInputParam(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    ReduceScatterAutoSelector selector;
    AutoSelectorBase& selectorBase = selector;

    // when
    std::string algName;
    SelectorStatus ret = selectorBase.SelectNdaAlgo(&topo, param, {}, algName);

    // then
    EXPECT_EQ(ret, SelectorStatus::MATCH);
    EXPECT_EQ(algName, "DpuReduceScatterPipeLineMeshNHRMeshNDA");
}

// ---------- 守护点 6: Reduce SelectNdaAlgo ----------
TEST(ReduceNdaPipelineGuardTest, SpecialInputsFallBackToSequenceAlgo)
{
    // given
    TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshClosNoPcieTopo();
    HcclCMDType opType = HcclCMDType::HCCL_CMD_REDUCE;
    OpParam int64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam uint64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_UINT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam fp64Param = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam prodParam = MakeSpecialInputParam(opType, HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_PROD);
    ReduceAutoSelector selector;
    AutoSelectorBase& selectorBase = selector;

    // when
    std::string int64Alg;
    SelectorStatus int64Ret = selectorBase.SelectNdaAlgo(&topo, int64Param, {}, int64Alg);
    std::string uint64Alg;
    SelectorStatus uint64Ret = selectorBase.SelectNdaAlgo(&topo, uint64Param, {}, uint64Alg);
    std::string fp64Alg;
    SelectorStatus fp64Ret = selectorBase.SelectNdaAlgo(&topo, fp64Param, {}, fp64Alg);
    std::string prodAlg;
    SelectorStatus prodRet = selectorBase.SelectNdaAlgo(&topo, prodParam, {}, prodAlg);

    // then
    EXPECT_EQ(int64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(int64Alg, "DpuReduceSequenceMeshNHRNDA");
    EXPECT_EQ(uint64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(uint64Alg, "DpuReduceSequenceMeshNHRNDA");
    EXPECT_EQ(fp64Ret, SelectorStatus::MATCH);
    EXPECT_EQ(fp64Alg, "DpuReduceSequenceMeshNHRNDA");
    EXPECT_EQ(prodRet, SelectorStatus::MATCH);
    EXPECT_EQ(prodAlg, "DpuReduceSequenceMeshNHRNDA");
}

TEST(ReduceNdaPipelineGuardTest, NonSpecialInputSelectsPipelineAlgo)
{
    // given
    TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshClosNoPcieTopo();
    OpParam param = MakeNormalInputParam(HcclCMDType::HCCL_CMD_REDUCE);
    ReduceAutoSelector selector;
    AutoSelectorBase& selectorBase = selector;

    // when
    std::string algName;
    SelectorStatus ret = selectorBase.SelectNdaAlgo(&topo, param, {}, algName);

    // then
    EXPECT_EQ(ret, SelectorStatus::MATCH);
    EXPECT_EQ(algName, "DpuReducePipeLineMeshNHRNHRNDA");
}

// ---------- hostDPUOnly 路径: 选路结果直接生效, 不经 costTable 属性过滤 ----------
TEST(AutoSelectorHostDpuOnlyTest, ReduceScatterSpecialInputSelectsSupportedSequenceAlgo)
{
    // given
    hccl_ut::SetHostDpuOnly(true);
    TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshClosNoPcieTopo();
    OpParam param = MakeSpecialInputParam(
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM);
    ReduceScatterAutoSelector selector;
    std::string algName;

    // when
    SelectorStatus ret = selector.Select(param, &topo, algName);
    hccl_ut::SetHostDpuOnly(false);

    // then
    EXPECT_EQ(ret, SelectorStatus::MATCH);
    EXPECT_EQ(algName, "DpuReduceScatterSequenceMeshMesh");
}
