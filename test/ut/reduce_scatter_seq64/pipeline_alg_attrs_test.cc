/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 本 PR 修改过注册属性的 10 处 pipeline 算法(6 个 DPU + 4 个 AICPU)的 costTable 过滤 UT。
// 本目标编入生产 executor 的 REGISTER_ALG_ATTRS（registration_stub.h 仅替换 executor 构造），
// 直接读取真实 AlgAttrsRegistry，不再由测试注入属性；生产注册漏调 SetPipelineAlgoOpAttrs 时用例失败。
// 验证 cost_table 的过滤合同: INT64/UINT64/FP64/PROD 入参下不残留 PipeLine 候选。

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "alg_attrs.h"
#include "alg_attrs_registry.h"
#include "cost_model.h"
#include "cost_table.h"

namespace {
using ops_hccl::CostAlgoParams;
using ops_hccl::CostModel;
using ops_hccl::CostModelParam;
using ops_hccl::CostTable;
using ops_hccl::CostTableManager;
using ops_hccl::OpParam;
using ops_hccl::TopoInfoWithNetLayerDetails;

struct PipelineAlgoCase {
    const char* name;
    HcclCMDType opType;
};

// 本 PR 改了注册属性的 10 处 pipeline 算法
const PipelineAlgoCase PIPELINE_ALGOS[] = {
    {"AicpuAllReducePipeLineMeshNHR", HcclCMDType::HCCL_CMD_ALLREDUCE},
    {"AicpuAllReducePipeLineMeshNHRNHR", HcclCMDType::HCCL_CMD_ALLREDUCE},
    {"DpuAllReducePipeLineMeshNHRNHR", HcclCMDType::HCCL_CMD_ALLREDUCE},
    {"DpuAllReducePipeLineMeshNHRNHRNDA", HcclCMDType::HCCL_CMD_ALLREDUCE},
    {"DpuReducePipeLineMeshNHRNHR", HcclCMDType::HCCL_CMD_REDUCE},
    {"DpuReducePipeLineMeshNHRNHRNDA", HcclCMDType::HCCL_CMD_REDUCE},
    {"AicpuReduceScatterPipeLineMeshNHR", HcclCMDType::HCCL_CMD_REDUCE_SCATTER},
    {"AicpuReduceScatterPipeLineMeshNHRMesh", HcclCMDType::HCCL_CMD_REDUCE_SCATTER},
    {"DpuReduceScatterPipeLineMeshNHRMesh", HcclCMDType::HCCL_CMD_REDUCE_SCATTER},
    {"DpuReduceScatterPipeLineMeshNHRMeshNDA", HcclCMDType::HCCL_CMD_REDUCE_SCATTER},
};

// CostModel 存储: params 提供每段系数, algos 持有指向 params 的指针
struct CostModelFixture {
    std::vector<CostModelParam> params;
    std::vector<CostAlgoParams> algos;
    CostModel model{nullptr, 0};

    explicit CostModelFixture(const std::vector<const char*>& names)
    {
        params.assign(names.size(), CostModelParam{1.0f, 1.0f, 0.0f, 0.0f, 0.0f});
        algos.reserve(names.size());
        for (size_t i = 0; i < names.size(); ++i) {
            CostAlgoParams algo{};
            algo.algName = names[i];
            algo.param = &params[i];
            algo.count = 1;
            algo.hcclAlgoPriority = 0;
            algos.push_back(algo);
        }
        model.costAlgoParams = algos.data();
        model.count = static_cast<int>(algos.size());
    }
};

// 该算子的 pipeline 算法名: AllReduce 4 个(Aicpu 2 + Dpu 2), Reduce 2 个(Dpu 2), ReduceScatter 4 个(Aicpu 2 + Dpu 2)
std::vector<const char*> PipelineAlgoNames(HcclCMDType opType)
{
    std::vector<const char*> names;
    for (const auto& algo : PIPELINE_ALGOS) {
        if (algo.opType == opType) {
            names.push_back(algo.name);
        }
    }
    return names;
}

TopoInfoWithNetLayerDetails MakeTopo()
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRankSize = 16;
    topo.topoLevelNums = 3;
    return topo;
}

OpParam MakeOpParam(HcclCMDType opType, HcclDataType dataType, HcclReduceOp reduceType)
{
    OpParam param{};
    param.opType = opType;
    param.DataDes.count = 1024;
    param.DataDes.dataType = dataType;
    param.reduceType = reduceType;
    return param;
}

// 按注册属性生成 costTable, 返回过滤后幸存的算法名
std::vector<std::string>
BuildCandidateNames(CostModel& model, const TopoInfoWithNetLayerDetails& topo, const OpParam& opParam)
{
    CostTable costTable{};
    EXPECT_EQ(CostTableManager::Global()->GenerateCostTable(model, costTable, &topo, opParam), HCCL_SUCCESS);
    std::vector<std::string> names;
    for (int i = 0; i < costTable.count; ++i) {
        names.emplace_back(costTable.costs[i].algName);
    }
    CostTableManager::Global()->FreeCostTable(costTable);
    return names;
}
} // namespace

// ---------- 生产注册属性守护: 直接读 executor 的真实 REGISTER_ALG_ATTRS ----------
TEST(PipelineAlgAttrsTest, ProductionRegistryAttrsMatchSetPipelineAlgoOpAttrs)
{
    for (const auto& algo : PIPELINE_ALGOS) {
        const ops_hccl::AlgAttrs* actual = ops_hccl::AlgAttrsRegistry::Instance().Get(algo.name);
        ASSERT_NE(actual, nullptr) << "missing production registration: " << algo.name;

        ops_hccl::AlgAttrs expected;
        expected.opType = algo.opType;
        ops_hccl::SetPipelineAlgoOpAttrs(expected.op);

        EXPECT_EQ(actual->opType, expected.opType) << algo.name;
        EXPECT_EQ(actual->op.isSupportProd, expected.op.isSupportProd) << algo.name;
        EXPECT_EQ(actual->op.unsupportedDataTypes, expected.op.unsupportedDataTypes) << algo.name;
    }
}

// ---------- AllReduce: 属性过滤后 costTable 无 pipeline 候选 ----------
TEST(PipelineAlgAttrsTest, AllReducePipelineCandidatesDroppedForSpecialInputs)
{
    // given
    CostModelFixture fixture(PipelineAlgoNames(HcclCMDType::HCCL_CMD_ALLREDUCE));
    TopoInfoWithNetLayerDetails topo = MakeTopo();
    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    OpParam int64Param = MakeOpParam(opType, HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam uint64Param = MakeOpParam(opType, HCCL_DATA_TYPE_UINT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam fp64Param = MakeOpParam(opType, HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam prodParam = MakeOpParam(opType, HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_PROD);
    OpParam normalParam = MakeOpParam(opType, HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_SUM);

    // when
    std::vector<std::string> int64Names = BuildCandidateNames(fixture.model, topo, int64Param);
    std::vector<std::string> uint64Names = BuildCandidateNames(fixture.model, topo, uint64Param);
    std::vector<std::string> fp64Names = BuildCandidateNames(fixture.model, topo, fp64Param);
    std::vector<std::string> prodNames = BuildCandidateNames(fixture.model, topo, prodParam);
    std::vector<std::string> normalNames = BuildCandidateNames(fixture.model, topo, normalParam);

    // then
    EXPECT_TRUE(int64Names.empty());
    EXPECT_TRUE(uint64Names.empty());
    EXPECT_TRUE(fp64Names.empty());
    EXPECT_TRUE(prodNames.empty());
    EXPECT_EQ(normalNames.size(), 4u);
}

// ---------- Reduce: 属性过滤后 costTable 无 pipeline 候选 ----------
TEST(PipelineAlgAttrsTest, ReducePipelineCandidatesDroppedForSpecialInputs)
{
    // given
    CostModelFixture fixture(PipelineAlgoNames(HcclCMDType::HCCL_CMD_REDUCE));
    TopoInfoWithNetLayerDetails topo = MakeTopo();
    HcclCMDType opType = HcclCMDType::HCCL_CMD_REDUCE;
    OpParam int64Param = MakeOpParam(opType, HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam uint64Param = MakeOpParam(opType, HCCL_DATA_TYPE_UINT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam fp64Param = MakeOpParam(opType, HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam prodParam = MakeOpParam(opType, HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_PROD);
    OpParam normalParam = MakeOpParam(opType, HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_SUM);

    // when
    std::vector<std::string> int64Names = BuildCandidateNames(fixture.model, topo, int64Param);
    std::vector<std::string> uint64Names = BuildCandidateNames(fixture.model, topo, uint64Param);
    std::vector<std::string> fp64Names = BuildCandidateNames(fixture.model, topo, fp64Param);
    std::vector<std::string> prodNames = BuildCandidateNames(fixture.model, topo, prodParam);
    std::vector<std::string> normalNames = BuildCandidateNames(fixture.model, topo, normalParam);

    // then
    EXPECT_TRUE(int64Names.empty());
    EXPECT_TRUE(uint64Names.empty());
    EXPECT_TRUE(fp64Names.empty());
    EXPECT_TRUE(prodNames.empty());
    EXPECT_EQ(normalNames.size(), 2u);
}

// ---------- ReduceScatter: 属性过滤后 costTable 无 pipeline 候选 ----------
TEST(PipelineAlgAttrsTest, ReduceScatterPipelineCandidatesDroppedForSpecialInputs)
{
    // given
    CostModelFixture fixture(PipelineAlgoNames(HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    TopoInfoWithNetLayerDetails topo = MakeTopo();
    HcclCMDType opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    OpParam int64Param = MakeOpParam(opType, HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam uint64Param = MakeOpParam(opType, HCCL_DATA_TYPE_UINT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam fp64Param = MakeOpParam(opType, HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam prodParam = MakeOpParam(opType, HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_PROD);
    OpParam normalParam = MakeOpParam(opType, HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_SUM);

    // when
    std::vector<std::string> int64Names = BuildCandidateNames(fixture.model, topo, int64Param);
    std::vector<std::string> uint64Names = BuildCandidateNames(fixture.model, topo, uint64Param);
    std::vector<std::string> fp64Names = BuildCandidateNames(fixture.model, topo, fp64Param);
    std::vector<std::string> prodNames = BuildCandidateNames(fixture.model, topo, prodParam);
    std::vector<std::string> normalNames = BuildCandidateNames(fixture.model, topo, normalParam);

    // then
    EXPECT_TRUE(int64Names.empty());
    EXPECT_TRUE(uint64Names.empty());
    EXPECT_TRUE(fp64Names.empty());
    EXPECT_TRUE(prodNames.empty());
    EXPECT_EQ(normalNames.size(), 4u);
}
