/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "auto_selector_base.h"
#include "cost_table.h"
#include "dlhcomm_function.h"
#include "reduce_auto_selector.h"
#include "selector_deps_stub.h"
#include "selector_engine.h"

namespace {
using ops_hccl::OpExecuteConfig;
using ops_hccl::OpParam;
using ops_hccl::TopoInfoWithNetLayerDetails;

// IsRollBackAiv 的触发拓扑: level0PcieMix && level0BigClosRange
TopoInfoWithNetLayerDetails MakeRollbackTopo()
{
    TopoInfoWithNetLayerDetails topo{};
    topo.level0PcieMix = true;
    topo.level0BigClosRange = true;
    return topo;
}

OpParam MakeParam(OpExecuteConfig config, HcclCMDType opType, HcclDataType dataType)
{
    OpParam param{};
    param.opExecuteConfig = config;
    param.opType = opType;
    param.DataDes.dataType = dataType;
    return param;
}

OpParam MakeReduceOpParam(HcclDataType dataType, HcclReduceOp reduceType)
{
    OpParam param{};
    param.DataDes.dataType = dataType;
    param.reduceType = reduceType;
    return param;
}

std::vector<OpExecuteConfig> EngineList(std::initializer_list<OpExecuteConfig> engines)
{
    return std::vector<OpExecuteConfig>(engines);
}
} // namespace

// ---------- IsRollBackAiv 触发判据: only 报错分支与基模式回退分支共用该条件 ----------
TEST(AutoSelectorBaseTest, AllToAllOnPcieMeshBigClosTriggersRollback)
{
    OpParam param = MakeParam(OpExecuteConfig::CCU_MS, HcclCMDType::HCCL_CMD_ALLTOALL, HCCL_DATA_TYPE_FP16);
    TopoInfoWithNetLayerDetails topo = MakeRollbackTopo();
    EXPECT_TRUE(ops_hccl::AutoSelectorBase::IsRollBackAiv(param, &topo));
}

// int64 与 fp64/uint64/PROD 统一走 AICPU（Issue #993）：PCIE 场景不再触发 AIV 回退
TEST(AutoSelectorBaseTest, Int64ReduceOpOnPcieMeshDoesNotTriggerRollback)
{
    OpParam param = MakeParam(OpExecuteConfig::CCU_MS, HcclCMDType::HCCL_CMD_ALLREDUCE, HCCL_DATA_TYPE_INT64);
    TopoInfoWithNetLayerDetails topo = MakeRollbackTopo();
    EXPECT_FALSE(ops_hccl::AutoSelectorBase::IsRollBackAiv(param, &topo));
}

TEST(AutoSelectorBaseTest, NonInt64ReduceOpDoesNotTriggerRollback)
{
    OpParam param = MakeParam(OpExecuteConfig::CCU_MS, HcclCMDType::HCCL_CMD_ALLREDUCE, HCCL_DATA_TYPE_FP16);
    TopoInfoWithNetLayerDetails topo = MakeRollbackTopo();
    EXPECT_FALSE(ops_hccl::AutoSelectorBase::IsRollBackAiv(param, &topo));
}

TEST(AutoSelectorBaseTest, AllToAllWithoutBigClosRangeDoesNotTriggerRollback)
{
    OpParam param = MakeParam(OpExecuteConfig::CCU_MS, HcclCMDType::HCCL_CMD_ALLTOALL, HCCL_DATA_TYPE_FP16);
    TopoInfoWithNetLayerDetails topo{};
    topo.level0PcieMix = true;
    topo.level0BigClosRange = false;
    EXPECT_FALSE(ops_hccl::AutoSelectorBase::IsRollBackAiv(param, &topo));
}

TEST(AutoSelectorBaseTest, PlainTopoDoesNotTriggerRollback)
{
    OpParam param = MakeParam(OpExecuteConfig::CCU_MS, HcclCMDType::HCCL_CMD_ALLREDUCE, HCCL_DATA_TYPE_FP16);
    TopoInfoWithNetLayerDetails topo{};
    EXPECT_FALSE(ops_hccl::AutoSelectorBase::IsRollBackAiv(param, &topo));
}

// ---------- only 模式命中回退条件时直接报错, 不降级到 AIV ----------
TEST(SelectorEngineRunTest, CcuMsOnlyModeRejectsRollbackToAiv)
{
    hccl_ut::ResetSelectorStubState();
    OpParam param = MakeParam(OpExecuteConfig::CCU_MS_ONLY, HcclCMDType::HCCL_CMD_ALLTOALL, HCCL_DATA_TYPE_FP16);
    TopoInfoWithNetLayerDetails topo = MakeRollbackTopo();
    std::string algName;
    HcclResult ret = ops_hccl::SelectorEngine::Global()->Run(nullptr, param, &topo, algName);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(param.opExecuteConfig, OpExecuteConfig::CCU_MS_ONLY);
}

TEST(SelectorEngineRunTest, CcuSchedOnlyModeRejectsRollbackToAiv)
{
    hccl_ut::ResetSelectorStubState();
    OpParam param = MakeParam(OpExecuteConfig::CCU_SCHED_ONLY, HcclCMDType::HCCL_CMD_ALLTOALL, HCCL_DATA_TYPE_FP16);
    TopoInfoWithNetLayerDetails topo = MakeRollbackTopo();
    std::string algName;
    HcclResult ret = ops_hccl::SelectorEngine::Global()->Run(nullptr, param, &topo, algName);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(param.opExecuteConfig, OpExecuteConfig::CCU_SCHED_ONLY);
}

// ---------- 基模式相同输入仍走原回退逻辑(切换到 AIV_ONLY), 不误报 ----------
TEST(SelectorEngineRunTest, BaseCcuMsModeStillRollsBackToAiv)
{
    hccl_ut::ResetSelectorStubState();
    hccl_ut::SetTunerInitResult(HCCL_E_INTERNAL);
    OpParam param = MakeParam(OpExecuteConfig::CCU_MS, HcclCMDType::HCCL_CMD_ALLTOALL, HCCL_DATA_TYPE_FP16);
    TopoInfoWithNetLayerDetails topo = MakeRollbackTopo();
    std::string algName;
    HcclResult ret = ops_hccl::SelectorEngine::Global()->Run(nullptr, param, &topo, algName);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(param.opExecuteConfig, OpExecuteConfig::AIV_ONLY);
}

TEST(SelectorEngineRunTest, BaseCcuSchedModeStillRollsBackToAiv)
{
    hccl_ut::ResetSelectorStubState();
    hccl_ut::SetTunerInitResult(HCCL_E_INTERNAL);
    OpParam param = MakeParam(OpExecuteConfig::CCU_SCHED, HcclCMDType::HCCL_CMD_ALLTOALL, HCCL_DATA_TYPE_FP16);
    TopoInfoWithNetLayerDetails topo = MakeRollbackTopo();
    std::string algName;
    HcclResult ret = ops_hccl::SelectorEngine::Global()->Run(nullptr, param, &topo, algName);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(param.opExecuteConfig, OpExecuteConfig::AIV_ONLY);
}

// ---------- 引擎候选链: only 模式无回退链且选不到算法时报错; 基模式保留完整回退序 ----------
TEST(SelectorEngineRunTest, CcuMsOnlyModeHasNoFallbackEngineChain)
{
    hccl_ut::ResetSelectorStubState();
    OpParam param = MakeParam(OpExecuteConfig::CCU_MS_ONLY, HcclCMDType::HCCL_CMD_ALLREDUCE, HCCL_DATA_TYPE_FP16);
    TopoInfoWithNetLayerDetails topo{};
    std::string algName;
    HcclResult ret = ops_hccl::SelectorEngine::Global()->Run(nullptr, param, &topo, algName);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(hccl_ut::LastCandidateEngines(), EngineList({OpExecuteConfig::CCU_MS}));
    EXPECT_EQ(param.opExecuteConfig, OpExecuteConfig::CCU_MS_ONLY);
}

TEST(SelectorEngineRunTest, CcuSchedOnlyModeHasNoFallbackEngineChain)
{
    hccl_ut::ResetSelectorStubState();
    OpParam param = MakeParam(OpExecuteConfig::CCU_SCHED_ONLY, HcclCMDType::HCCL_CMD_ALLREDUCE, HCCL_DATA_TYPE_FP16);
    TopoInfoWithNetLayerDetails topo{};
    std::string algName;
    HcclResult ret = ops_hccl::SelectorEngine::Global()->Run(nullptr, param, &topo, algName);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(hccl_ut::LastCandidateEngines(), EngineList({OpExecuteConfig::CCU_SCHED}));
    EXPECT_EQ(param.opExecuteConfig, OpExecuteConfig::CCU_SCHED_ONLY);
}

TEST(SelectorEngineRunTest, BaseCcuMsModeKeepsFullFallbackEngineChain)
{
    hccl_ut::ResetSelectorStubState();
    OpParam param = MakeParam(OpExecuteConfig::CCU_MS, HcclCMDType::HCCL_CMD_ALLREDUCE, HCCL_DATA_TYPE_FP16);
    TopoInfoWithNetLayerDetails topo{};
    std::string algName;
    HcclResult ret = ops_hccl::SelectorEngine::Global()->Run(nullptr, param, &topo, algName);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(
        hccl_ut::LastCandidateEngines(), EngineList(
                                             {OpExecuteConfig::CCU_MS, OpExecuteConfig::CCU_SCHED,
                                              OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU}));
}

TEST(SelectorEngineRunTest, BaseCcuSchedModeKeepsFullFallbackEngineChain)
{
    hccl_ut::ResetSelectorStubState();
    OpParam param = MakeParam(OpExecuteConfig::CCU_SCHED, HcclCMDType::HCCL_CMD_ALLREDUCE, HCCL_DATA_TYPE_FP16);
    TopoInfoWithNetLayerDetails topo{};
    std::string algName;
    HcclResult ret = ops_hccl::SelectorEngine::Global()->Run(nullptr, param, &topo, algName);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(
        hccl_ut::LastCandidateEngines(),
        EngineList({OpExecuteConfig::CCU_SCHED, OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU}));
}

// ---------- 正向命中链路: only 模式经候选引擎/前缀过滤后选中 CCU 算法 ----------
TEST(SelectorEngineRunTest, CcuMsOnlyModeSelectsCcuMsAlgo)
{
    hccl_ut::ResetSelectorStubState();
    hccl_ut::SetCostModelAlgos({"CcuMSAllReduceRing", "AicpuAllReduceNHR"});
    OpParam param = MakeParam(OpExecuteConfig::CCU_MS_ONLY, HcclCMDType::HCCL_CMD_ALLREDUCE, HCCL_DATA_TYPE_FP16);
    TopoInfoWithNetLayerDetails topo{};
    std::string algName;
    HcclResult ret = ops_hccl::SelectorEngine::Global()->Run(nullptr, param, &topo, algName);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algName, "CcuMSAllReduceRing");
    EXPECT_EQ(hccl_ut::LastCandidateEngines(), EngineList({OpExecuteConfig::CCU_MS}));
    EXPECT_EQ(hccl_ut::LastCandidatePrefixes(), std::vector<std::string>({"CcuMS"}));
    // 归一化为基模式发生在 op_common.cc 的 Selector(), SelectorEngine 层保持 only 配置不变
    EXPECT_EQ(param.opExecuteConfig, OpExecuteConfig::CCU_MS_ONLY);
}

TEST(SelectorEngineRunTest, CcuSchedOnlyModeSelectsCcuSchedAlgo)
{
    hccl_ut::ResetSelectorStubState();
    hccl_ut::SetCostModelAlgos({"CcuSchedAllReduceRing", "AicpuAllReduceNHR"});
    OpParam param = MakeParam(OpExecuteConfig::CCU_SCHED_ONLY, HcclCMDType::HCCL_CMD_ALLREDUCE, HCCL_DATA_TYPE_FP16);
    TopoInfoWithNetLayerDetails topo{};
    std::string algName;
    HcclResult ret = ops_hccl::SelectorEngine::Global()->Run(nullptr, param, &topo, algName);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algName, "CcuSchedAllReduceRing");
    EXPECT_EQ(hccl_ut::LastCandidateEngines(), EngineList({OpExecuteConfig::CCU_SCHED}));
    EXPECT_EQ(hccl_ut::LastCandidatePrefixes(), std::vector<std::string>({"CcuSched"}));
    EXPECT_EQ(param.opExecuteConfig, OpExecuteConfig::CCU_SCHED_ONLY);
}

// ---------- GetOpDataType: 按算子类型读union对应成员(取错成员即日志失真根因) ----------
TEST(GetOpDataTypeTest, VOpReadsVDataDesUnionMember)
{
    OpParam param{};
    param.vDataDes.dataType = HCCL_DATA_TYPE_FP16; // union偏移16, 与DataDes.dataType(偏移8)不同
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER_V;
    EXPECT_EQ(ops_hccl::GetOpDataType(param), HCCL_DATA_TYPE_FP16);
}

// ---------- IsDataTypeOrReduceTypeSpecial: 64bit 数据类型与 PROD 归约不支持 DPU pipeline ----------
// 三种64bit类型(INT64/UINT64/FP64)都必须命中, 漏判会让该类型的 Reduce/ReduceScatter 选到 pipeline 算法
TEST(IsDataTypeOrReduceTypeSpecialTest, All64BitDataTypesAreSpecial)
{
    // given
    OpParam int64Param = MakeReduceOpParam(HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam uint64Param = MakeReduceOpParam(HCCL_DATA_TYPE_UINT64, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam fp64Param = MakeReduceOpParam(HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_SUM);
    // when
    // then
    EXPECT_TRUE(ops_hccl::IsDataTypeOrReduceTypeSpecial(int64Param));
    EXPECT_TRUE(ops_hccl::IsDataTypeOrReduceTypeSpecial(uint64Param));
    EXPECT_TRUE(ops_hccl::IsDataTypeOrReduceTypeSpecial(fp64Param));
}

// 非64bit类型不命中; INT128(16字节)不属于 Is64BitDataType, 沿用普通类型选路
TEST(IsDataTypeOrReduceTypeSpecialTest, Non64BitDataTypesAreNotSpecial)
{
    // given
    OpParam int8Param = MakeReduceOpParam(HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam uint32Param = MakeReduceOpParam(HCCL_DATA_TYPE_UINT32, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam fp32Param = MakeReduceOpParam(HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam bfp16Param = MakeReduceOpParam(HCCL_DATA_TYPE_BFP16, HcclReduceOp::HCCL_REDUCE_SUM);
    OpParam int128Param = MakeReduceOpParam(HCCL_DATA_TYPE_INT128, HcclReduceOp::HCCL_REDUCE_SUM);

    // when
    // then
    EXPECT_FALSE(ops_hccl::IsDataTypeOrReduceTypeSpecial(int8Param));
    EXPECT_FALSE(ops_hccl::IsDataTypeOrReduceTypeSpecial(uint32Param));
    EXPECT_FALSE(ops_hccl::IsDataTypeOrReduceTypeSpecial(fp32Param));
    EXPECT_FALSE(ops_hccl::IsDataTypeOrReduceTypeSpecial(bfp16Param));
    EXPECT_FALSE(ops_hccl::IsDataTypeOrReduceTypeSpecial(int128Param));
}

// PROD 与数据类型无关, 非64bit也必须命中; 与64bit同时满足时同样只判为特殊
TEST(IsDataTypeOrReduceTypeSpecialTest, ProdReduceTypeIsSpecialRegardlessOfDataType)
{
    // given
    OpParam fp16Param = MakeReduceOpParam(HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_PROD);
    OpParam int64Param = MakeReduceOpParam(HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_PROD);

    // when
    // then
    EXPECT_TRUE(ops_hccl::IsDataTypeOrReduceTypeSpecial(fp16Param));
    EXPECT_TRUE(ops_hccl::IsDataTypeOrReduceTypeSpecial(int64Param));
}

// 只有 PROD 需要挡住 pipeline, 其余归约类型不受影响
TEST(IsDataTypeOrReduceTypeSpecialTest, NonProdReduceTypesAreNotSpecial)
{
    // given
    OpParam maxParam = MakeReduceOpParam(HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_MAX);
    OpParam minParam = MakeReduceOpParam(HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_MIN);

    // when
    // then
    EXPECT_FALSE(ops_hccl::IsDataTypeOrReduceTypeSpecial(maxParam));
    EXPECT_FALSE(ops_hccl::IsDataTypeOrReduceTypeSpecial(minParam));
}

// 默认构造(RESERVED/RESERVED)不能被误判为特殊, 否则默认配置会整体绕开 pipeline 选路
TEST(IsDataTypeOrReduceTypeSpecialTest, DefaultOpParamIsNotSpecial)
{
    // given
    OpParam param{};

    // when
    // then
    EXPECT_FALSE(ops_hccl::IsDataTypeOrReduceTypeSpecial(param));
}

// ---------- ReduceAutoSelector 门禁回归锁(Issue #1030): MESH_1D_CLOS 全互连分支拦截 INT8+SUM ----------
namespace {
TopoInfoWithNetLayerDetails MakeMesh1dClosAllConnectedTopo()
{
    TopoInfoWithNetLayerDetails topo{};
    topo.topoLevelNums = 1;
    topo.userRankSize = 4;
    topo.level0Topo = ops_hccl::Level0Shape::MESH_1D_CLOS;
    topo.topoInstDetailsOfLayer.resize(1);
    // 单个 CLOS 实例内 4 rank 全互连（IsLayerAllConnetedWithTopo 读 rankNumForTopoType 与 localNetInsSizeOfLayer）
    topo.topoInstDetailsOfLayer[0].topoInstNum = 1;
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[CommTopo::COMM_TOPO_1DMESH] = {4};
    topo.netLayerDetails.localNetInsSizeOfLayer = {4};
    return topo;
}

OpParam MakeReduceParam(HcclDataType dataType, HcclReduceOp reduceOp)
{
    OpParam param{};
    param.opExecuteConfig = OpExecuteConfig::CCU_SCHED;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE;
    param.DataDes.dataType = dataType;
    param.DataDes.count = 1024; // 1KB, 小于 8MB 阈值
    param.reduceType = reduceOp;
    return param;
}

using ops_hccl::ReduceAutoSelector;
using ops_hccl::SelectorStatus;
} // namespace

TEST(ReduceAutoSelectorTest, Int8SumRejectedByCcuScheduleAndFallsBackToAicpu)
{
    ReduceAutoSelector selector;
    OpParam param = MakeReduceParam(HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM);
    TopoInfoWithNetLayerDetails topo = MakeMesh1dClosAllConnectedTopo();
    std::string algName;
    EXPECT_EQ(selector.Select(param, &topo, algName), SelectorStatus::MATCH);
    EXPECT_EQ(algName, "AicpuReduceSoleMesh");
    EXPECT_EQ(param.opExecuteConfig, OpExecuteConfig::AICPU_TS);
}

// ---------- 链接桩: 仅 auto selector 的 DPU/AIV 分支引用, 门禁口径 UT 不执行到 ----------
namespace ops_hccl {
bool CanUseDpuInlineReduce(const OpParam&, const TopoInfoWithNetLayerDetails*) { return false; }

DlHcommFunction::~DlHcommFunction() = default;

DlHcommFunction& DlHcommFunction::GetInstance()
{
    static DlHcommFunction instance;
    return instance;
}
} // namespace ops_hccl

extern "C" {
int GetHcommVersion(void) { return 0; }

HcclResult HcclGetHcclBuffer(HcclComm, void**, uint64_t*) { return HCCL_E_PARA; }
}
