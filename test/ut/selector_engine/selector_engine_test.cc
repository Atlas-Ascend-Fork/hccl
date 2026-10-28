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

TEST(AutoSelectorBaseTest, Int64ReduceOpOnPcieMeshTriggersRollback)
{
    OpParam param = MakeParam(OpExecuteConfig::CCU_MS, HcclCMDType::HCCL_CMD_ALLREDUCE, HCCL_DATA_TYPE_INT64);
    TopoInfoWithNetLayerDetails topo = MakeRollbackTopo();
    EXPECT_TRUE(ops_hccl::AutoSelectorBase::IsRollBackAiv(param, &topo));
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
