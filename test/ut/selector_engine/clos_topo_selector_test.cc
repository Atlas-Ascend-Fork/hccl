/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "auto_selector_base.h"
#include "dlsym_common.h"
#include "dlhcomm_function.h"
#include "dpu_inline_reduce_selector.h"
#include "selector_registry.h"

namespace {
using namespace ops_hccl;

constexpr u64 KB = 1024;
constexpr u64 CLOS_SMALL_BYTES = 4 * KB;
constexpr u64 CLOS_NHR_BYTES = 4 * 1024 * KB;
constexpr u32 CLOS_RANK_SIZE = 8;

TopoInfoWithNetLayerDetails MakeClosTopo(u32 rankSize = CLOS_RANK_SIZE, u32 closInstNum = 1)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.deviceType = HcclDevType::DEV_TYPE_950;
    topo.allDpuSupportInlineReduce = false;
    topo.hostDpuOnly = false;
    topo.userRankSize = rankSize;
    topo.topoLevelNums = 1;
    topo.level0Topo = Level0Shape::CLOS;
    topo.level0PcieMix = false;
    // CLOS Mesh 要求 level0 只有一个 CLOS 拓扑实例（即单 die 的 channel）
    TopoInstDetails details{};
    details.topoInstNum = closInstNum;
    if (closInstNum != 0) {
        const u32 ranksPerInst = rankSize / closInstNum;
        for (u32 i = 0; i < closInstNum; ++i) {
            details.rankNumForTopoType[COMM_TOPO_CLOS].push_back(ranksPerInst);
        }
    }
    topo.topoInstDetailsOfLayer.push_back(details);
    return topo;
}

OpParam MakeClosParam(HcclCMDType op, u64 bytes, HcclDataType dataType = HCCL_DATA_TYPE_FP32, bool inPlace = false)
{
    static u8 inputBuffer[4096];
    static u8 outputBuffer[4096];
    OpParam param{};
    param.opType = op;
    param.opExecuteConfig = OpExecuteConfig::CCU_SCHED;
    param.DataDes.dataType = dataType;
    param.DataDes.count = dataType == HCCL_DATA_TYPE_INT8 ? bytes : bytes / sizeof(float);
    param.reduceType = HCCL_REDUCE_SUM;
    param.inputPtr = inPlace ? static_cast<void*>(outputBuffer) : static_cast<void*>(inputBuffer);
    param.outputPtr = static_cast<void*>(outputBuffer);
    param.inputSize = sizeof(inputBuffer);
    param.outputSize = sizeof(outputBuffer);
    return param;
}

SelectorStatus SelectCcuSchedule(const OpParam& param, TopoInfoWithNetLayerDetails& topo, std::string& name)
{
    auto selectors = SelectorRegistry::Global()->GetSelectorsByOpType(param.opType);
    if (selectors.size() != 1) {
        return SelectorStatus::NOT_MATCH;
    }
    std::map<HcclCMDType, std::vector<HcclAlgoType>> configAlgMap;
    return selectors.begin()->second->SelectCcuScheduleAlgo(&topo, param, configAlgMap, name);
}

class ClosTopoSelectorTest : public testing::Test {};

TEST_F(ClosTopoSelectorTest, AllReduceSelectsMeshForSmallData)
{
    auto topo = MakeClosTopo();
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_ALLREDUCE, CLOS_SMALL_BYTES);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedAllReduceSoleMesh");
}

TEST_F(ClosTopoSelectorTest, AllReduceSelectsNhrForLargeData)
{
    auto topo = MakeClosTopo();
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_ALLREDUCE, CLOS_NHR_BYTES);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedAllReduceSoleNHR");
}

TEST_F(ClosTopoSelectorTest, AllReduceSelectsNhrForInt8)
{
    auto topo = MakeClosTopo();
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_ALLREDUCE, CLOS_SMALL_BYTES, HCCL_DATA_TYPE_INT8);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedAllReduceSoleNHR");
}

TEST_F(ClosTopoSelectorTest, AllGatherSelectsMeshForSmallData)
{
    auto topo = MakeClosTopo();
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_ALLGATHER, CLOS_SMALL_BYTES);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedAllGatherSoleMesh");
}

TEST_F(ClosTopoSelectorTest, AllGatherSelectsNhrForLargeData)
{
    auto topo = MakeClosTopo();
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_ALLGATHER, CLOS_NHR_BYTES);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedAllGatherSoleNHR");
}

TEST_F(ClosTopoSelectorTest, ReduceScatterSelectsMeshForSmallData)
{
    auto topo = MakeClosTopo();
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CLOS_SMALL_BYTES);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedReduceScatterSoleMesh");
}

TEST_F(ClosTopoSelectorTest, ReduceScatterSelectsNhrForLargeData)
{
    auto topo = MakeClosTopo();
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CLOS_NHR_BYTES);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedReduceScatterSoleNHR");
}

TEST_F(ClosTopoSelectorTest, ReduceSelectsMeshForSmallData)
{
    auto topo = MakeClosTopo();
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_REDUCE, CLOS_SMALL_BYTES);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedReduceSoleMesh");
}

TEST_F(ClosTopoSelectorTest, ReduceSelectsNhrForLargeData)
{
    auto topo = MakeClosTopo();
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_REDUCE, CLOS_NHR_BYTES);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedReduceSoleNHR");
}

TEST_F(ClosTopoSelectorTest, ReduceSelectsNhrForInt8)
{
    auto topo = MakeClosTopo();
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_REDUCE, CLOS_SMALL_BYTES, HCCL_DATA_TYPE_INT8);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedReduceSoleNHR");
}

TEST_F(ClosTopoSelectorTest, ReduceSelectsNhrForRankSizeAboveMeshLimit)
{
    auto topo = MakeClosTopo(16);
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_REDUCE, CLOS_SMALL_BYTES);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedReduceSoleNHR");
}

TEST_F(ClosTopoSelectorTest, BroadcastSelectsMeshForSmallData)
{
    auto topo = MakeClosTopo();
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_BROADCAST, CLOS_SMALL_BYTES);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedBroadcastSoleMesh");
}

TEST_F(ClosTopoSelectorTest, BroadcastSelectsNhrForLargeData)
{
    auto topo = MakeClosTopo();
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_BROADCAST, CLOS_NHR_BYTES);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedBroadcastSoleNHR");
}

TEST_F(ClosTopoSelectorTest, ScatterSelectsMeshForSmallData)
{
    auto topo = MakeClosTopo();
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_SCATTER, CLOS_SMALL_BYTES);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedScatterSoleMesh");
}

TEST_F(ClosTopoSelectorTest, ScatterSelectsNhrForLargeData)
{
    auto topo = MakeClosTopo();
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_SCATTER, CLOS_NHR_BYTES);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedScatterSoleNHR");
}

TEST_F(ClosTopoSelectorTest, ScatterSelectsNhrForInPlace)
{
    auto topo = MakeClosTopo();
    auto param = MakeClosParam(HcclCMDType::HCCL_CMD_SCATTER, CLOS_SMALL_BYTES, HCCL_DATA_TYPE_FP32, true);
    std::string name;
    EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "CcuSchedScatterSoleNHR");
}

// 校验六个算子在非单 CLOS 实例的 level0 拓扑下均回退 NHR，不误选 Mesh
void ExpectAllSelectorsFallbackToNhr(TopoInfoWithNetLayerDetails& topo)
{
    const std::vector<std::pair<HcclCMDType, std::string>> cases = {
        {HcclCMDType::HCCL_CMD_ALLREDUCE, "CcuSchedAllReduceSoleNHR"},
        {HcclCMDType::HCCL_CMD_ALLGATHER, "CcuSchedAllGatherSoleNHR"},
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "CcuSchedReduceScatterSoleNHR"},
        {HcclCMDType::HCCL_CMD_REDUCE, "CcuSchedReduceSoleNHR"},
        {HcclCMDType::HCCL_CMD_BROADCAST, "CcuSchedBroadcastSoleNHR"},
        {HcclCMDType::HCCL_CMD_SCATTER, "CcuSchedScatterSoleNHR"},
    };
    for (const auto& item : cases) {
        SCOPED_TRACE(item.second);
        auto param = MakeClosParam(item.first, CLOS_SMALL_BYTES);
        std::string name;
        EXPECT_EQ(SelectCcuSchedule(param, topo, name), SelectorStatus::MATCH);
        EXPECT_EQ(name, item.second);
    }
}

TEST_F(ClosTopoSelectorTest, SelectsNhrForDualClosInst)
{
    // level0 有两个 CLOS 实例时跨 die 建链异常，必须回退 NHR
    auto topo = MakeClosTopo(CLOS_RANK_SIZE, 2);
    ExpectAllSelectorsFallbackToNhr(topo);
}

TEST_F(ClosTopoSelectorTest, SelectsNhrWhenLevel0HasNoClosInst)
{
    // 未识别出 CLOS 实例时不可判定单 die，保守回退 NHR
    auto topo = MakeClosTopo(CLOS_RANK_SIZE, 0);
    ExpectAllSelectorsFallbackToNhr(topo);
}

TEST_F(ClosTopoSelectorTest, SelectsNhrWhenLevel0ClosInstListEmpty)
{
    // CLOS 键存在但实例列表为空（如 level0 为多个 1DMESH 实例兜底置 CLOS），同样不可判定单 die
    auto topo = MakeClosTopo();
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[COMM_TOPO_CLOS].clear();
    ExpectAllSelectorsFallbackToNhr(topo);
}

TEST_F(ClosTopoSelectorTest, IsLevel0SingleClosInstBoundary)
{
    EXPECT_FALSE(AutoSelectorBase::IsLevel0SingleClosInst(nullptr));

    // level0 明细缺失
    TopoInfoWithNetLayerDetails noLayer{};
    EXPECT_FALSE(AutoSelectorBase::IsLevel0SingleClosInst(&noLayer));

    // 单实例: 唯一可判定为单 die 的形态
    auto single = MakeClosTopo();
    EXPECT_TRUE(AutoSelectorBase::IsLevel0SingleClosInst(&single));

    // 双实例
    auto dual = MakeClosTopo(CLOS_RANK_SIZE, 2);
    EXPECT_FALSE(AutoSelectorBase::IsLevel0SingleClosInst(&dual));

    // CLOS 条目缺失
    auto noKey = MakeClosTopo(CLOS_RANK_SIZE, 0);
    EXPECT_FALSE(AutoSelectorBase::IsLevel0SingleClosInst(&noKey));

    // CLOS 条目存在但列表为空
    auto emptyList = MakeClosTopo();
    emptyList.topoInstDetailsOfLayer[0].rankNumForTopoType[COMM_TOPO_CLOS].clear();
    EXPECT_FALSE(AutoSelectorBase::IsLevel0SingleClosInst(&emptyList));
}

TEST_F(ClosTopoSelectorTest, ReduceScatterFallsBackFor64BitAndProd)
{
    // Mesh 注册 unsupportedDataTypes 含 64bit 且 isSupportProd=false, 入口即拒绝, 不会误选 Mesh
    auto topo = MakeClosTopo();
    std::string name;
    auto int64Param = MakeClosParam(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CLOS_SMALL_BYTES, HCCL_DATA_TYPE_INT64);
    EXPECT_EQ(SelectCcuSchedule(int64Param, topo, name), SelectorStatus::NOT_MATCH);

    auto prodParam = MakeClosParam(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CLOS_SMALL_BYTES);
    prodParam.reduceType = HCCL_REDUCE_PROD;
    EXPECT_EQ(SelectCcuSchedule(prodParam, topo, name), SelectorStatus::NOT_MATCH);
}
} // namespace

// 以下符号仅在 auto selector 的 DPU/AIV 分支引用, CLOS CCU_SCHED 路径不会执行到;
// 本 UT 不链接对应实现, 提供桩以满足链接。
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
int GetHcommVersion(void) { return CANN_VERSION(9, 2, 0, 2); }

HcclResult HcclGetHcclBuffer(HcclComm, void**, uint64_t*)
{
    ADD_FAILURE() << "CLOS CCU_SCHED selector UT must not enter AIV resource preparation";
    return HCCL_E_NOT_SUPPORT;
}
}
