/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 大禹POD 机型放行与算子白名单收口 UT（MR !4158）
// 被测对象：RefreshSymmetricMemory / CheckDayuMainboard / IsDayuSymMemAllowedOp（op_common.cc）
//           IsReduceOpSymMemDeterministic 的 MultiLink 白名单（sym_mem_gate.cc）
// 覆盖点：
//   1. 大禹机型 + AICPU_TS + 单级拓扑放行 ReduceScatter（SoleNHRMultiLink）
//   2. 大禹机型门禁否决：多级拓扑、非大禹主板、主板查询失败、device 查询失败、AIV 引擎
//   3. ReduceScatter 参数门禁：in-place、INT64、PROD、非白名单算法
//   4. 算子白名单收口：ReduceScatter 放行、ReduceScatterV 拒（与 RS 同 case 分支但不进白名单）
//   5. MultiLink 算法仅单级放行：多级被级数检查拦截（与 topoLevelNums == 1 条件对齐）

#include <gtest/gtest.h>

#include "hcomm_stub.h"
#include "op_common.h"
#include "securec.h"

using namespace ops_hccl;
using namespace ut_stub;

namespace {
constexpr s64 UT_DAYU_MAINBOARD_ID = 0x48; // 与 op_common.cc DAYU_MAINBOARD_ID 一致
constexpr s64 UT_OTHER_MAINBOARD_ID = 0x47;

class DayuSymMemGateTest : public testing::Test {
protected:
    void SetUp() override
    {
        UtSymWinStubReset();
        UtDeviceStubReset();
        param_ = OpParam{};
        topo_ = TopoInfoWithNetLayerDetails{};
        param_.opMode = OpMode::OPBASE;
        param_.engine = CommEngine::COMM_ENGINE_AICPU_TS;
        param_.hcclComm = reinterpret_cast<void*>(0x1);
        param_.inputPtr = reinterpret_cast<void*>(0x1000);
        param_.inputSize = 4096;
        param_.outputPtr = reinterpret_cast<void*>(0x2000);
        param_.outputSize = 512;
        param_.reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
        param_.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
        param_.vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
        // 大禹 CLOS 单级组网：server（MESH_1D）与 UBX（MESH_1D_CLOS）分支均不成立，
        // 隔离机型条件单独对放行结果的影响
        topo_.level0Topo = Level0Shape::CLOS;
        topo_.topoLevelNums = TOPO_LEVEL_NUM_1;
        param_.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
        SetAlg("AicpuReduceScatterSoleNHRMultiLink");
    }

    void SetAlg(const char* algName) { ASSERT_EQ(strcpy_s(param_.algName, sizeof(param_.algName), algName), EOK); }

    void SetDayuMainboard(s64 mainboardId = UT_DAYU_MAINBOARD_ID)
    {
        g_utMainboardInfoOk = true;
        g_utMainboardId = mainboardId;
    }

    OpParam param_;
    TopoInfoWithNetLayerDetails topo_;
};

// ---------------- 大禹机型放行 ReduceScatter ----------------

TEST_F(DayuSymMemGateTest, DayuReduceScatterMultiLinkAllowed)
{
    // 大禹 + AICPU_TS + 单级 + SoleNHRMultiLink + 非 in-place/非64位/非PROD → 放行，双侧探测
    SetDayuMainboard();
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_TRUE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 2U);
    EXPECT_NE(param_.inputSymWindow, nullptr);
    EXPECT_NE(param_.outputSymWindow, nullptr);
}

TEST_F(DayuSymMemGateTest, DayuReduceScatterRejectedOnMultiLevelTopology)
{
    // topoLevelNums > 1：存在跨机链路（ROCE），机型条件不成立 → 不放行、不探测
    SetDayuMainboard();
    topo_.topoLevelNums = 2;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuSymMemGateTest, DayuReduceScatterRejectedOnOtherMainboard)
{
    // 主板 ID 严格相等：非 0x48 一律判非大禹
    SetDayuMainboard(UT_OTHER_MAINBOARD_ID);
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuSymMemGateTest, DayuReduceScatterRejectedWhenMainboardUnavailable)
{
    // 取不到主板 ID（默认桩行为）：按非大禹处理，不放行
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuSymMemGateTest, DayuReduceScatterRejectedOnDeviceQueryFailure)
{
    // aclrtGetDevice 失败：无法定位机型，按非大禹处理
    SetDayuMainboard();
    g_utDeviceQueryOk = false;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuSymMemGateTest, DayuReduceScatterRejectedOnAivEngine)
{
    // 非 UBX + AIV 引擎在基础门禁直接返回，机型条件不参与
    SetDayuMainboard();
    param_.engine = CommEngine::COMM_ENGINE_AIV;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

// ---------------- ReduceScatter 参数门禁（大禹机型下同样生效） ----------------

TEST_F(DayuSymMemGateTest, DayuReduceScatterRejectedOnInplace)
{
    SetDayuMainboard();
    param_.inputPtr = param_.outputPtr;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
}

TEST_F(DayuSymMemGateTest, DayuReduceScatterRejectedOnInt64)
{
    // RS 走 DataDes.dataType：64 位整型被 IsReduceOpSymMemAllowed 拦截
    SetDayuMainboard();
    param_.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT64;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
}

TEST_F(DayuSymMemGateTest, DayuReduceScatterRejectedOnProd)
{
    SetDayuMainboard();
    param_.reduceType = HcclReduceOp::HCCL_REDUCE_PROD;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
}

TEST_F(DayuSymMemGateTest, DayuReduceScatterRejectedOnUnknownAlg)
{
    SetDayuMainboard();
    SetAlg("AicpuReduceScatterUnknown");
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
}

TEST_F(DayuSymMemGateTest, DayuReduceScatterRejectedOnStrictOrderedMeshAlg)
{
    // StrictOrderedMesh 不在确定性白名单：大禹单级组网下选到也不能开对称内存
    SetDayuMainboard();
    SetAlg("AicpuReduceScatterStrictOrderedMesh");
    EXPECT_FALSE(IsReduceOpSymMemDeterministic(
        "AicpuReduceScatterStrictOrderedMesh", &topo_, HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuSymMemGateTest, DayuReduceScatterRejectedOnSoleNhrAicpuReduceAlg)
{
    // SoleNHRAicpuReduce 与 SoleNHR 前缀相同但不是同一算法：白名单为精确匹配，不能误放行
    SetDayuMainboard();
    SetAlg("AicpuReduceScatterSoleNHRAicpuReduce");
    EXPECT_FALSE(IsReduceOpSymMemDeterministic(
        "AicpuReduceScatterSoleNHRAicpuReduce", &topo_, HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

// ---------------- 算子白名单收口（IsDayuSymMemAllowedOp） ----------------

TEST_F(DayuSymMemGateTest, DayuReduceScatterVRejectedByOpWhitelist)
{
    // RSV 不在大禹白名单：switch 内条件全部满足也会被尾部收口剔除
    SetDayuMainboard();
    param_.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
    SetAlg("AicpuReduceScatterVSoleMeshChunk");
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

// ---------------- MultiLink 算法白名单的级数边界 ----------------

TEST_F(DayuSymMemGateTest, MultiLinkAlgOnlyAllowedOnSingleLevel)
{
    // MultiLink 位于单级白名单尾部（SoleNHR 豁免级数不覆盖它）：单级放行、多级拒绝
    EXPECT_TRUE(IsReduceOpSymMemDeterministic(
        "AicpuReduceScatterSoleNHRMultiLink", &topo_, HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    topo_.topoLevelNums = 2;
    EXPECT_FALSE(IsReduceOpSymMemDeterministic(
        "AicpuReduceScatterSoleNHRMultiLink", &topo_, HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
}
} // namespace
