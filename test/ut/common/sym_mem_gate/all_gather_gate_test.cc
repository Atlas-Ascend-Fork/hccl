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

#include "hcomm_stub.h"
#include "op_common.h"
#include "securec.h"

using namespace ops_hccl;
using namespace ut_stub;

namespace {
constexpr s64 UT_DAYU_MAINBOARD_ID = 0x48;
constexpr s64 UT_OTHER_MAINBOARD_ID = 0x47;

class DayuAllGatherGateTest : public testing::Test {
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
        topo_.level0Topo = Level0Shape::CLOS;
        topo_.topoLevelNums = TOPO_LEVEL_NUM_1;
        SetOp(HcclCMDType::HCCL_CMD_ALLGATHER, "AicpuAllGatherSoleNHRMultiLink");
    }

    void SetOp(HcclCMDType opType, const char* algName)
    {
        param_.opType = opType;
        SetAlg(algName);
    }

    void SetAlg(const char* algName) { ASSERT_EQ(strcpy_s(param_.algName, sizeof(param_.algName), algName), EOK); }

    void SetDayuMainboard(s64 mainboardId = UT_DAYU_MAINBOARD_ID)
    {
        g_utMainboardInfoOk = true;
        g_utMainboardId = mainboardId;
    }

    void PreprobeAsEntry()
    {
        param_.symMemChecked = true;
        param_.inputSymWindow = reinterpret_cast<HcclCommSymWindow>(0x1111U);
        param_.outputSymWindow = reinterpret_cast<HcclCommSymWindow>(0x2222U);
    }

    void ExpectAllowed(bool expected)
    {
        EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
        EXPECT_EQ(param_.supportSymmetricMemory, expected);
    }

    OpParam param_;
    TopoInfoWithNetLayerDetails topo_;
};

TEST_F(DayuAllGatherGateTest, DayuAllGatherAllowedOnSingleLevelClos)
{
    SetDayuMainboard();
    ExpectAllowed(true);
    EXPECT_EQ(g_utSymWinGetCalls, 2U);
    EXPECT_NE(param_.inputSymWindow, nullptr);
    EXPECT_NE(param_.outputSymWindow, nullptr);
}

TEST_F(DayuAllGatherGateTest, DayuAllGatherAllowedOnUbxShapeOutsideUbxWhitelist)
{
    topo_.level0Topo = Level0Shape::MESH_1D_CLOS;
    SetDayuMainboard();
    ExpectAllowed(true);
    EXPECT_EQ(g_utSymWinGetCalls, 2U);
}

TEST_F(DayuAllGatherGateTest, DayuAllGatherAllowedForInt64AndProd)
{
    SetDayuMainboard();
    param_.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT64;
    param_.reduceType = HcclReduceOp::HCCL_REDUCE_PROD;
    ExpectAllowed(true);
}

TEST_F(DayuAllGatherGateTest, DayuAllGatherRejectedOnMultiLevelClosTopology)
{
    SetDayuMainboard();
    topo_.topoLevelNums = 2;
    ExpectAllowed(false);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllGatherGateTest, DayuAllGatherRejectedOnOtherMainboard)
{
    SetDayuMainboard(UT_OTHER_MAINBOARD_ID);
    ExpectAllowed(false);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllGatherGateTest, DayuAllGatherRejectedWhenMainboardUnavailable)
{
    ExpectAllowed(false);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllGatherGateTest, DayuAllGatherRejectedOnDeviceQueryFailure)
{
    SetDayuMainboard();
    g_utDeviceQueryOk = false;
    ExpectAllowed(false);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllGatherGateTest, DayuAllGatherRejectedOnAivEngine)
{
    SetDayuMainboard();
    param_.engine = CommEngine::COMM_ENGINE_AIV;
    ExpectAllowed(false);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllGatherGateTest, DayuAllGatherRejectedInGraphMode)
{
    SetDayuMainboard();
    param_.opMode = OpMode::OFFLOAD;
    ExpectAllowed(false);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllGatherGateTest, DayuAllGatherRejectedOnOldHcommVersion)
{
    SetDayuMainboard();
    g_utHcommVersion = CANN_VERSION(9, 0, 0);
    ExpectAllowed(false);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllGatherGateTest, DayuAllGatherVRejectedByOpWhitelist)
{
    SetDayuMainboard();
    SetOp(HcclCMDType::HCCL_CMD_ALLGATHER_V, "AicpuAllGatherVSoleMesh");
    ExpectAllowed(false);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllGatherGateTest, UbxAllGatherNonWhitelistAlgStillRejected)
{
    topo_.level0Topo = Level0Shape::MESH_1D_CLOS;
    ExpectAllowed(false);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllGatherGateTest, DayuAllGatherSymMemCheckedRestoresWithoutProbe)
{
    SetDayuMainboard();
    PreprobeAsEntry();
    ExpectAllowed(true);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllGatherGateTest, DayuAllGatherSymMemCheckedAllowsInplaceOverlap)
{
    // 主干 !4159 起 inplace 公共门禁只约束 AIV，AICPU_TS 的 inplace 不再由公共层拦截
    SetDayuMainboard();
    PreprobeAsEntry();
    param_.inputPtr = param_.outputPtr;
    param_.inputSize = 4096;
    param_.outputSize = 512;
    ExpectAllowed(true);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllGatherGateTest, DayuAllGatherSymMemCheckedClearedWhenGateFails)
{
    SetDayuMainboard();
    PreprobeAsEntry();
    topo_.topoLevelNums = 2;
    ExpectAllowed(false);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}
} // namespace
