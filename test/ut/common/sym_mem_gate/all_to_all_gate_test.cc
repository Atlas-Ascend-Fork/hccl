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

class DayuAllToAllGateTest : public testing::Test {
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
        topo_.level0Topo = Level0Shape::CLOS;
        topo_.topoLevelNums = TOPO_LEVEL_NUM_1;
        SetAllToAllOp(HcclCMDType::HCCL_CMD_ALLTOALL, "AicpuAllToAllSoleMesh");
    }

    void SetOp(HcclCMDType opType, const char* algName)
    {
        param_.opType = opType;
        SetAlg(algName);
    }

    void SetAllToAllOp(HcclCMDType opType, const char* algName)
    {
        SetOp(opType, algName);
        param_.all2AllVDataDes.sendType = HcclDataType::HCCL_DATA_TYPE_FP32;
        param_.all2AllVDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_FP32;
        param_.all2AllVDataDes.sendCounts = &utAllToAllCount_;
        param_.all2AllVDataDes.recvCounts = &utAllToAllCount_;
        param_.all2AllVDataDes.sdispls = &utAllToAllDispl_;
        param_.all2AllVDataDes.rdispls = &utAllToAllDispl_;
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
    u64 utAllToAllCount_ = 0;
    u64 utAllToAllDispl_ = 0;
};

TEST_F(DayuAllToAllGateTest, DayuAllToAllAllowedOnSingleLevelClos)
{
    SetDayuMainboard();
    ExpectAllowed(true);
    EXPECT_EQ(g_utSymWinGetCalls, 2U);
    EXPECT_NE(param_.inputSymWindow, nullptr);
    EXPECT_NE(param_.outputSymWindow, nullptr);
}

TEST_F(DayuAllToAllGateTest, DayuAllToAllSmallDataNotBlockedByUbxMultiJettyThreshold)
{
    topo_.level0Topo = Level0Shape::MESH_1D_CLOS;
    topo_.userRankSize = 8;
    SetDayuMainboard();
    SetAllToAllOp(HcclCMDType::HCCL_CMD_ALLTOALL, "AicpuAllToAllSoleMeshMultiJetty");
    utAllToAllCount_ = 1024;
    param_.inputSize = 4096;
    param_.outputSize = 4096;
    ExpectAllowed(true);
}

TEST_F(DayuAllToAllGateTest, DayuAllToAllRejectedOnMultiLevelClosTopology)
{
    SetDayuMainboard();
    topo_.topoLevelNums = 2;
    ExpectAllowed(false);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllToAllGateTest, DayuAllToAllRejectedOnOtherMainboard)
{
    SetDayuMainboard(UT_OTHER_MAINBOARD_ID);
    ExpectAllowed(false);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllToAllGateTest, DayuAllToAllPairwiseAllowedByAllToAllBranch)
{
    // ALLTOALL 门禁分支不校验 algName，pairwise 与 SoleMesh 同口径放行；V/VC 算法名仅注册在 ALLTOALLV/VC 下，不在此覆盖
    SetDayuMainboard();
    SetAllToAllOp(HcclCMDType::HCCL_CMD_ALLTOALL, "AicpuAllToAllSolePairwise");
    ExpectAllowed(true);
    EXPECT_EQ(g_utSymWinGetCalls, 2U);
}

TEST_F(DayuAllToAllGateTest, DayuAllToAllVRejectedByDefaultBranch)
{
    SetDayuMainboard();
    SetAllToAllOp(HcclCMDType::HCCL_CMD_ALLTOALLV, "AicpuAllToAllVSoleMesh");
    ExpectAllowed(false);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllToAllGateTest, DayuAllToAllVCRejectedByOpWhitelist)
{
    SetDayuMainboard();
    SetAllToAllOp(HcclCMDType::HCCL_CMD_ALLTOALLVC, "AicpuAllToAllVCSoleMesh");
    ExpectAllowed(false);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllToAllGateTest, UbxAllToAllSoleMeshNotWhitelistedRejected)
{
    topo_.level0Topo = Level0Shape::MESH_1D_CLOS;
    ExpectAllowed(false);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(DayuAllToAllGateTest, DayuAllToAllSymMemCheckedRestoresWithoutProbe)
{
    SetDayuMainboard();
    PreprobeAsEntry();
    ExpectAllowed(true);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}
} // namespace
