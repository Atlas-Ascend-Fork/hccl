/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include "op_common.h"

namespace {
using namespace ops_hccl;
constexpr char MULTILINK[] = "AicpuAllReduceSoleNHRMultiLink";
constexpr size_t SIZE = 64;
std::array<char, 3 * SIZE> arena;
int hcommVersion = 0;
unsigned probes = 0;
int32_t currentDevice = 0;
aclError getDeviceResult = ACL_SUCCESS;
int64_t boardId = 0x48;
unsigned boardQueries = 0;
uint32_t queriedDevice = 0;
class MultiLinkRefresh : public testing::Test {
protected:
    void SetUp() override
    {
        hcommVersion = CANN_VERSION(9, 1, 0);
        probes = 0;
        currentDevice = 0;
        getDeviceResult = ACL_SUCCESS;
        boardId = 0x48;
        boardQueries = 0;
        queriedDevice = 0;
        topo.level0Topo = Level0Shape::CLOS;
        topo.topoLevelNums = 1;
        param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
        param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
        param.opMode = OpMode::OPBASE;
        param.DataDes.dataType = HCCL_DATA_TYPE_FP32;
        param.reduceType = HCCL_REDUCE_SUM;
        std::copy_n(MULTILINK, sizeof(MULTILINK), param.algName);
        param.hcclComm = arena.data();
        param.inputPtr = arena.data() + SIZE;
        param.outputPtr = arena.data() + 2 * SIZE;
        param.inputSize = SIZE;
        param.outputSize = SIZE;
    }

    void Preprobe()
    {
        CheckAndSetSymmetricMemory(param);
        ASSERT_TRUE(param.supportSymmetricMemory);
        ASSERT_EQ(probes, 2U);
    }

    void ExpectFlag(bool expected)
    {
        ASSERT_EQ(RefreshSymmetricMemory(param, &topo), HCCL_SUCCESS);
        EXPECT_EQ(param.supportSymmetricMemory, expected);
    }

    TopoInfoWithNetLayerDetails topo{};
    OpParam param{};
};
} // namespace

extern "C" int GetHcommVersion() { return hcommVersion; }
extern "C" aclError aclrtGetDevice(int32_t* device)
{
    if (getDeviceResult == ACL_SUCCESS) {
        *device = currentDevice;
    }
    return getDeviceResult;
}
extern "C" aclError aclrtGetDeviceInfo(uint32_t device, aclrtDevAttr attr, int64_t* id)
{
    if (attr != ACL_DEV_ATTR_MAINBOARD_ID) {
        return ACL_ERROR_INVALID_PARAM;
    }
    ++boardQueries;
    queriedDevice = device;
    *id = boardId;
    return ACL_SUCCESS;
}

extern "C" HcclResult
HcclCommSymWinGet(HcclComm comm, void* ptr, size_t size, HcclCommSymWindow* window, size_t* offset)
{
    ++probes;
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    const auto base = reinterpret_cast<uintptr_t>(arena.data());
    if (comm != arena.data() || addr < base || addr - base > arena.size() || size > arena.size() - (addr - base)) {
        return HCCL_E_NOT_FOUND;
    }
    *window = arena.data();
    *offset = addr - base;
    return HCCL_SUCCESS;
}

TEST_F(MultiLinkRefresh, FinalMultiLinkRequiresSingleLogicalLevel)
{
    ASSERT_NO_FATAL_FAILURE(Preprobe());
    topo.topoLevelNums = 2;
    ExpectFlag(false);
    topo.topoLevelNums = 1;
    ExpectFlag(true);
    EXPECT_EQ(probes, 2U);
}

TEST_F(MultiLinkRefresh, UnsupportedExecutionClearsPreprobedFlag)
{
    ASSERT_NO_FATAL_FAILURE(Preprobe());
    const auto valid = param;
    for (auto engine : {CommEngine::COMM_ENGINE_CCU, CommEngine::COMM_ENGINE_AIV}) {
        SCOPED_TRACE(static_cast<int>(engine));
        param = valid;
        param.engine = engine;
        ExpectFlag(false);
    }
    param = valid;
    param.opMode = OpMode::OFFLOAD;
    ExpectFlag(false);
    param = valid;
    hcommVersion = CANN_VERSION(9, 0, 0);
    ExpectFlag(false);
    EXPECT_EQ(probes, 2U);
}

TEST_F(MultiLinkRefresh, FinalAlgorithmMustRemainEligibleAfterWindowProbe)
{
    ASSERT_NO_FATAL_FAILURE(Preprobe());
    ExpectFlag(true);
    constexpr char strictAlgorithm[] = "AicpuAllReduceStrictOrderedMesh";
    std::copy_n(strictAlgorithm, sizeof(strictAlgorithm), param.algName);
    ExpectFlag(false);
    EXPECT_EQ(probes, 2U);
}

TEST_F(MultiLinkRefresh, CurrentDeviceQueryFailureClearsFlagBeforeBoardLookup)
{
    ASSERT_NO_FATAL_FAILURE(Preprobe());
    getDeviceResult = ACL_ERROR_INVALID_PARAM;
    ExpectFlag(false);
    EXPECT_EQ(boardQueries, 0U);
    EXPECT_EQ(probes, 2U);
}

TEST_F(MultiLinkRefresh, LooksUpDayuBoardOnTheCurrentDevice)
{
    ASSERT_NO_FATAL_FAILURE(Preprobe());
    currentDevice = 3;
    ExpectFlag(true);
    EXPECT_EQ(boardQueries, 1U);
    EXPECT_EQ(queriedDevice, 3U);
}

TEST_F(MultiLinkRefresh, DayuWhitelistAlsoRestrictsBroadcastOnServerMesh)
{
    topo.level0Topo = Level0Shape::MESH_1D;
    topo.level0PcieMix = false;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;
    constexpr char broadcastAlgorithm[] = "AicpuBroadcastSoleMeshTwoShot";
    std::copy_n(broadcastAlgorithm, sizeof(broadcastAlgorithm), param.algName);
    ASSERT_NO_FATAL_FAILURE(Preprobe());
    boardId = 0x44;
    ExpectFlag(true);
    boardId = 0x48;
    ExpectFlag(false);
    EXPECT_EQ(probes, 2U);
}

TEST_F(MultiLinkRefresh, NonDayuServerMeshRejectsPcieAfterWindowProbe)
{
    boardId = 0x44;
    topo.level0Topo = Level0Shape::MESH_1D;
    ASSERT_NO_FATAL_FAILURE(Preprobe());
    topo.level0PcieMix = false;
    ExpectFlag(true);
    topo.level0PcieMix = true;
    ExpectFlag(false);
    EXPECT_EQ(probes, 2U);
}

TEST_F(MultiLinkRefresh, IneligibleReductionClearsPreprobedFlagEvenWithMultiLinkName)
{
    ASSERT_NO_FATAL_FAILURE(Preprobe());
    const auto valid = param;
    for (auto type : {HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64}) {
        SCOPED_TRACE(static_cast<int>(type));
        param = valid;
        param.DataDes.dataType = type;
        ExpectFlag(false);
    }
    param = valid;
    param.reduceType = HCCL_REDUCE_PROD;
    ExpectFlag(false);
    EXPECT_EQ(probes, 2U);
}

TEST_F(MultiLinkRefresh, UncheckedFinalAicpuEngineProbesBothOperands)
{
    param.engine = CommEngine::COMM_ENGINE_CCU;
    ExpectFlag(false);
    EXPECT_EQ(probes, 0U);
    param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    ExpectFlag(true);
    EXPECT_EQ(probes, 2U);
    EXPECT_EQ(param.inputOffset, SIZE);
    EXPECT_EQ(param.outputOffset, 2 * SIZE);
}

TEST_F(MultiLinkRefresh, AicpuTsOverlapAllowedByUpstreamInplacePolicy)
{
    // 主干 !4159 起 inplace 公共门禁只约束 AIV；AICPU_TS 的重叠输入/输出由算法自行处理
    const auto initial = param;
    for (auto offset : {SIZE - 4, SIZE, SIZE + 4}) {
        SCOPED_TRACE(offset);
        probes = 0;
        param = initial;
        param.outputPtr = arena.data() + offset;
        ExpectFlag(true);
        EXPECT_EQ(probes, 2U);
        param.symMemChecked = true;
        param.inputSymWindow = param.outputSymWindow = arena.data();
        param.supportSymmetricMemory = true;
        ExpectFlag(true);
        EXPECT_EQ(probes, 2U);
    }
}

// Hold the final algorithm fixed. This tests the board branch independently of selector reachability.
TEST_F(MultiLinkRefresh, DayuEligibilityDoesNotRequireTheOldPureClosPredicate)
{
    ASSERT_NO_FATAL_FAILURE(Preprobe());
    topo.level0Topo = Level0Shape::MESH_1D_CLOS;
    ExpectFlag(true);
    EXPECT_EQ(probes, 2U);
}
