/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// aclgraph零拷贝gate UT（PR检视要求：TryEnableAclgraphZeroCopy三分支 + gate条件组合回归兜底）
// 被测对象（op_common.cc / stream_capture.cc，直编生产源文件 + capture/hcomm桩）：
//   1. IsAclgraphZeroCopySupportedOp 算子白名单矩阵（放行6算子；allreduce/gather/scatter/V类拒绝）
//   2. IsAclgraphZeroCopyGateHit 条件组合（捕获态/OPBASE/AICPU_TS/白名单/非协商域/非对称，逐项负例）
//   3. TryEnableAclgraphZeroCopy tag追加分支：tag带_aclgraph_<seq>、opMode=OFFLOAD、algTag按memTag重拼
//   4. TryEnableAclgraphZeroCopy 无空间回退分支：tag/opMode/algTag零改写、返回SUCCESS（优雅降级）
//   4b. TryEnableAclgraphZeroCopy 取号失败分支（ctx创建失败/条目尺寸异常）：错误码
//      透传整算子报错、tag/opMode/algTag零改写（seq为跨rank契约，禁止本rank静默回退）
//   5. seq分桶：计数簿记挂comm的engine ctx，桶随comm隔离——集合类按(comm,opType)、send/recv按
//      (comm,发送方,接收方)。单进程多rank（HcclCommInitAll每卡一comm）下匹配的send/recv各自独立
//      计数取同号；多进程部署每进程一comm同理由同号
// 说明：seq簿记随engine ctx桩存取（UtEngineCtxStubReset每用例复位，seq从1起），用例以不同
// comm句柄（UT_COMM_A/B）模拟单进程多rank下的桶隔离。

#include <gtest/gtest.h>

#include <string>

#include "hcomm_stub.h"
#include "op_common.h"
#include "securec.h"

using namespace ops_hccl;
using namespace ut_stub;

namespace {

// 模拟单进程多 rank 场景下两个 rank 各自的通信域句柄
const HcclComm UT_COMM_A = reinterpret_cast<HcclComm>(0x10);
const HcclComm UT_COMM_B = reinterpret_cast<HcclComm>(0x20);

class AclgraphGateTest : public testing::Test {
protected:
    void SetUp() override
    {
        UtCaptureStubReset();
        UtEngineCtxStubReset();
        param_ = OpParam{};
        topo_ = TopoInfoWithNetLayerDetails{};
        // 默认满足gate全部条件：捕获态+OPBASE+AICPU_TS+白名单算子+非协商域+非对称
        param_.opMode = OpMode::OPBASE;
        param_.engine = CommEngine::COMM_ENGINE_AICPU_TS;
        param_.stream = reinterpret_cast<aclrtStream>(0x1);
        param_.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
        param_.hcclComm = UT_COMM_A;
        ASSERT_EQ(strcpy_s(param_.tag, sizeof(param_.tag), "AllGather_ut_comm"), EOK);
        ASSERT_EQ(strcpy_s(param_.commName, sizeof(param_.commName), "ut_comm"), EOK);
        topo_.userRank = 0;
    }

    void SetCommName(const std::string& name)
    {
        ASSERT_EQ(strcpy_s(param_.commName, sizeof(param_.commName), name.c_str()), EOK);
    }

    OpParam param_;
    TopoInfoWithNetLayerDetails topo_;
};

// ---------------- 1. 算子白名单矩阵 ----------------

TEST_F(AclgraphGateTest, WhitelistAllowsSixSupportedOps)
{
    EXPECT_TRUE(IsAclgraphZeroCopySupportedOp(HcclCMDType::HCCL_CMD_ALLGATHER));
    EXPECT_TRUE(IsAclgraphZeroCopySupportedOp(HcclCMDType::HCCL_CMD_ALLGATHER_V));
    EXPECT_TRUE(IsAclgraphZeroCopySupportedOp(HcclCMDType::HCCL_CMD_BROADCAST));
    EXPECT_TRUE(IsAclgraphZeroCopySupportedOp(HcclCMDType::HCCL_CMD_SEND));
    EXPECT_TRUE(IsAclgraphZeroCopySupportedOp(HcclCMDType::HCCL_CMD_RECEIVE));
    EXPECT_TRUE(IsAclgraphZeroCopySupportedOp(HcclCMDType::HCCL_CMD_ALLTOALL));
}

TEST_F(AclgraphGateTest, WhitelistRejectsUnsupportedOps)
{
    // allreduce/reduce_scatter暂未接入；V类（ALLTOALLV/VC）需对端rdispls交换不接；scatter/reduce无键控路径
    EXPECT_FALSE(IsAclgraphZeroCopySupportedOp(HcclCMDType::HCCL_CMD_ALLREDUCE));
    EXPECT_FALSE(IsAclgraphZeroCopySupportedOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    EXPECT_FALSE(IsAclgraphZeroCopySupportedOp(HcclCMDType::HCCL_CMD_ALLTOALLV));
    EXPECT_FALSE(IsAclgraphZeroCopySupportedOp(HcclCMDType::HCCL_CMD_ALLTOALLVC));
    EXPECT_FALSE(IsAclgraphZeroCopySupportedOp(HcclCMDType::HCCL_CMD_SCATTER));
    EXPECT_FALSE(IsAclgraphZeroCopySupportedOp(HcclCMDType::HCCL_CMD_GATHER));
    EXPECT_FALSE(IsAclgraphZeroCopySupportedOp(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV));
    EXPECT_FALSE(IsAclgraphZeroCopySupportedOp(HcclCMDType::HCCL_CMD_INVALID));
}

// ---------------- 2. gate条件组合 ----------------

TEST_F(AclgraphGateTest, GateHitWhenAllConditionsMet) { EXPECT_TRUE(IsAclgraphZeroCopyGateHit(param_)); }

TEST_F(AclgraphGateTest, GateMissWhenStreamNotCapturing)
{
    g_utCaptureStatus = 0; // ACL_MODEL_RI_CAPTURE_STATUS_NONE
    EXPECT_FALSE(IsAclgraphZeroCopyGateHit(param_));
}

TEST_F(AclgraphGateTest, GateMissWhenCaptureQueryFails)
{
    // 查询失败按非捕获态处理（IsStreamInCaptureMode失败降级），gate不命中
    g_utCaptureRet = 1;
    EXPECT_FALSE(IsAclgraphZeroCopyGateHit(param_));
}

TEST_F(AclgraphGateTest, GateMissWhenNullStream)
{
    param_.stream = nullptr;
    EXPECT_FALSE(IsAclgraphZeroCopyGateHit(param_));
}

TEST_F(AclgraphGateTest, GateMissWhenAlreadyOffload)
{
    param_.opMode = OpMode::OFFLOAD; // GE图模式入口已是OFFLOAD，gate跳过
    EXPECT_FALSE(IsAclgraphZeroCopyGateHit(param_));
}

TEST_F(AclgraphGateTest, GateMissWhenEngineNotAicpuTs)
{
    param_.engine = CommEngine::COMM_ENGINE_CCU;
    EXPECT_FALSE(IsAclgraphZeroCopyGateHit(param_));
    param_.engine = CommEngine::COMM_ENGINE_AIV;
    EXPECT_FALSE(IsAclgraphZeroCopyGateHit(param_));
    param_.engine = CommEngine::COMM_ENGINE_CPU;
    EXPECT_FALSE(IsAclgraphZeroCopyGateHit(param_));
}

TEST_F(AclgraphGateTest, GateMissWhenOpNotWhitelisted)
{
    param_.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    EXPECT_FALSE(IsAclgraphZeroCopyGateHit(param_));
}

TEST_F(AclgraphGateTest, GateMissWhenNegotiationComm)
{
    SetCommName("ut_comm_negotiation");
    EXPECT_FALSE(IsAclgraphZeroCopyGateHit(param_));
}

TEST_F(AclgraphGateTest, GateMissWhenSymmetricMemoryEnabled)
{
    // 对称内存优先：走对称零拷贝通道，gate不命中（新设计语义）
    param_.supportSymmetricMemory = true;
    EXPECT_FALSE(IsAclgraphZeroCopyGateHit(param_));
}

// ---------------- 3. TryEnable：tag追加分支 ----------------

TEST_F(AclgraphGateTest, TryEnableAppendsSuffixAndSetsOffloadAndAlgTag)
{
    SetCommName("ut_tag_rewrite");
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    // tag追加"_aclgraph_<seq>"（独立commName，seq从1起）
    EXPECT_STREQ(param_.tag, "AllGather_ut_comm_aclgraph_1");
    // opMode翻OFFLOAD（复用OFFLOAD路径的信号）
    EXPECT_EQ(param_.opMode, OpMode::OFFLOAD);
    // algTag按memTag重拼：AICPU_TS引擎launchMode为"device"
    EXPECT_STREQ(param_.algTag, "AllGather_ut_comm_aclgraph_1_AicpuTestAlg_device");
}

TEST_F(AclgraphGateTest, TryEnableRejectsNullTopo)
{
    SetCommName("ut_null_topo");
    EXPECT_EQ(TryEnableAclgraphZeroCopy(param_, nullptr, std::string("AicpuTestAlg")), HCCL_E_PTR);
}

// ---------------- 4. TryEnable：无空间回退分支 ----------------

TEST_F(AclgraphGateTest, TryEnableNoRoomFallsBackWithoutAnyMutation)
{
    SetCommName("ut_no_room");
    // tag填至TAG_LENGTH-1（追加"_aclgraph_N"必然越界）
    std::string fullTag(TAG_LENGTH - 1, 'a');
    ASSERT_EQ(strcpy_s(param_.tag, sizeof(param_.tag), fullTag.c_str()), EOK);
    ASSERT_EQ(strcpy_s(param_.algTag, sizeof(param_.algTag), "alg_sentinel"), EOK);
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    // 整体回退：tag/opMode/algTag零改写
    EXPECT_EQ(strnlen(param_.tag, TAG_LENGTH), TAG_LENGTH - 1);
    EXPECT_EQ(param_.tag[0], 'a');
    EXPECT_EQ(param_.opMode, OpMode::OPBASE);
    EXPECT_STREQ(param_.algTag, "alg_sentinel");
}

// ---------------- 4b. TryEnable：取号失败硬报错分支 ----------------

TEST_F(AclgraphGateTest, TryEnableHardFailsWhenAllocSeqFails)
{
    SetCommName("ut_alloc_fail");
    ASSERT_EQ(strcpy_s(param_.algTag, sizeof(param_.algTag), "alg_sentinel"), EOK);
    // 本rank ctx创建失败：错误码透传整算子报错（静默回退会导致同桶跨rank永久错位）
    ut_stub::g_utEngineCtxCreateFail = true;
    EXPECT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_E_INTERNAL);
    // 硬报错路径同样零改写tag/opMode/algTag
    EXPECT_STREQ(param_.tag, "AllGather_ut_comm");
    EXPECT_EQ(param_.opMode, OpMode::OPBASE);
    EXPECT_STREQ(param_.algTag, "alg_sentinel");
}

// ---------------- 5. seq分桶（簿记随comm的engine ctx隔离，每用例桩复位） ----------------

TEST_F(AclgraphGateTest, SeqIncrementsWithinSameOpTypeBucket)
{
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_1") != nullptr);
    ASSERT_EQ(strcpy_s(param_.tag, sizeof(param_.tag), "AllGather_ut_comm"), EOK);
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_2") != nullptr);
}

TEST_F(AclgraphGateTest, SeqIndependentAcrossOpTypeAndCommBuckets)
{
    // 同comm不同opType：各自独立计数
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_1") != nullptr);
    param_.opType = HcclCMDType::HCCL_CMD_BROADCAST;
    ASSERT_EQ(strcpy_s(param_.tag, sizeof(param_.tag), "Broadcast_ut_comm"), EOK);
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_1") != nullptr);
    // 不同comm：桶随comm隔离，独立计数
    param_.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    param_.hcclComm = UT_COMM_B;
    ASSERT_EQ(strcpy_s(param_.tag, sizeof(param_.tag), "AllGather_ut_comm"), EOK);
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_1") != nullptr);
}

TEST_F(AclgraphGateTest, PairSeqSameAcrossPairedCommRanks)
{
    // 单进程多rank（HcclCommInitAll每卡一comm）下的匹配流配对：rank0的send(comm_A)与rank1的
    // recv(comm_B)生成同一pair键("P2P_0_1")，计数桶随comm隔离后两侧各自独立计数，
    // 同一次配对通信取到同号，memTag一致完成注册/反查配对
    param_.opType = HcclCMDType::HCCL_CMD_SEND;
    param_.sendRecvRemoteRank = 1;
    topo_.userRank = 0;
    ASSERT_EQ(strcpy_s(param_.tag, sizeof(param_.tag), "SendRecv_ut_comm_0_1"), EOK);
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_1") != nullptr);

    // 对端recv视角：comm_B、userRank=1、对端=0 → 独立桶计数，与send侧同号
    param_ = OpParam{};
    param_.opMode = OpMode::OPBASE;
    param_.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    param_.hcclComm = UT_COMM_B;
    param_.opType = HcclCMDType::HCCL_CMD_RECEIVE;
    param_.sendRecvRemoteRank = 0;
    topo_.userRank = 1;
    ASSERT_EQ(strcpy_s(param_.commName, sizeof(param_.commName), "ut_comm"), EOK);
    ASSERT_EQ(strcpy_s(param_.tag, sizeof(param_.tag), "SendRecv_ut_comm_0_1"), EOK);
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_1") != nullptr);
}

TEST_F(AclgraphGateTest, PairSeqSharedWithinSameComm)
{
    // 同一comm内两侧视角生成同一pair键、共享同一桶，续号（send取1、recv续2）——
    // 续号即"键方向无关"证据：若pair键误按方向分桶，recv将从1独立起号
    param_.opType = HcclCMDType::HCCL_CMD_SEND;
    param_.sendRecvRemoteRank = 1;
    topo_.userRank = 0;
    ASSERT_EQ(strcpy_s(param_.tag, sizeof(param_.tag), "SendRecv_ut_comm_0_1"), EOK);
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_1") != nullptr);

    param_ = OpParam{};
    param_.opMode = OpMode::OPBASE;
    param_.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    param_.hcclComm = UT_COMM_A;
    param_.opType = HcclCMDType::HCCL_CMD_RECEIVE;
    param_.sendRecvRemoteRank = 0;
    topo_.userRank = 1;
    ASSERT_EQ(strcpy_s(param_.commName, sizeof(param_.commName), "ut_comm"), EOK);
    ASSERT_EQ(strcpy_s(param_.tag, sizeof(param_.tag), "SendRecv_ut_comm_0_1"), EOK);
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_2") != nullptr);
}

TEST_F(AclgraphGateTest, PairBucketIndependentPerPair)
{
    param_.opType = HcclCMDType::HCCL_CMD_SEND;
    param_.sendRecvRemoteRank = 1;
    topo_.userRank = 0;
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_1") != nullptr);
    // 不同pair(0→2)：独立桶，seq重新从1起
    param_.sendRecvRemoteRank = 2;
    ASSERT_EQ(strcpy_s(param_.tag, sizeof(param_.tag), "SendRecv_ut_comm_0_2"), EOK);
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_1") != nullptr);
}

TEST_F(AclgraphGateTest, SeqCtxIsolatedFromCpuTsTagSpace)
{
    // CPU_TS域存在同字符串大条目（topo缓存/algTag键空间含用户tag，用户可构造同串）：
    // 簿记在CPU域，(ctxTag, engine)二元组不同，不受影响正常取号
    std::string collTag = "__aclgraph_seq_coll_" + std::to_string(static_cast<u32>(param_.opType));
    UtEngineCtxPreset(UT_COMM_A, collTag, CommEngine::COMM_ENGINE_CPU_TS, 4096);
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_1") != nullptr);
}

TEST_F(AclgraphGateTest, SeqSizeMismatchHardFailsWithoutMutation)
{
    // CPU域同键非4字节条目（异常条目）：同为rank局部失败源，整算子硬报错透传
    ASSERT_EQ(strcpy_s(param_.algTag, sizeof(param_.algTag), "alg_sentinel"), EOK);
    std::string collTag = "__aclgraph_seq_coll_" + std::to_string(static_cast<u32>(param_.opType));
    UtEngineCtxPreset(UT_COMM_A, collTag, CommEngine::COMM_ENGINE_CPU, 8);
    EXPECT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_E_INTERNAL);
    EXPECT_STREQ(param_.tag, "AllGather_ut_comm");
    EXPECT_TRUE(param_.opMode == OpMode::OPBASE);
    EXPECT_STREQ(param_.algTag, "alg_sentinel");
}

} // namespace
