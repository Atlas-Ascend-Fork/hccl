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
//   5. seq分桶：集合类按(comm,opType)、send/recv按(发送方,接收方)——两侧视角同一pair键，
//      多进程部署下两侧进程独立计数、同一次通信同号
// 说明：g_aclgraphTagCtxMap为进程级状态（号永不归零），各用例使用独立commName隔离计数桶。

#include <gtest/gtest.h>

#include <string>

#include "hcomm_stub.h"
#include "op_common.h"
#include "securec.h"

using namespace ops_hccl;
using namespace ut_stub;

namespace {

class AclgraphGateTest : public testing::Test {
protected:
    void SetUp() override
    {
        UtCaptureStubReset();
        param_ = OpParam{};
        topo_ = TopoInfoWithNetLayerDetails{};
        // 默认满足gate全部条件：捕获态+OPBASE+AICPU_TS+白名单算子+非协商域+非对称
        param_.opMode = OpMode::OPBASE;
        param_.engine = CommEngine::COMM_ENGINE_AICPU_TS;
        param_.stream = reinterpret_cast<aclrtStream>(0x1);
        param_.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
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

// ---------------- 5. seq分桶（跨用例状态用独立commName隔离） ----------------

TEST_F(AclgraphGateTest, SeqIncrementsWithinSameOpTypeBucket)
{
    SetCommName("ut_seq_same_bucket");
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_1") != nullptr);
    ASSERT_EQ(strcpy_s(param_.tag, sizeof(param_.tag), "AllGather_ut_comm"), EOK);
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_2") != nullptr);
}

TEST_F(AclgraphGateTest, SeqIndependentAcrossOpTypeAndCommBuckets)
{
    // 同comm不同opType：各自独立计数
    SetCommName("ut_seq_buckets");
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_1") != nullptr);
    param_.opType = HcclCMDType::HCCL_CMD_BROADCAST;
    ASSERT_EQ(strcpy_s(param_.tag, sizeof(param_.tag), "Broadcast_ut_comm"), EOK);
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_1") != nullptr);
    // 不同comm：独立计数
    SetCommName("ut_seq_other_comm");
    ASSERT_EQ(strcpy_s(param_.tag, sizeof(param_.tag), "AllGather_ut_comm"), EOK);
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_1") != nullptr);
}

TEST_F(AclgraphGateTest, PairBucketSharedBySendRecvOfSamePair)
{
    // 匹配流原则核心机制：send(0→1)与对端recv(1←0)两侧视角生成同一pair键("P2P_0_1")，落入同一pair桶。
    // 多进程部署（每rank独立进程、各进程计数互不可见）下两侧各自独立计数，同一次通信的send与recv
    // 取到同号，memTag一致完成注册/反查配对；本UT单进程模拟两侧调用、共享同一进程级pair桶，表现为
    // 续号（send取1、recv续2）——续号即"同桶"证据：若pair键误按方向分桶，recv将从1独立起号。
    SetCommName("ut_pair");
    param_.opType = HcclCMDType::HCCL_CMD_SEND;
    param_.sendRecvRemoteRank = 1;
    topo_.userRank = 0;
    ASSERT_EQ(strcpy_s(param_.tag, sizeof(param_.tag), "SendRecv_ut_comm_0_1"), EOK);
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_1") != nullptr);

    // 对端recv视角：userRank=1、对端=0 → 同一pair键(0,1) → seq续号为2
    param_ = OpParam{};
    param_.opMode = OpMode::OPBASE;
    param_.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    param_.opType = HcclCMDType::HCCL_CMD_RECEIVE;
    param_.sendRecvRemoteRank = 0;
    topo_.userRank = 1;
    ASSERT_EQ(strcpy_s(param_.commName, sizeof(param_.commName), "ut_pair"), EOK);
    ASSERT_EQ(strcpy_s(param_.tag, sizeof(param_.tag), "SendRecv_ut_comm_0_1"), EOK);
    ASSERT_EQ(TryEnableAclgraphZeroCopy(param_, &topo_, std::string("AicpuTestAlg")), HCCL_SUCCESS);
    EXPECT_TRUE(strstr(param_.tag, "_aclgraph_2") != nullptr);
}

TEST_F(AclgraphGateTest, PairBucketIndependentPerPair)
{
    SetCommName("ut_pair_independent");
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

} // namespace
