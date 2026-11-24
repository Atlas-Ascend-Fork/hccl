/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 溢出守卫回归 UT (检视: 新增防护分支零用例): M1/M3/M4 修复引入的每条守卫路径
// 至少一个触发用例 + 对应正常路径, 与 ConvertVCountsToElementUnit 溢出用例同口径。
// 经公开框架入口 (Execute/直调 public 纯函数) 驱动, 上下文为 gert_context_helper.h POD 构造。
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gert_context_helper.h"
#include "stub/stub_comm_recorder.h"
#include "hcom_alltoallv_op.h"
#include "hcom_all_gather_v_op.h"
#include "hcom_receive_op.h"

// 桩层注入项
extern bool g_stubCtxValRet[8];
extern uint64_t g_stubCtxVal[8];

namespace {
constexpr int kIdxStaticModel = 0;
constexpr int kIdxAddrRefresh = 1;

class GuardTest : public ::testing::Test {
protected:
    void SetUp() override { ResetAll(); }
    void TearDown() override { ResetAll(); }
    static void ResetAll()
    {
        for (size_t i = 0; i < 8; i++) {
            g_stubCtxValRet[i] = false;
            g_stubCtxVal[i] = 0;
        }
        g_stubCtxValRet[kIdxStaticModel] = true;
        g_stubCtxVal[kIdxStaticModel] = 1; // 默认静态
        g_stubCtxValRet[kIdxAddrRefresh] = true;
        g_stubCtxVal[kIdxAddrRefresh] = 0; // 默认不刷新
        for (size_t i = 0; i < 16; i++) {
            g_stubAttrs[i] = nullptr;
        }
        g_stubCclBufSize = 1U << 20;
        g_stubOutputTensors.clear();
        g_stubArCalls.clear();
        g_stubRsCalls.clear();
        g_stubSendCalls.clear();
        g_stubBcastCalls.clear();
        g_stubReduceCalls.clear();
        g_stubRecvCalls.clear();
    }
};

} // namespace

// ---------------------------------------------------------------------------
// CalcVRecvDim (M3): recvDispls + recvCounts u64 域回绕 / int64 值域双守卫
// ---------------------------------------------------------------------------
TEST_F(GuardTest, AtoAllVExecuteRecvEndExceedsInt64Rejected)
{
    uint64_t sendData[8] = {};
    float out[8] = {};
    // u64 域和未回绕但超 int64 值域 → 输出维度越界拒绝
    const uint64_t sendCounts[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    const uint64_t sendDispls[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint64_t recvCounts[8] = {2, 0, 0, 0, 0, 0, 0, 0};
    const uint64_t recvDispls[8] = {static_cast<uint64_t>(INT64_MAX), 0, 0, 0, 0, 0, 0, 0};

    HcomAllToAllVOp op(HcomAllToAllKind::kAllToAllV);
    hcut::CtxBuilder ctx;
    ctx.Input({8}, ge::DT_FLOAT, sendData);
    ctx.Input({8}, ge::DT_INT64, sendCounts);
    ctx.Input({8}, ge::DT_INT64, sendDispls);
    ctx.Input({8}, ge::DT_INT64, recvCounts);
    ctx.Input({8}, ge::DT_INT64, recvDispls);
    ctx.Output({8}, ge::DT_FLOAT, out);
    ctx.Attr(0, "ut_group");

    EXPECT_EQ(ge::GRAPH_FAILED, op.Execute(ctx.Build()));
}

TEST_F(GuardTest, AtoAllVExecuteSmallCountsSucceeds)
{
    uint64_t sendData[8] = {};
    float out[8] = {};
    const uint64_t sendCounts[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    const uint64_t sendDispls[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const uint64_t recvCounts[8] = {2, 1, 1, 1, 1, 1, 1, 1};
    const uint64_t recvDispls[8] = {0, 2, 3, 4, 5, 6, 7, 8};

    HcomAllToAllVOp op(HcomAllToAllKind::kAllToAllV);
    hcut::CtxBuilder ctx;
    ctx.Input({8}, ge::DT_FLOAT, sendData);
    ctx.Input({8}, ge::DT_INT64, sendCounts);
    ctx.Input({8}, ge::DT_INT64, sendDispls);
    ctx.Input({8}, ge::DT_INT64, recvCounts);
    ctx.Input({8}, ge::DT_INT64, recvDispls);
    ctx.Output({8}, ge::DT_FLOAT, out);
    ctx.Attr(0, "ut_group");

    EXPECT_EQ(ge::GRAPH_SUCCESS, op.Execute(ctx.Build()));
}

// ---------------------------------------------------------------------------
// CalcCommMemSize (M3): V 分支 send counts+displs 加法守卫 (recv 侧须先通过)
// ---------------------------------------------------------------------------
TEST_F(GuardTest, AtoAllVRefreshCommMemSendCountMultiplyOverflowRejected)
{
    uint64_t sendData[8] = {};
    float out[8] = {};
    // V 加法守卫经真实数据路径不可达 (counts 读取层拒绝 ≥2^63, 两正值之和恒 ≤ UINT64_MAX-1);
    // 换可达的乘法守卫: sendCount=2^62+1 (合法正值) × unitSize=4 回绕 → CalcCommMemSize 拦截
    const uint64_t sendCounts[8] = {(UINT64_MAX >> 2) + 1, 0, 0, 0, 0, 0, 0, 0};
    const uint64_t sendDispls[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint64_t recvCounts[8] = {2, 1, 1, 1, 1, 1, 1, 1};
    const uint64_t recvDispls[8] = {0, 2, 3, 4, 5, 6, 7, 8};
    g_stubCtxVal[kIdxAddrRefresh] = 1; // Refresh 路径才消费 CalcCommMemSize

    HcomAllToAllVOp op(HcomAllToAllKind::kAllToAllV);
    hcut::CtxBuilder ctx;
    ctx.Input({8}, ge::DT_FLOAT, sendData);
    ctx.Input({8}, ge::DT_INT64, sendCounts);
    ctx.Input({8}, ge::DT_INT64, sendDispls);
    ctx.Input({8}, ge::DT_INT64, recvCounts);
    ctx.Input({8}, ge::DT_INT64, recvDispls);
    ctx.Output({8}, ge::DT_FLOAT, out);
    ctx.Attr(0, "ut_group");

    EXPECT_EQ(ge::GRAPH_FAILED, op.Execute(ctx.Build()));
}

// ---------------------------------------------------------------------------
// CalcCommMemSize (M3): VC 分支行累加守卫 (列和合法、行和回绕)
// ---------------------------------------------------------------------------
TEST_F(GuardTest, AtoAllVCRefreshCommMemRowSumOverflowRejected)
{
    uint64_t sendData[8] = {};
    float out[8] = {};
    // 列 0 (matrix[0,8,16,...]) 全 0 → CalcVCRecvDim 列和通过; 行 0 累加至 UINT64_MAX 后再 +1 回绕
    // (元素均 ≤ 2^63-1 合法正值, reader 非负检查通过)
    uint64_t matrix[64] = {0};
    matrix[0] = UINT64_MAX / 2; // 2^63-1
    matrix[1] = UINT64_MAX / 2; // 行累加 = 2^64-2 = UINT64_MAX-1
    matrix[2] = 1;              // → UINT64_MAX (仍不回绕)
    matrix[3] = 1;              // 第 4 次累加触发守卫
    g_stubCtxVal[kIdxAddrRefresh] = 1;

    HcomAllToAllVOp op(HcomAllToAllKind::kAllToAllVC);
    hcut::CtxBuilder ctx;
    ctx.Input({8}, ge::DT_FLOAT, sendData);
    ctx.Input({64}, ge::DT_INT64, matrix);
    ctx.Output({8}, ge::DT_FLOAT, out);
    ctx.Attr(0, static_cast<int64_t>(0)); // ATTR_IDX_RANK_VC
    ctx.Attr(1, "ut_group");              // ATTR_IDX_GROUP_VC

    EXPECT_EQ(ge::GRAPH_FAILED, op.Execute(ctx.Build()));
}

// ---------------------------------------------------------------------------
// InferVRecvShape (M3): 编译期 shape 推导的受检窄化 (直调 public 成员)
// ---------------------------------------------------------------------------
TEST_F(GuardTest, InferVRecvShapeOverflowRejected)
{
    // 2^63 值被读取层非负检查拦截, 经读取路径的可达分支是 int64 值域越界:
    // d=2^63-1 (合法正值) + c=2 → end 超 INT64_MAX → 编译期 shape 推导拒绝
    const uint64_t recvCounts[8] = {2, 0, 0, 0, 0, 0, 0, 0};
    const uint64_t rdispls[8] = {UINT64_MAX / 2, 0, 0, 0, 0, 0, 0, 0};
    HcomAllToAllVOp op(HcomAllToAllKind::kAllToAllV);
    hcut::CtxBuilder ctx;
    // 函数读取 INPUT_IDX_RECV_COUNTS=3 / INPUT_IDX_RECV_DISPLS=4, 0-2 槽位补哑输入
    ctx.Input({1}, ge::DT_INT64, rdispls);    // 0: 哑
    ctx.Input({1}, ge::DT_INT64, recvCounts); // 1: 哑
    ctx.Input({1}, ge::DT_INT64, rdispls);    // 2: 哑
    ctx.Input({8}, ge::DT_INT64, recvCounts); // 3: recv_counts
    ctx.Input({8}, ge::DT_INT64, rdispls);    // 4: recv_displacements
    gert::Shape shape;
    EXPECT_EQ(ge::GRAPH_FAILED, op.InferVRecvShape(ctx.BuildAsInferShape(), &shape));
}

TEST_F(GuardTest, InferVRecvShapeSmallCountsSetsDim)
{
    const uint64_t recvCounts[8] = {1, 2, 1, 0, 0, 0, 0, 0};
    const uint64_t rdispls[8] = {0, 3, 5, 0, 0, 0, 0, 0};
    HcomAllToAllVOp op(HcomAllToAllKind::kAllToAllV);
    hcut::CtxBuilder ctx;
    ctx.Input({1}, ge::DT_INT64, recvCounts);
    ctx.Input({1}, ge::DT_INT64, rdispls);
    ctx.Input({1}, ge::DT_INT64, recvCounts);
    ctx.Input({8}, ge::DT_INT64, recvCounts);
    ctx.Input({8}, ge::DT_INT64, rdispls);
    gert::Shape shape;
    ASSERT_EQ(ge::GRAPH_SUCCESS, op.InferVRecvShape(ctx.BuildAsInferShape(), &shape));
    EXPECT_EQ(6, shape.GetDim(0)); // max(0+1, 3+2, 5+1)
}

// ---------------------------------------------------------------------------
// ResolveRecvDisplsForInferShape (M1): AGV 前缀和回退的 u64 回绕守卫 (直调 public 成员)
// ---------------------------------------------------------------------------
TEST_F(GuardTest, AgvPrefixSumOverflowRejected)
{
    // displs 张量未接入 (空上下文 → GetInputTensor null → 前缀和回退), counts 回绕 → 拒绝
    const uint64_t recvCounts[2] = {UINT64_MAX / 2 + 1, UINT64_MAX / 2 + 1};
    HcomAllGatherVOp op;
    hcut::CtxBuilder ctx; // 0 输入 → displs 走回退分支
    std::vector<uint64_t> counts(recvCounts, recvCounts + 2);
    std::vector<uint64_t> displs;
    EXPECT_EQ(ge::GRAPH_FAILED, op.ResolveRecvDisplsForInferShape(ctx.BuildAsInferShape(), counts, displs));
}

TEST_F(GuardTest, AgvPrefixSumSmallCountsSucceeds)
{
    const uint64_t recvCounts[3] = {1, 2, 3};
    HcomAllGatherVOp op;
    hcut::CtxBuilder ctx;
    std::vector<uint64_t> counts(recvCounts, recvCounts + 3);
    std::vector<uint64_t> displs;
    ASSERT_EQ(ge::GRAPH_SUCCESS, op.ResolveRecvDisplsForInferShape(ctx.BuildAsInferShape(), counts, displs));
    ASSERT_EQ(3U, displs.size());
    EXPECT_EQ(0U, displs[0]);
    EXPECT_EQ(1U, displs[1]);
    EXPECT_EQ(3U, displs[2]);
}

// ---------------------------------------------------------------------------
// AllocateNegotiatedOutput (M4): Recv 动态输出分配粒度防御 (直调 public static 成员)
// ---------------------------------------------------------------------------
TEST_F(GuardTest, RecvNegotiatedOutputGranularityInsufficientRejected)
{
    // GE 按协商 shape 分配 1 元素 (4B) < 对齐 count 8×4=32B → 报错而非越界写
    float smallOut[1] = {};
    HcomReceiveOp op;
    hcut::CtxBuilder ctx;
    ctx.Output({1}, ge::DT_INT32, smallOut);
    ctx.Attr(4, static_cast<int64_t>(ge::DT_INT32)); // ATTR_DTYPE

    hccl::HcclOpState st;
    st.ctx = ctx.Build();
    st.opType = "HcomReceive";
    gert::Shape negotiatedShape({1});
    size_t outputSize = 0;
    EXPECT_EQ(ge::GRAPH_FAILED, HcomReceiveOp::AllocateNegotiatedOutput(st, negotiatedShape, 8, 4, outputSize));
}

TEST_F(GuardTest, RecvNegotiatedOutputSufficientSucceeds)
{
    float outBuf[8] = {};
    HcomReceiveOp op;
    hcut::CtxBuilder ctx;
    ctx.Output({8}, ge::DT_INT32, outBuf);
    ctx.Attr(4, static_cast<int64_t>(ge::DT_INT32));

    hccl::HcclOpState st;
    st.ctx = ctx.Build();
    st.opType = "HcomReceive";
    gert::Shape negotiatedShape({8});
    size_t outputSize = 0;
    ASSERT_EQ(ge::GRAPH_SUCCESS, HcomReceiveOp::AllocateNegotiatedOutput(st, negotiatedShape, 8, 4, outputSize));
    EXPECT_EQ(32U, outputSize);
    ASSERT_EQ(1U, st.outputPtrs.size());
    EXPECT_EQ(static_cast<void*>(outBuf), st.outputPtrs[0]);
}

// ---------------------------------------------------------------------------
// Recv srTag/srcRank 值域守卫 (检视: 静态/动态两处新增检查无 UT)
// 注: 动态路径守卫位于 ExtractParamsDynamic 入口段, 两段式协议之前, 可经 Execute 直达
// ---------------------------------------------------------------------------
TEST_F(GuardTest, RecvStaticSrTagOverflowRejected)
{
    HcomReceiveOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({1}, ge::DT_INT32, nullptr); // Recv 无数据输入, 占位以满足提取链
    ctx.Output({1}, ge::DT_INT32, nullptr);
    // ATTR_GROUP=0, ATTR_SR_TAG=1, ATTR_SRC_RANK=2, ATTR_SHAPE=3, ATTR_DTYPE=4
    ctx.Attr(0, "ut_group").Attr(1, static_cast<int64_t>(-1)).Attr(2, static_cast<int64_t>(0));
    ctx.Attr(3, static_cast<int64_t>(0)).Attr(4, static_cast<int64_t>(ge::DT_INT32));

    EXPECT_EQ(ge::GRAPH_FAILED, op.Execute(ctx.Build()));
    EXPECT_EQ(0U, g_stubRecvCalls.size()); // 守卫先于通信入口, 零记录
}

TEST_F(GuardTest, RecvDynamicSrTagOverflowRejected)
{
    g_stubCtxVal[kIdxStaticModel] = 0; // 动态路径 (守卫在两段式协议之前)
    HcomReceiveOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({1}, ge::DT_INT32, nullptr);
    ctx.Output({1}, ge::DT_INT32, nullptr);
    ctx.Attr(0, "ut_group").Attr(1, static_cast<int64_t>(-1)).Attr(2, static_cast<int64_t>(0));
    ctx.Attr(3, static_cast<int64_t>(0)).Attr(4, static_cast<int64_t>(ge::DT_INT32));

    EXPECT_EQ(ge::GRAPH_FAILED, op.Execute(ctx.Build()));
    EXPECT_EQ(0U, g_stubRecvCalls.size()); // 守卫先于通信入口, 零记录
}

// ---------------------------------------------------------------------------
// InferVCRecvShape 方阵校验 (检视: 非完全平方矩阵拒绝无触发用例)
// ---------------------------------------------------------------------------
TEST_F(GuardTest, InferVCRecvShapeNonSquareMatrixRejected)
{
    uint64_t matrix[99];
    std::fill(matrix, matrix + 99, 1); // sqrt(99)=9, 81≠99 → 非方阵拒绝
    HcomAllToAllVOp op(HcomAllToAllKind::kAllToAllVC);
    hcut::CtxBuilder ctx;
    ctx.Input({1}, ge::DT_INT64, matrix);  // 0: send_data 哑占位
    ctx.Input({99}, ge::DT_INT64, matrix); // 1: INPUT_IDX_SEND_COUNT_MATRIX
    ctx.Attr(0, static_cast<int64_t>(0));  // ATTR_IDX_RANK_VC
    gert::Shape shape;
    EXPECT_EQ(ge::GRAPH_FAILED, op.InferVCRecvShape(ctx.BuildAsInferShape(), &shape));
}

TEST_F(GuardTest, InferVCRecvShapeSquareMatrixSucceeds)
{
    // 递增值: rank=0 列和 = matrix[0,4,8,12] = 1+5+9+13 = 28 (行和 1+2+3+4=10 可区分,
    // 列读取误写为行读取时用例变红)
    uint64_t matrix[16];
    for (int i = 0; i < 16; i++) {
        matrix[i] = static_cast<uint64_t>(i) + 1;
    }
    HcomAllToAllVOp op(HcomAllToAllKind::kAllToAllVC);
    hcut::CtxBuilder ctx;
    ctx.Input({1}, ge::DT_INT64, matrix);
    ctx.Input({16}, ge::DT_INT64, matrix);
    ctx.Attr(0, static_cast<int64_t>(0));
    gert::Shape shape;
    ASSERT_EQ(ge::GRAPH_SUCCESS, op.InferVCRecvShape(ctx.BuildAsInferShape(), &shape));
    EXPECT_EQ(28, shape.GetDim(0));
}

TEST_F(GuardTest, InferVCRecvShapeColumnOverflowRejected)
{
    // 累加溢出守卫 (cnt > INT64_MAX 单值分支经编译期读取路径不可达: ReadHostU64Vector 按
    // int64 非负拒绝 2^63, 与执行侧 D2H 原生 u64 口径不同)。可达分支: 各元素均 ≤ INT64_MAX
    // 的合法正值, 列累加至 INT64_MAX 后再 +1 → recvCount > INT64_MAX - cnt 守卫拒绝
    // 2×2 方阵 (r-major), rank=0 列元素为 matrix[0] 与 matrix[0+1*2]=matrix[2]
    uint64_t matrix[4] = {static_cast<uint64_t>(INT64_MAX), 0, 1, 0};
    HcomAllToAllVOp op(HcomAllToAllKind::kAllToAllVC);
    hcut::CtxBuilder ctx;
    ctx.Input({1}, ge::DT_INT64, matrix);
    ctx.Input({4}, ge::DT_INT64, matrix);
    ctx.Attr(0, static_cast<int64_t>(0));
    gert::Shape shape;
    EXPECT_EQ(ge::GRAPH_FAILED, op.InferVCRecvShape(ctx.BuildAsInferShape(), &shape));
}
