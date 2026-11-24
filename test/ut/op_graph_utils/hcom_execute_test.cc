/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Execute 全链路 UT (M2/M10 缺口闭环): 经真实入口驱动 ExtractParams 静态/动态双路径与
// Launch Direct/Refresh/Dynamic 三分支, 断言经通信入口记录桩核对 count/tag/迭代守恒。
// 真实算子 TU 链接 (all_reduce/reduce_scatter/send/broadcast + hccl_custom_op),
// 上下文为 gert_context_helper.h 的 POD 构造, 生产代码零改动。
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gert_context_helper.h"
#include "hcom_all_reduce_op.h"
#include "hcom_reduce_op.h"
#include "hcom_reduce_scatter_op.h"
#include "hcom_broadcast_op.h"
#include "hcom_send_op.h"

// 桩层注入项
extern bool g_stubCtxValRet[8];
extern uint64_t g_stubCtxVal[8];
extern uint64_t g_stubCclBufSize;

// 记录桩共享头: StubCommCall 与 g_stub*Calls 单点维护 (op_layer_stub.cc 定义)
#include "stub/stub_comm_recorder.h"

extern std::vector<gert::Tensor*> g_stubOutputTensors;

namespace {
// ContextParam 桩索引 (kernel_args.h 枚举值)
constexpr int kIdxStaticModel = 0;
constexpr int kIdxAddrRefresh = 1;

// 每用例复位全部注入, 防跨用例泄漏
class ExecuteTest : public ::testing::Test {
protected:
    void SetUp() override { ResetAll(); }
    void TearDown() override { ResetAll(); }
    static void ResetAll()
    {
        for (size_t i = 0; i < 8; i++) {
            g_stubCtxValRet[i] = false;
            g_stubCtxVal[i] = 0;
        }
        // 两个分派信号须成功注入 (ResolvePathSignals/ResolveStaticSignals 对查询失败按错误终止,
        // false 不是"缺省静态"): 默认静态图 + 不刷新, 用例按需覆写值
        g_stubCtxValRet[kIdxStaticModel] = true;
        g_stubCtxVal[kIdxStaticModel] = 1;
        g_stubCtxValRet[kIdxAddrRefresh] = true;
        g_stubCtxVal[kIdxAddrRefresh] = 0;
        for (size_t i = 0; i < 16; i++) {
            g_stubAttrs[i] = nullptr;
        }
        g_stubCclBufSize = 1U << 20;
        g_stubArCalls.clear();
        g_stubRsCalls.clear();
        g_stubSendCalls.clear();
        g_stubBcastCalls.clear();
        g_stubReduceCalls.clear();
        g_stubRecvCalls.clear();
        g_stubOutputTensors.clear();
    }
};

} // namespace

// ---------------------------------------------------------------------------
// M10: AllReduce ExtractParamsStatic + LaunchDirect (静态路径, 单输入)
// ---------------------------------------------------------------------------
TEST_F(ExecuteTest, AllReduceStaticDirectSingleInput)
{
    float data[8] = {};
    float outBuf[8] = {};
    HcomAllReduceOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({8}, ge::DT_FLOAT, data).Output({8}, ge::DT_FLOAT, outBuf);
    ctx.Attr(0, "sum").Attr(1, "ut_group").Attr(2, static_cast<int64_t>(0)).Attr(3, static_cast<int64_t>(0));

    EXPECT_EQ(ge::GRAPH_SUCCESS, op.Execute(ctx.Build()));
    // LaunchDirect 经记录桩断言: count 为 512B 对齐融合口径 (32B → 512B → 128 元素),
    // 直传用户地址, tag 为桩基准串
    ASSERT_EQ(1U, g_stubArCalls.size());
    EXPECT_EQ(128U, g_stubArCalls[0].count);
    EXPECT_EQ(static_cast<const void*>(data), g_stubArCalls[0].buf1);
    EXPECT_EQ(static_cast<const void*>(outBuf), g_stubArCalls[0].buf2);
    EXPECT_EQ("UT_BASE_TAG", g_stubArCalls[0].tag);
}

// ---------------------------------------------------------------------------
// M10: AllReduce 静态多输入融合 count 口径 (512B 对齐累加, crack 累计)
// ---------------------------------------------------------------------------
TEST_F(ExecuteTest, AllReduceStaticMultiInputAlignedCount)
{
    float d0[4] = {}; // 16B → 对齐 512B
    float d1[2] = {}; // 8B → 对齐 512B
    float out0[4] = {};
    float out1[2] = {};
    HcomAllReduceOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({4}, ge::DT_FLOAT, d0).Input({2}, ge::DT_FLOAT, d1);
    ctx.Output({4}, ge::DT_FLOAT, out0).Output({2}, ge::DT_FLOAT, out1);
    ctx.Attr(0, "sum").Attr(1, "ut_group").Attr(2, static_cast<int64_t>(0)).Attr(3, static_cast<int64_t>(0));

    EXPECT_EQ(ge::GRAPH_SUCCESS, op.Execute(ctx.Build()));
    ASSERT_EQ(1U, g_stubArCalls.size());
    // count = (512+512)B / 4B = 256 (InfoStore CalcCountForAlignedOp 口径)
    EXPECT_EQ(256U, g_stubArCalls[0].count);
}

// ---------------------------------------------------------------------------
// M10: AllReduce 动态路径 (kStaticModel=0) — ExtractParamsDynamic + LaunchBatchDynamic
// ---------------------------------------------------------------------------
TEST_F(ExecuteTest, AllReduceDynamicBatchSingleSlice)
{
    float d0[2] = {};
    float d1[3] = {};
    float out0[2] = {};
    float out1[3] = {};
    g_stubCtxVal[kIdxStaticModel] = 0; // 动态图 (信号成功注入, 值 0)
    HcomAllReduceOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({2}, ge::DT_FLOAT, d0).Input({3}, ge::DT_FLOAT, d1);
    ctx.Output({2}, ge::DT_FLOAT, out0).Output({3}, ge::DT_FLOAT, out1);
    ctx.Attr(0, "sum").Attr(1, "ut_group").Attr(2, static_cast<int64_t>(0)).Attr(3, static_cast<int64_t>(0));

    EXPECT_EQ(ge::GRAPH_SUCCESS, op.Execute(ctx.Build()));
    // 多 input 经 CCL buf 中转单 slice: commCount = 2+3 = 5
    ASSERT_EQ(1U, g_stubArCalls.size());
    EXPECT_EQ(5U, g_stubArCalls[0].count);
}

// ---------------------------------------------------------------------------
// M8 端到端回归: 全部输入 0 元素 → count==0 提前成功, 零通信调用
// ---------------------------------------------------------------------------
TEST_F(ExecuteTest, AllReduceDynamicAllEmptyInputsNoOp)
{
    float out0[1] = {};
    g_stubCtxVal[kIdxStaticModel] = 0; // 动态 (信号成功注入, 值 0)
    HcomAllReduceOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({0}, ge::DT_FLOAT, nullptr).Input({0}, ge::DT_FLOAT, nullptr);
    ctx.Output({0}, ge::DT_FLOAT, out0).Output({0}, ge::DT_FLOAT, out0);
    ctx.Attr(0, "sum").Attr(1, "ut_group").Attr(2, static_cast<int64_t>(0)).Attr(3, static_cast<int64_t>(0));

    EXPECT_EQ(ge::GRAPH_SUCCESS, op.Execute(ctx.Build()));
    EXPECT_EQ(0U, g_stubArCalls.size()); // no-op: 不进任何通信
}

// ---------------------------------------------------------------------------
// 0 元素输入回归 (检视: Reduce 两路径与 AllReduce 静态路径无 UT)
// ---------------------------------------------------------------------------
TEST_F(ExecuteTest, AllReduceStaticEmptyInputNoOp)
{
    float out[1] = {};
    HcomAllReduceOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({0}, ge::DT_FLOAT, nullptr).Output({0}, ge::DT_FLOAT, out);
    ctx.Attr(0, "sum").Attr(1, "ut_group").Attr(2, static_cast<int64_t>(0)).Attr(3, static_cast<int64_t>(0));

    EXPECT_EQ(ge::GRAPH_SUCCESS, op.Execute(ctx.Build()));
    // 静态路径 LaunchDirect 无 count==0 提前返回, 直传空地址+零计数给通信入口 (HCCL 入口
    // 对 count==0 合法成功)。通信入口必须恰好一次: 少于一次 = 路径意外短路跳过通信
    ASSERT_EQ(1U, g_stubArCalls.size());
    EXPECT_EQ(0U, g_stubArCalls[0].count);
}

TEST_F(ExecuteTest, ReduceStaticEmptyInputNoOp)
{
    float out[1] = {};
    HcomReduceOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({0}, ge::DT_FLOAT, nullptr).Output({0}, ge::DT_FLOAT, out);
    ctx.Attr(0, static_cast<int64_t>(0)).Attr(1, "sum").Attr(2, "ut_group");
    ctx.Attr(3, static_cast<int64_t>(0)).Attr(4, static_cast<int64_t>(0));

    EXPECT_EQ(ge::GRAPH_SUCCESS, op.Execute(ctx.Build()));
    // 通信入口必须恰好一次且 count==0: 少于一次 = 意外短路, 多于一次 = 重复调用
    ASSERT_EQ(1U, g_stubReduceCalls.size());
    EXPECT_EQ(0U, g_stubReduceCalls[0].count);
}

TEST_F(ExecuteTest, ReduceDynamicEmptyInputNoOp)
{
    float out[1] = {};
    g_stubCtxVal[kIdxStaticModel] = 0; // 动态路径
    HcomReduceOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({0}, ge::DT_FLOAT, nullptr).Output({0}, ge::DT_FLOAT, out);
    ctx.Attr(0, static_cast<int64_t>(0)).Attr(1, "sum").Attr(2, "ut_group");
    ctx.Attr(3, static_cast<int64_t>(0)).Attr(4, static_cast<int64_t>(0));

    EXPECT_EQ(ge::GRAPH_SUCCESS, op.Execute(ctx.Build()));
    // 通信入口必须恰好一次且 count==0: 少于一次 = 意外短路, 多于一次 = 重复调用
    ASSERT_EQ(1U, g_stubReduceCalls.size());
    EXPECT_EQ(0U, g_stubReduceCalls[0].count);
}

// ---------------------------------------------------------------------------
// Send 动态路径负 shapeSize 守卫 (检视: 同款 Broadcast 守卫有 UT 本条没有)
// ---------------------------------------------------------------------------
TEST_F(ExecuteTest, SendDynamicNegativeShapeSizeRejected)
{
    g_stubCtxVal[kIdxStaticModel] = 0; // 动态路径 → ExtractParamsDynamic 的先检后 cast 守卫
    HcomSendOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({-1, 512}, ge::DT_INT8, nullptr); // GetOriginShape 乘积 -512
    ctx.Output({512}, ge::DT_INT8, nullptr);
    ctx.Attr(0, "ut_group").Attr(1, static_cast<int64_t>(0)).Attr(2, static_cast<int64_t>(0));

    EXPECT_EQ(ge::GRAPH_FAILED, op.Execute(ctx.Build()));
    EXPECT_EQ(0U, g_stubSendCalls.size()); // 守卫先于通信入口
}

// ---------------------------------------------------------------------------
// AllReduce 动态路径混合空+非空守卫 (检视: 静态两用例已有, 动态侧缺触发用例)
// ---------------------------------------------------------------------------
TEST_F(ExecuteTest, AllReduceDynamicMixedEmptyAndNonEmptyRejected)
{
    float data[8] = {};
    float out[8] = {};
    g_stubCtxVal[kIdxStaticModel] = 0; // 动态路径
    HcomAllReduceOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({0}, ge::DT_FLOAT, nullptr); // 空: inputPtrs[0]=null
    ctx.Input({8}, ge::DT_FLOAT, data);    // 非空: count=8 > 0
    ctx.Output({0}, ge::DT_FLOAT, out);
    ctx.Output({8}, ge::DT_FLOAT, out);
    ctx.Attr(0, "sum").Attr(1, "ut_group").Attr(2, static_cast<int64_t>(0)).Attr(3, static_cast<int64_t>(0));

    EXPECT_EQ(ge::GRAPH_FAILED, op.Execute(ctx.Build()));
    EXPECT_EQ(0U, g_stubArCalls.size());
}

// ---------------------------------------------------------------------------
// 混合空+非空输入 (检视: 全空用例只覆盖 count==0, 混合时 nullptr buf 配 count>0)
// 首输入为空 → inputPtrs[0]=null, count 含非空对齐量 >0 → 通信入口必须被拦下
// ---------------------------------------------------------------------------
TEST_F(ExecuteTest, AllReduceStaticMixedEmptyAndNonEmptyRejected)
{
    float data[8] = {};
    float out[8] = {};
    HcomAllReduceOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({0}, ge::DT_FLOAT, nullptr); // 空: GetAddr=null, SetEmptyOutputTensor
    ctx.Input({8}, ge::DT_FLOAT, data);    // 非空: 32B → 对齐 512B → count=128
    ctx.Output({0}, ge::DT_FLOAT, out);
    ctx.Output({8}, ge::DT_FLOAT, out);
    ctx.Attr(0, "sum").Attr(1, "ut_group").Attr(2, static_cast<int64_t>(0)).Attr(3, static_cast<int64_t>(0));

    EXPECT_EQ(ge::GRAPH_FAILED, op.Execute(ctx.Build()));
    EXPECT_EQ(0U, g_stubArCalls.size()); // null buf + count>0 不得进通信
}

TEST_F(ExecuteTest, ReduceStaticMixedEmptyAndNonEmptyRejected)
{
    float data[8] = {};
    float out[8] = {};
    HcomReduceOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({0}, ge::DT_FLOAT, nullptr);
    ctx.Input({8}, ge::DT_FLOAT, data);
    ctx.Output({0}, ge::DT_FLOAT, out);
    ctx.Output({8}, ge::DT_FLOAT, out);
    ctx.Attr(0, static_cast<int64_t>(0)).Attr(1, "sum").Attr(2, "ut_group");
    ctx.Attr(3, static_cast<int64_t>(0)).Attr(4, static_cast<int64_t>(0));

    EXPECT_EQ(ge::GRAPH_FAILED, op.Execute(ctx.Build()));
    EXPECT_EQ(0U, g_stubReduceCalls.size());
}

// ---------------------------------------------------------------------------
// Send 动态路径 srTag/destRank 值域守卫 (检视: 原用例仅覆盖静态同名守卫)
// ---------------------------------------------------------------------------
TEST_F(ExecuteTest, SendDynamicNegativeSrTagRejected)
{
    float data[2] = {};
    float out[2] = {};
    g_stubCtxVal[kIdxStaticModel] = 0; // 动态路径 → ExtractParamsDynamic 的同款守卫
    HcomSendOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({2}, ge::DT_FLOAT, data).Output({2}, ge::DT_FLOAT, out);
    ctx.Attr(0, "ut_group").Attr(1, static_cast<int64_t>(-1)).Attr(2, static_cast<int64_t>(0));

    EXPECT_EQ(ge::GRAPH_FAILED, op.Execute(ctx.Build()));
    EXPECT_EQ(0U, g_stubSendCalls.size()); // 被动态路径守卫拦截, 不进通信
}

// ---------------------------------------------------------------------------
// 0 维标量拦截 (检视: AllocRsOutput GetDimNum 守卫无触发用例)
// InferShape 在编译期拦 0 维, eager 直执行可绕过 — Execute 侧守卫兜底
// ---------------------------------------------------------------------------
TEST_F(ExecuteTest, ReduceScatterStaticScalarInputRejected)
{
    int32_t scalar = 0;
    int32_t out[8] = {};
    HcomReduceScatterOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({}, ge::DT_INT32, &scalar); // 0 维标量
    ctx.Output({8}, ge::DT_INT32, out);   // 注册 output: 守卫缺失时用例才能走到成功返回而变红
    ctx.Attr(0, "sum").Attr(1, static_cast<int64_t>(0)).Attr(2, static_cast<int64_t>(0));
    ctx.Attr(3, "ut_group").Attr(4, static_cast<int64_t>(8));

    EXPECT_EQ(ge::GRAPH_FAILED, op.Execute(ctx.Build()));
    EXPECT_EQ(0U, g_stubRsCalls.size()); // 守卫先于通信入口
}

// ---------------------------------------------------------------------------
// M2: RS Refresh 循环 (kAddressRefreshEnabled=1) — 单迭代
// ---------------------------------------------------------------------------
TEST_F(ExecuteTest, ReduceScatterRefreshSingleIteration)
{
    float in[64] = {}; // 64 元素, rank_size=8 → count=8 (32B), CCL 1MB 装得下 → 单迭代
    float out[8] = {};
    g_stubCtxValRet[kIdxAddrRefresh] = true;
    g_stubCtxVal[kIdxAddrRefresh] = 1;
    HcomReduceScatterOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({64}, ge::DT_FLOAT, in).Output({8}, ge::DT_FLOAT, out);
    ctx.Attr(0, "sum").Attr(1, static_cast<int64_t>(0)).Attr(2, static_cast<int64_t>(0));
    ctx.Attr(3, "ut_group").Attr(4, static_cast<int64_t>(8));

    EXPECT_EQ(ge::GRAPH_SUCCESS, op.Execute(ctx.Build()));
    ASSERT_EQ(1U, g_stubRsCalls.size());
    EXPECT_EQ(8U, g_stubRsCalls[0].count);
    // RS 在分批覆盖表内: loopMaxTime≥1, tag 恒带 _looptime_ 序号 (单迭代为 _0)
    EXPECT_EQ("UT_BASE_TAG_looptime_0", g_stubRsCalls[0].tag);
}

// ---------------------------------------------------------------------------
// M2: RS Refresh 多迭代 — tag 数与实际迭代数守恒 (检视意见点名的回归风险点)
// ---------------------------------------------------------------------------
TEST_F(ExecuteTest, ReduceScatterRefreshMultiIterationTagConservation)
{
    float in[64] = {}; // count=8 元素=32B; CCL buf 压到 16B → maxCountPerLoop=4 元素 → 2 迭代
    float out[8] = {};
    g_stubCtxValRet[kIdxAddrRefresh] = true;
    g_stubCtxVal[kIdxAddrRefresh] = 1;
    g_stubCclBufSize = 128; // ≥ rankSize*unitSize=32B 最小膨胀单元; 128B → maxCountPerLoop=4
    HcomReduceScatterOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({64}, ge::DT_FLOAT, in).Output({8}, ge::DT_FLOAT, out);
    ctx.Attr(0, "sum").Attr(1, static_cast<int64_t>(0)).Attr(2, static_cast<int64_t>(0));
    ctx.Attr(3, "ut_group").Attr(4, static_cast<int64_t>(8));

    EXPECT_EQ(ge::GRAPH_SUCCESS, op.Execute(ctx.Build()));
    // 守恒: 通信调用次数 == 实际迭代数, tag 逐迭代唯一且带 _looptime_ 序号
    ASSERT_EQ(2U, g_stubRsCalls.size());
    EXPECT_EQ("UT_BASE_TAG_looptime_0", g_stubRsCalls[0].tag);
    EXPECT_EQ("UT_BASE_TAG_looptime_1", g_stubRsCalls[1].tag);
    EXPECT_EQ(4U, g_stubRsCalls[0].count); // 每迭代 count = min(countLeft, maxCountPerLoop)
    EXPECT_EQ(4U, g_stubRsCalls[1].count);
}

// ---------------------------------------------------------------------------
// A 类修复回归 (LOW): send srTag 负值 → Execute 静态路径报错, 零通信调用
// ---------------------------------------------------------------------------
TEST_F(ExecuteTest, SendNegativeSrTagRejected)
{
    float data[2] = {};
    float out[2] = {};
    HcomSendOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({2}, ge::DT_FLOAT, data).Output({2}, ge::DT_FLOAT, out);
    ctx.Attr(0, "ut_group").Attr(1, static_cast<int64_t>(-1)).Attr(2, static_cast<int64_t>(0));

    EXPECT_EQ(ge::GRAPH_FAILED, op.Execute(ctx.Build()));
    EXPECT_EQ(0U, g_stubSendCalls.size()); // 被边界校验拦截, 不进通信
}

TEST_F(ExecuteTest, SendDestRankOverflowRejected)
{
    float data[2] = {};
    float out[2] = {};
    HcomSendOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({2}, ge::DT_FLOAT, data).Output({2}, ge::DT_FLOAT, out);
    ctx.Attr(0, "ut_group").Attr(1, static_cast<int64_t>(7)).Attr(2, static_cast<int64_t>(0x100000000LL));

    EXPECT_EQ(ge::GRAPH_FAILED, op.Execute(ctx.Build()));
    EXPECT_EQ(0U, g_stubSendCalls.size());
}

// ---------------------------------------------------------------------------
// A 类修复回归 (LOW): broadcast 负 shapeSize (UNKNOWN_DIM 混入) → 动态路径报错
// ---------------------------------------------------------------------------
TEST_F(ExecuteTest, BroadcastNegativeShapeSizeRejected)
{
    g_stubCtxVal[kIdxStaticModel] = 0; // 动态路径
    HcomBroadcastOp op;
    hcut::CtxBuilder ctx;
    ctx.Input({-1, 512}, ge::DT_INT8, nullptr);               // shapeSize = -512
    ctx.Attr(0, static_cast<int64_t>(0)).Attr(1, "ut_group"); // ROOT_RANK=0, GROUP=1 (对照算子头)

    EXPECT_EQ(ge::GRAPH_FAILED, op.Execute(ctx.Build()));
    EXPECT_EQ(0U, g_stubBcastCalls.size());
}
