/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 算子层纯函数 UT (M2/M10): RS SplitV 融合布局校验/推导链 + AllReduce 动态分段
// — 真实算子 TU 链接 (hcom_reduce_scatter_op.cc / hcom_all_reduce_op.cc),
// 上下文/引擎/通信入口经 stub 桩化, 布局函数以向量直测各分支与推导结果
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hccl_custom_op.h"
#include "hcom_all_reduce_op.h"
#include "hcom_reduce_scatter_op.h"

using hccl::HcclOpState;

namespace {
// 地址辅助: 以 base 为起点按字节偏移构造指针向量
std::vector<void*> Addrs(uintptr_t base, const std::vector<uint64_t>& offsets)
{
    std::vector<void*> v;
    v.reserve(offsets.size());
    for (uint64_t o : offsets) {
        v.push_back(reinterpret_cast<void*>(base + o));
    }
    return v;
}
} // namespace

// ---------------------------------------------------------------------------
// ValidateFusedSliceSizes (校验 1): 同一 op 的各槽位切片字节数一致
// 布局: input[r*N+op], fusedOps=N, 槽位步长 = inBytes[op] … 校验各槽位对应切片相等
// ---------------------------------------------------------------------------
TEST(FusedLayoutTest, SliceSizesUniformSucceeds)
{
    // r-major 布局 input[r*N+op] (N=2, rankSize=3): op0 切片恒 128B, op1 恒 256B, 跨槽一致
    std::vector<uint64_t> inBytes = {128, 256, 128, 256, 128, 256};
    EXPECT_EQ(ge::GRAPH_SUCCESS, HcomReduceScatterOp::ValidateFusedSliceSizes(inBytes, 2, 3));
}

TEST(FusedLayoutTest, SliceSizeMismatchFails)
{
    // op1 的 rank2 切片 255B ≠ 同槽位其它 256B
    std::vector<uint64_t> inBytes = {128, 128, 128, 256, 256, 255};
    EXPECT_EQ(ge::GRAPH_FAILED, HcomReduceScatterOp::ValidateFusedSliceSizes(inBytes, 2, 3));
}

// ---------------------------------------------------------------------------
// ValidateFusedContiguity (校验 2): 输入地址连续升序不重叠
// ---------------------------------------------------------------------------
TEST(FusedLayoutTest, ContiguousAddrsSucceeds)
{
    const uintptr_t base = 0x10000;
    std::vector<uint64_t> bytes = {128, 128, 128};
    // 相邻恰好首尾相接 (addr[i+1] = addr[i] + bytes[i])
    auto addrs = Addrs(base, {0, 128, 256});
    EXPECT_EQ(ge::GRAPH_SUCCESS, HcomReduceScatterOp::ValidateFusedContiguity(addrs, bytes));
}

TEST(FusedLayoutTest, OverlapAddrsFail)
{
    const uintptr_t base = 0x10000;
    std::vector<uint64_t> bytes = {128, 128};
    // 第二个输入起点落进第一个输入范围内 → 重叠
    auto addrs = Addrs(base, {0, 64});
    EXPECT_EQ(ge::GRAPH_FAILED, HcomReduceScatterOp::ValidateFusedContiguity(addrs, bytes));
}

TEST(FusedLayoutTest, GapsAreLegalCrackRegions)
{
    const uintptr_t base = 0x10000;
    std::vector<uint64_t> bytes = {128, 128};
    // 中间缝隙合法: 间隙正是 crack 清零来源, 连续性校验仅拒绝重叠/降序
    auto addrs = Addrs(base, {0, 256});
    EXPECT_EQ(ge::GRAPH_SUCCESS, HcomReduceScatterOp::ValidateFusedContiguity(addrs, bytes));
}

// ---------------------------------------------------------------------------
// CalcUniformSlotBytes (校验 3): 槽位均匀 (input[N]-input[0] 为槽位步长, 各槽位步长相等) + 对齐
// ---------------------------------------------------------------------------
TEST(FusedLayoutTest, UniformSlotsSucceed)
{
    const uintptr_t base = 0x10000;
    // N=2, rankSize=2: 槽0 = [0,128),[128,256); 槽1 = [512,640),[640,768) — slotBytes=512
    auto addrs = Addrs(base, {0, 128, 512, 640});
    uint64_t slotBytes = 0;
    EXPECT_EQ(ge::GRAPH_SUCCESS, HcomReduceScatterOp::CalcUniformSlotBytes(addrs, 2, 2, 4, slotBytes));
    EXPECT_EQ(512U, slotBytes);
}

TEST(FusedLayoutTest, UnevenSlotsFail)
{
    const uintptr_t base = 0x10000;
    // N=2, rankSize=3: slotBytes=addr[2]-addr[0]=512; 中间槽位步长 addr[4]-addr[2]=1536 ≠ 512 → 失败
    auto addrs = Addrs(base, {0, 128, 512, 640, 2048, 2176});
    uint64_t slotBytes = 0;
    EXPECT_EQ(ge::GRAPH_FAILED, HcomReduceScatterOp::CalcUniformSlotBytes(addrs, 2, 3, 4, slotBytes));
}

// ---------------------------------------------------------------------------
// ValidateFusedOutputMirror (校验 4): 输出槽位布局镜像输入
// ---------------------------------------------------------------------------
TEST(FusedLayoutTest, MirroredOutputsSucceed)
{
    const uintptr_t inBase = 0x100000;
    const uintptr_t outBase = 0x200000;
    auto inAddrs = Addrs(inBase, {0, 128, 256});
    auto outAddrs = Addrs(outBase, {0, 128, 256});
    EXPECT_EQ(ge::GRAPH_SUCCESS, HcomReduceScatterOp::ValidateFusedOutputMirror(inAddrs, outAddrs, 3));
}

TEST(FusedLayoutTest, NonMirroredOutputsFail)
{
    const uintptr_t inBase = 0x100000;
    const uintptr_t outBase = 0x200000;
    auto inAddrs = Addrs(inBase, {0, 128, 256});
    auto outAddrs = Addrs(outBase, {0, 128, 999}); // 第 3 输出偏移不镜像
    EXPECT_EQ(ge::GRAPH_FAILED, HcomReduceScatterOp::ValidateFusedOutputMirror(inAddrs, outAddrs, 3));
}

// ---------------------------------------------------------------------------
// DeriveFusedCracks: crack 由实际地址间隙推导 (相邻输入间隙 + 尾隙)
// ---------------------------------------------------------------------------
TEST(FusedLayoutTest, DeriveCracksFromGaps)
{
    // N=2, rankSize=2 → inputNum=4, slotBytes=288; 相邻输入间隙 16/0/16, 尾隙 32
    // inAddrs 偏移: 0, 144(128+16), 272(144+128+0), 416(272+128+16); 末输入尾 544, 总覆盖 288*2=576
    const uintptr_t base = 0x10000;
    auto inAddrs = Addrs(base, {0, 144, 272, 416});
    std::vector<uint64_t> inBytes = {128, 128, 128, 128};
    HcclOpState st;
    HcomReduceScatterOp::DeriveFusedCracks(inAddrs, inBytes, 4, 288, 2, st);
    // 期望: gap0 {offset 128 size 16} / gap2 {offset 400 size 16} / 尾隙 {offset 544 size 32}
    ASSERT_EQ(3U, st.crackOffsets.size());
    ASSERT_EQ(3U, st.crackSizes.size());
    EXPECT_EQ(128U, st.crackOffsets[0]);
    EXPECT_EQ(16U, st.crackSizes[0]);
    EXPECT_EQ(400U, st.crackOffsets[1]);
    EXPECT_EQ(16U, st.crackSizes[1]);
    EXPECT_EQ(544U, st.crackOffsets[2]);
    EXPECT_EQ(32U, st.crackSizes[2]);
}

// ---------------------------------------------------------------------------
// HcomAllReduceOp::CalcBatchSlices: 动态路径按 CCL buffer 容量分段
// ---------------------------------------------------------------------------
namespace {
HcclOpState MakeStateWithCounts(const std::vector<uint64_t>& counts)
{
    HcclOpState st;
    st.inputCounts = counts;
    st.count = 0;
    for (uint64_t c : counts) {
        st.count += c;
    }
    return st;
}
} // namespace

TEST(CalcBatchSlicesTest, SingleInputFitsOneSlice)
{
    auto st = MakeStateWithCounts({100});
    std::vector<uint32_t> slices;
    EXPECT_EQ(ge::GRAPH_SUCCESS, HcomAllReduceOp::CalcBatchSlices(st, 400, slices));
    ASSERT_EQ(1U, slices.size());
    EXPECT_EQ(1U, slices[0]);
}

TEST(CalcBatchSlicesTest, SplitAtCapacityBoundary)
{
    // 容量 400, 计数 {100,200,150,50}: 100+200=300 ≤400, +150 → 超 → 首 slice [2], 尾 slice [2]
    auto st = MakeStateWithCounts({100, 200, 150, 50});
    std::vector<uint32_t> slices;
    EXPECT_EQ(ge::GRAPH_SUCCESS, HcomAllReduceOp::CalcBatchSlices(st, 400, slices));
    ASSERT_EQ(2U, slices.size());
    EXPECT_EQ(2U, slices[0]);
    EXPECT_EQ(2U, slices[1]);
}

TEST(CalcBatchSlicesTest, ExactCapacityStaysOneSlice)
{
    // 200+200 = 400 恰好等于容量 (<=) → 单 slice
    auto st = MakeStateWithCounts({200, 200});
    std::vector<uint32_t> slices;
    EXPECT_EQ(ge::GRAPH_SUCCESS, HcomAllReduceOp::CalcBatchSlices(st, 400, slices));
    ASSERT_EQ(1U, slices.size());
    EXPECT_EQ(2U, slices[0]);
}

TEST(CalcBatchSlicesTest, SingleInputExceedsCapacityBecomesDirectSlice)
{
    // 4239 语义 (对照 lowering sliceIdx==0 分支): 单 input 超容独立成 slice(1) 直传, 不报错
    auto st = MakeStateWithCounts({401});
    std::vector<uint32_t> slices;
    EXPECT_EQ(ge::GRAPH_SUCCESS, HcomAllReduceOp::CalcBatchSlices(st, 400, slices));
    ASSERT_EQ(1U, slices.size());
    EXPECT_EQ(1U, slices[0]);
}

TEST(CalcBatchSlicesTest, MidPositionOversizeFlushesPendingSlice)
{
    // 中位超容: else 分支 flush 前段({300}), pending=1, 尾部收口 push(size-sliceSum)
    // 300≤400 累入 sliceIdx=1; 500 超界 flush [1], sliceIdx 重置 1 待定; 末尾 push(2-1)=1
    auto st = MakeStateWithCounts({300, 500});
    std::vector<uint32_t> slices;
    EXPECT_EQ(ge::GRAPH_SUCCESS, HcomAllReduceOp::CalcBatchSlices(st, 400, slices));
    ASSERT_EQ(2U, slices.size());
    EXPECT_EQ(1U, slices[0]);
    EXPECT_EQ(1U, slices[1]);
}

TEST(CalcBatchSlicesTest, ConsecutiveOversizeEachIndependent)
{
    // 连续超容: 首 100 走累加分支 (sliceIdx=1 待定), 两个 500 依次走 else 分支
    // flush 待定段再自身待定, 尾部收口补齐 — 覆盖中位超容的 flush+收口链
    auto st = MakeStateWithCounts({100, 500, 500});
    std::vector<uint32_t> slices;
    EXPECT_EQ(ge::GRAPH_SUCCESS, HcomAllReduceOp::CalcBatchSlices(st, 400, slices));
    ASSERT_EQ(3U, slices.size());
    EXPECT_EQ(1U, slices[0]);
    EXPECT_EQ(1U, slices[1]);
    EXPECT_EQ(1U, slices[2]);
}

TEST(CalcBatchSlicesTest, OversizeThenSmallInputsSliceCorrectly)
{
    // 超容独立成 slice 后, 后续小 inputs 正常累积分段
    auto st = MakeStateWithCounts({500, 100, 200});
    std::vector<uint32_t> slices;
    EXPECT_EQ(ge::GRAPH_SUCCESS, HcomAllReduceOp::CalcBatchSlices(st, 400, slices));
    // [500] 独立直传 + [100,200]=300≤400 同段
    ASSERT_EQ(2U, slices.size());
    EXPECT_EQ(1U, slices[0]);
    EXPECT_EQ(2U, slices[1]);
}

TEST(CalcBatchSlicesTest, ZeroCountInputsGroupWithOthers)
{
    // 0 计数输入混入: 与后续输入同 slice (自身不触发分段)
    auto st = MakeStateWithCounts({0, 300, 0, 300});
    std::vector<uint32_t> slices;
    EXPECT_EQ(ge::GRAPH_SUCCESS, HcomAllReduceOp::CalcBatchSlices(st, 400, slices));
    // 0+300=300 ≤400, +0 ≤400, +300 → 600 超 → 首 slice [3], 尾 slice [1]
    ASSERT_EQ(2U, slices.size());
    EXPECT_EQ(3U, slices[0]);
    EXPECT_EQ(1U, slices[1]);
}
