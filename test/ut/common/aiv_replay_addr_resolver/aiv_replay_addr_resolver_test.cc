/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// AIV 指令缓存录制/回放的地址归属分派（hccl_aiv_utils.h）UT。
// 单级算法 input/output 均来自外部 buffer；多级算法中间级可能以 output/cclBuffer 为输入/输出，
// 归属只能按 BufferType 判定：类型分派、下溢防御、DEFAULT 拒缓存、回放基址还原逐项覆盖。

#include <gtest/gtest.h>

#include "hccl_aiv_utils.h"

using namespace ops_hccl;

namespace {
constexpr u64 BASE_IN = 0x10000000ULL;
constexpr u64 BASE_OUT = 0x20000000ULL;
constexpr u64 CCL_ADDR = 0x50000000ULL;
} // namespace

// 录制：INPUT/OUTPUT 按各自基址计算偏移
TEST(AivReplayAddrResolverTest, calcOffsetByExternalBuffType)
{
    AivAddrOffsetResult inRes = CalcAivAddrOffset(BufferType::INPUT, BASE_IN + 0x400, BASE_IN, BASE_OUT);
    EXPECT_TRUE(inRes.cacheable);
    EXPECT_EQ(inRes.offset, 0x400U);

    // 框内级 output 当输入（AG Sequence intra）：input 侧标 OUTPUT，偏移按 output 基址算
    AivAddrOffsetResult outAsInRes = CalcAivAddrOffset(BufferType::OUTPUT, BASE_OUT + 0x800, BASE_IN, BASE_OUT);
    EXPECT_TRUE(outAsInRes.cacheable);
    EXPECT_EQ(outAsInRes.offset, 0x800U);

    // Broadcast 落点在 input buffer：output 侧标 INPUT，偏移按 input 基址算
    AivAddrOffsetResult inAsOutRes = CalcAivAddrOffset(BufferType::INPUT, BASE_IN + 0x100, BASE_IN, BASE_OUT);
    EXPECT_TRUE(inAsOutRes.cacheable);
    EXPECT_EQ(inAsOutRes.offset, 0x100U);

    // 基址本身：偏移 0 合法
    AivAddrOffsetResult zeroRes = CalcAivAddrOffset(BufferType::INPUT, BASE_IN, BASE_IN, BASE_OUT);
    EXPECT_TRUE(zeroRes.cacheable);
    EXPECT_EQ(zeroRes.offset, 0U);
}

// 录制：HCCL_BUFFER 归属 comm 级内部 buffer，偏移恒 0，回放复用录制期绝对地址
TEST(AivReplayAddrResolverTest, calcOffsetByCclBuffType)
{
    AivAddrOffsetResult res = CalcAivAddrOffset(BufferType::HCCL_BUFFER, CCL_ADDR, BASE_IN, BASE_OUT);
    EXPECT_TRUE(res.cacheable);
    EXPECT_EQ(res.offset, 0U);
}

// 录制：DEFAULT 无法判定归属，拒绝缓存
TEST(AivReplayAddrResolverTest, calcOffsetByDefaultTypeUncacheable)
{
    AivAddrOffsetResult res = CalcAivAddrOffset(BufferType::DEFAULT, BASE_IN, BASE_IN, BASE_OUT);
    EXPECT_FALSE(res.cacheable);
    EXPECT_EQ(res.offset, 0U);
}

// 录制：类型与地址矛盾（addr 小于类型对应基址）拒绝缓存，不得回绕成巨大偏移
TEST(AivReplayAddrResolverTest, calcOffsetUnderflowUncacheable)
{
    AivAddrOffsetResult inUnderflow = CalcAivAddrOffset(BufferType::INPUT, BASE_IN - 0x10, BASE_IN, BASE_OUT);
    EXPECT_FALSE(inUnderflow.cacheable);
    EXPECT_EQ(inUnderflow.offset, 0U);

    AivAddrOffsetResult outUnderflow = CalcAivAddrOffset(BufferType::OUTPUT, BASE_OUT - 0x10, BASE_IN, BASE_OUT);
    EXPECT_FALSE(outUnderflow.cacheable);
    EXPECT_EQ(outUnderflow.offset, 0U);
}

// 回放：外部归属按本次基址 + 录制偏移还原，跨调用 buffer 变化正确重定向
TEST(AivReplayAddrResolverTest, resolveReplayAddrByExternalBuffType)
{
    constexpr u64 NEW_BASE_IN = 0x70000000ULL;
    constexpr u64 NEW_BASE_OUT = 0x80000000ULL;

    u64 newAddr = 0;
    EXPECT_EQ(
        ResolveAivReplayAddr(BufferType::INPUT, NEW_BASE_IN, NEW_BASE_OUT, 0x400, BASE_IN + 0x400, newAddr),
        HCCL_SUCCESS);
    EXPECT_EQ(newAddr, NEW_BASE_IN + 0x400);

    EXPECT_EQ(
        ResolveAivReplayAddr(BufferType::OUTPUT, NEW_BASE_IN, NEW_BASE_OUT, 0x800, BASE_OUT + 0x800, newAddr),
        HCCL_SUCCESS);
    EXPECT_EQ(newAddr, NEW_BASE_OUT + 0x800);

    // Broadcast 语义：output 侧标 INPUT，按本次 input 基址还原
    EXPECT_EQ(
        ResolveAivReplayAddr(BufferType::INPUT, NEW_BASE_IN, NEW_BASE_OUT, 0x100, BASE_IN + 0x100, newAddr),
        HCCL_SUCCESS);
    EXPECT_EQ(newAddr, NEW_BASE_IN + 0x100);
}

// 回放：HCCL_BUFFER 复用录制期绝对地址（cclBuffer 随 comm 稳定，跨调用不变）
TEST(AivReplayAddrResolverTest, resolveReplayAddrByCclBuffType)
{
    u64 newAddr = 0;
    EXPECT_EQ(ResolveAivReplayAddr(BufferType::HCCL_BUFFER, BASE_IN, BASE_OUT, 0, CCL_ADDR, newAddr), HCCL_SUCCESS);
    EXPECT_EQ(newAddr, CCL_ADDR);
}

// 回放：不可缓存类型不应出现在缓存的指令集中，出现即报错
TEST(AivReplayAddrResolverTest, resolveReplayAddrByDefaultTypeRejected)
{
    u64 newAddr = 0;
    EXPECT_EQ(ResolveAivReplayAddr(BufferType::DEFAULT, BASE_IN, BASE_OUT, 0, BASE_IN, newAddr), HCCL_E_INTERNAL);
}

// 端到端场景：URMA AG Sequence 框内级（input=output buffer + 偏移）录制后在本次调用新 buffer 上还原
TEST(AivReplayAddrResolverTest, urmaAgIntraLevelRecordThenReplay)
{
    constexpr u64 OFF = 0x2000;
    // 录制：input = 旧 outputPtr + off，类型 OUTPUT
    AivAddrOffsetResult res = CalcAivAddrOffset(BufferType::OUTPUT, BASE_OUT + OFF, BASE_IN, BASE_OUT);
    ASSERT_TRUE(res.cacheable);
    ASSERT_EQ(res.offset, OFF);

    // 回放：以本次 outputPtr 为基址还原
    constexpr u64 NEW_BASE_OUT = 0x90000000ULL;
    u64 newAddr = 0;
    ASSERT_EQ(
        ResolveAivReplayAddr(BufferType::OUTPUT, BASE_IN, NEW_BASE_OUT, res.offset, BASE_OUT + OFF, newAddr),
        HCCL_SUCCESS);
    EXPECT_EQ(newAddr, NEW_BASE_OUT + OFF);
}
