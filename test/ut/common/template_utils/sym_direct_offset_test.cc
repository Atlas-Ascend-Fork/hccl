/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>
#include "ins_temp_all_to_all_v_pairwise.h"

using namespace ops_hccl;

// AllToAllV/AllToAllVC pairwise 对称内存直达的 TX 偏移推导与越界守卫（生产调用方：
// ins_temp_all_to_all_v_pairwise.cc 的 RunRingStep/RunFullMesh）

namespace {
constexpr u32 kTargetRank = 1;
constexpr u32 kMyAlgRank = 3;
constexpr u64 kDtypeSize = 2;

// 只填对称推导消费的字段，其余走默认值
TemplateDataParams MakeParams(u64 inputSize, u64 outputSize, u64 sendCounts, u64 sdispls, u64 peerRdispls)
{
    TemplateDataParams p;
    p.buffInfo.inputSize = inputSize;
    p.buffInfo.outputSize = outputSize;
    p.sendCounts = {0, sendCounts};
    p.sdispls = {0, sdispls};
    p.peerRdispls = {0, peerRdispls};
    return p;
}
} // namespace

// V 分支基址 = myAlgRank × sendCounts[target] × dataTypeSize（各 rank 发送布局一致的均匀排布）
TEST(AllToAllSymDirectTxDstBaseTest, VBaseOff)
{
    auto p = MakeParams(1024, 1024, 100, 10, 0);
    u64 base = 0;
    ASSERT_EQ(
        CalcSymDirectTxDstBase(p, HcclCMDType::HCCL_CMD_ALLTOALLV, kTargetRank, kMyAlgRank, kDtypeSize, base),
        HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(base, kMyAlgRank * 100 * kDtypeSize);
}

// VC 分支基址 = peerRdispls[target] × dataTypeSize（矩阵直给，与 sendCounts 无关）
TEST(AllToAllSymDirectTxDstBaseTest, VCBaseOff)
{
    auto p = MakeParams(1024, 1024, 100, 10, 77);
    u64 base = 0;
    ASSERT_EQ(
        CalcSymDirectTxDstBase(p, HcclCMDType::HCCL_CMD_ALLTOALLVC, kTargetRank, kMyAlgRank, kDtypeSize, base),
        HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(base, 77 * kDtypeSize);
}

// 0 长度发送：基址为 0 且校验通过（调用方 sendSize=0 时不调用，此处钉死函数自身对 0 的行为）
TEST(AllToAllSymDirectTxDstBaseTest, ZeroSend)
{
    auto p = MakeParams(1024, 1024, 0, 5, 77);
    u64 base = 99;
    ASSERT_EQ(
        CalcSymDirectTxDstBase(p, HcclCMDType::HCCL_CMD_ALLTOALLV, kTargetRank, kMyAlgRank, kDtypeSize, base),
        HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(base, 0u);
}

// input 界：偏移 + 长度恰好贴边通过，越 1 字节失败（两个越界方向分开钉）
TEST(AllToAllSymDirectTxDstBaseTest, InputSliceBoundary)
{
    // 800 + 200 = 1000，恰好等于 inputSize
    auto ok = MakeParams(1000, 1024, 100, 400, 0);
    u64 base = 0;
    EXPECT_EQ(
        CalcSymDirectTxDstBase(ok, HcclCMDType::HCCL_CMD_ALLTOALLV, kTargetRank, kMyAlgRank, kDtypeSize, base),
        HcclResult::HCCL_SUCCESS);

    // 长度越界：800 + 200 = 1000 > 999
    auto overLen = MakeParams(999, 1024, 100, 400, 0);
    EXPECT_NE(
        CalcSymDirectTxDstBase(overLen, HcclCMDType::HCCL_CMD_ALLTOALLV, kTargetRank, kMyAlgRank, kDtypeSize, base),
        HcclResult::HCCL_SUCCESS);

    // 偏移越界：1000 > 999
    auto overOff = MakeParams(999, 1024, 0, 500, 0);
    EXPECT_NE(
        CalcSymDirectTxDstBase(overOff, HcclCMDType::HCCL_CMD_ALLTOALLV, kTargetRank, kMyAlgRank, kDtypeSize, base),
        HcclResult::HCCL_SUCCESS);
}

// output 界（窗口 size 生效）：起点 + 长度恰好贴边通过，偏移或长度越 1 字节失败
TEST(AllToAllSymDirectPeerOutputSliceTest, RemoteWindowSize)
{
    auto p = MakeParams(1024, 1024, 100, 0, 0);
    // 起点 600 + 100 = 700，长度 200，700 + 200 = 900，恰好等于窗口 size
    EXPECT_EQ(
        CheckSymDirectPeerOutputSlice(p, HcclCMDType::HCCL_CMD_ALLTOALLV, kTargetRank, 600, 100, 200, 900),
        HcclResult::HCCL_SUCCESS);
    // 长度越界：700 + 201 = 901 > 900
    EXPECT_NE(
        CheckSymDirectPeerOutputSlice(p, HcclCMDType::HCCL_CMD_ALLTOALLV, kTargetRank, 600, 100, 201, 900),
        HcclResult::HCCL_SUCCESS);
    // 偏移越界：起点 901 > 900
    EXPECT_NE(
        CheckSymDirectPeerOutputSlice(p, HcclCMDType::HCCL_CMD_ALLTOALLV, kTargetRank, 600, 301, 200, 900),
        HcclResult::HCCL_SUCCESS);
}

// output 界（窗口未填 size 恒 0）：回退本地 outputSize 判界
TEST(AllToAllSymDirectPeerOutputSliceTest, LocalOutputSizeFallback)
{
    auto p = MakeParams(1024, 900, 100, 0, 0);
    // 起点 700 + 长度 200 = 900，恰好等于本地 outputSize
    EXPECT_EQ(
        CheckSymDirectPeerOutputSlice(p, HcclCMDType::HCCL_CMD_ALLTOALLV, kTargetRank, 600, 100, 200, 0),
        HcclResult::HCCL_SUCCESS);
    // 长度越界：700 + 201 = 901 > 900
    EXPECT_NE(
        CheckSymDirectPeerOutputSlice(p, HcclCMDType::HCCL_CMD_ALLTOALLV, kTargetRank, 600, 100, 201, 0),
        HcclResult::HCCL_SUCCESS);
}

// VC 不校验 output 界：偏移与上界同源于矩阵，本地推不出独立上界，明显越界也放行
TEST(AllToAllSymDirectPeerOutputSliceTest, VCSkipCheck)
{
    auto p = MakeParams(1024, 0, 100, 0, 0);
    EXPECT_EQ(
        CheckSymDirectPeerOutputSlice(p, HcclCMDType::HCCL_CMD_ALLTOALLVC, kTargetRank, 600, 300, 999999, 0),
        HcclResult::HCCL_SUCCESS);
}
