/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// V 变体参数解析链 UT: ReadVCountPairWithCheck / GetVRankSizeByGroup / ConvertVCountsToElementUnit /
// CalcOtherDimsFromShape / CrackCleaner 拷贝路径 — gert::Tensor 真实构造
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "exe_graph/runtime/runtime_tensor.h"
#include "graph/types.h"
#include "hccl/hcom.h"
#include "hccl_op_utils.h"
#include "hccl_custom_op.h"

using namespace hccl;
using gert::StorageFormat;
using gert::StorageShape;
using gert::Tensor;

namespace {
Tensor MakeHostTensor(std::initializer_list<int64_t> shape, ge::DataType dtype, const void* buf)
{
    return Tensor(
        StorageShape(shape, shape), StorageFormat(), gert::kOnHost, dtype,
        reinterpret_cast<gert::TensorAddress>(const_cast<void*>(buf)));
}
} // namespace

// 桩层可注入项 (op_deps_stub.cc 定义): MallocWorkSpace 返回缓冲 / aclrt 拷贝返回码
extern void* g_stubMallocWsRet;
extern aclError g_stubAclMemcpyRet;
extern aclError g_stubAclMemcpyAsyncRet;

// ---------------------------------------------------------------------------
// ReadVCountPairWithCheck: 双张量 + 长度契约
// ---------------------------------------------------------------------------
TEST(ReadVCountPairWithCheckTest, ValidPairPasses)
{
    const uint64_t count[] = {5};
    const uint64_t counts[] = {10, 20, 30, 40};
    Tensor countT = MakeHostTensor({1}, ge::DT_INT64, count);
    Tensor countsT = MakeHostTensor({4}, ge::DT_INT64, counts);
    std::vector<uint64_t> rawCount;
    std::vector<uint64_t> rawCounts;
    ASSERT_EQ(
        ge::GRAPH_SUCCESS,
        ReadVCountPairWithCheck(&countT, &countsT, "ut", "count", "counts", 4, rawCount, rawCounts, nullptr));
    ASSERT_EQ(1U, rawCount.size());
    EXPECT_EQ(5U, rawCount[0]);
    ASSERT_EQ(4U, rawCounts.size());
    EXPECT_EQ(30U, rawCounts[2]);
}

TEST(ReadVCountPairWithCheckTest, CountsShorterThanRankSizeFails)
{
    const uint64_t count[] = {5};
    const uint64_t counts[] = {10, 20}; // 只有 2 个, rankSize=4
    Tensor countT = MakeHostTensor({1}, ge::DT_INT64, count);
    Tensor countsT = MakeHostTensor({2}, ge::DT_INT64, counts);
    std::vector<uint64_t> rawCount;
    std::vector<uint64_t> rawCounts;
    EXPECT_EQ(
        ge::GRAPH_FAILED,
        ReadVCountPairWithCheck(&countT, &countsT, "ut", "count", "counts", 4, rawCount, rawCounts, nullptr));
}

TEST(ReadVCountPairWithCheckTest, CountNullFails)
{
    const uint64_t counts[] = {10, 20, 30, 40};
    Tensor countsT = MakeHostTensor({4}, ge::DT_INT64, counts);
    std::vector<uint64_t> rawCount;
    std::vector<uint64_t> rawCounts;
    EXPECT_EQ(
        ge::GRAPH_FAILED,
        ReadVCountPairWithCheck(nullptr, &countsT, "ut", "count", "counts", 4, rawCount, rawCounts, nullptr));
}

// ---------------------------------------------------------------------------
// ConvertVCountsToElementUnit: 溢出 + 长度 + 零值 (c431149f 补的校验)
// ---------------------------------------------------------------------------
TEST(ConvertVCountsToElementUnitTest, NormalCase)
{
    const std::vector<uint64_t> raw = {1, 2, 3};
    const std::vector<uint64_t> displs = {0, 5, 10};
    std::vector<uint64_t> counts, disp;
    ASSERT_EQ(ge::GRAPH_SUCCESS, ConvertVCountsToElementUnit(raw, displs, 4, counts, disp));
    ASSERT_EQ(3U, counts.size());
    EXPECT_EQ(4U, counts[0]);
    EXPECT_EQ(12U, counts[2]);
    EXPECT_EQ(40U, disp[2]);
}

TEST(ConvertVCountsToElementUnitTest, OverflowFails)
{
    const std::vector<uint64_t> raw = {UINT64_MAX / 2 + 1};
    const std::vector<uint64_t> displs = {0};
    std::vector<uint64_t> counts, disp;
    EXPECT_EQ(ge::GRAPH_FAILED, ConvertVCountsToElementUnit(raw, displs, 2, counts, disp));
}

TEST(ConvertVCountsToElementUnitTest, DisplsOverflowFails)
{
    const std::vector<uint64_t> raw = {1};
    const std::vector<uint64_t> displs = {UINT64_MAX};
    std::vector<uint64_t> counts, disp;
    EXPECT_EQ(ge::GRAPH_FAILED, ConvertVCountsToElementUnit(raw, displs, 2, counts, disp));
}

TEST(ConvertVCountsToElementUnitTest, ZeroOtherDimsFails)
{
    const std::vector<uint64_t> raw = {1};
    const std::vector<uint64_t> displs = {0};
    std::vector<uint64_t> counts, disp;
    EXPECT_EQ(ge::GRAPH_FAILED, ConvertVCountsToElementUnit(raw, displs, 0, counts, disp));
}

TEST(ConvertVCountsToElementUnitTest, LengthMismatchFails)
{
    const std::vector<uint64_t> raw = {1, 2, 3};
    const std::vector<uint64_t> displs = {0, 5}; // 比 raw 短
    std::vector<uint64_t> counts, disp;
    EXPECT_EQ(ge::GRAPH_FAILED, ConvertVCountsToElementUnit(raw, displs, 4, counts, disp));
}

// ---------------------------------------------------------------------------
// CalcOtherDimsFromShape
// ---------------------------------------------------------------------------
TEST(CalcOtherDimsFromShapeTest, MultiDimProduct)
{
    // shape [2,3,4] → otherDims = 3*4 = 12
    gert::Shape shape({2, 3, 4});
    uint64_t otherDims = 0;
    ASSERT_EQ(ge::GRAPH_SUCCESS, CalcOtherDimsFromShape(shape, "ut", otherDims));
    EXPECT_EQ(12U, otherDims);
}

TEST(CalcOtherDimsShapeTest, OneDimIsOne)
{
    gert::Shape shape({6});
    uint64_t otherDims = 0;
    ASSERT_EQ(ge::GRAPH_SUCCESS, CalcOtherDimsFromShape(shape, "ut", otherDims));
    EXPECT_EQ(1U, otherDims);
}

// ---------------------------------------------------------------------------
// CrackCleaner::Clean: 拷贝路径 (非平凡分支 — 需 MallocWorkSpace 桩 + crack 参数)
// ---------------------------------------------------------------------------
TEST(CrackCleanerTest, NullCtxMallocFailsReturnsGraphFailed)
{
    HcclOpState st;
    st.opType = "ut";
    CrackCleaner cleaner;
    // MallocWorkSpace 桩返回非空静态缓冲即可 (真实拷贝到桩缓冲)
    uint8_t dummyBuf[16] = {};
    std::vector<void*> addrs = {reinterpret_cast<void*>(dummyBuf)};
    std::vector<uint64_t> sizes = {16};
    // Clean 内部: MallocWorkSpace(st.ctx) — st.ctx 是 null, MallocWorkSpace 桩返回 nullptr
    // 所以这个用例测的是 ctx null → MallocWorkSpace 返回 null → GRAPH_FAILED 路径
    // 真正的拷贝路径需要构造 EagerOpExecutionContext, 超出纯函数 UT 范围
    // 这里验证长度一致但 ctx null 时的防御
    EXPECT_EQ(ge::GRAPH_FAILED, cleaner.Clean(st, addrs, sizes));
}

TEST(CrackCleanerTest, MultipleCracksCorrectSizes)
{
    // 多 crack 真实拷贝路径: MallocWorkSpace 桩返回有效缓冲 → H2D 零化 → D2D 清零,
    // 逐 crack 断言实际内容被清零 (尺寸各自生效)
    static uint8_t zeroBuf[32] = {};
    g_stubMallocWsRet = zeroBuf;

    HcclOpState st;
    st.opType = "ut";
    st.ctx = reinterpret_cast<gert::EagerOpExecutionContext*>(0x10); // 任意非 null 指针 (触发桩)
    st.stream = reinterpret_cast<void*>(0x20); // 真源 HcclRtMemAsyncCopy 入口判 stream 非空
    CrackCleaner cleaner;

    uint8_t crackA[8];
    uint8_t crackB[16];
    memset(crackA, 0xFF, sizeof(crackA));
    memset(crackB, 0xFF, sizeof(crackB));

    std::vector<void*> addrs = {reinterpret_cast<void*>(crackA), reinterpret_cast<void*>(crackB)};
    std::vector<uint64_t> sizes = {sizeof(crackA), sizeof(crackB)};
    ASSERT_EQ(ge::GRAPH_SUCCESS, cleaner.Clean(st, addrs, sizes));
    // 清零结果断言: 两个 crack 缓冲实际内容全 0 (D2D 桩为真 memcpy, 零源已 H2D 写入桩缓冲)
    for (size_t i = 0; i < sizeof(crackA); i++) {
        EXPECT_EQ(0U, crackA[i]) << "crackA[" << i << "]";
    }
    for (size_t i = 0; i < sizeof(crackB); i++) {
        EXPECT_EQ(0U, crackB[i]) << "crackB[" << i << "]";
    }
    g_stubMallocWsRet = nullptr; // 本用例不在 CrackCleanerCopyTest fixture 内, 尾部手动复位防泄漏
}

// ---------------------------------------------------------------------------
// CrackCleaner 真实拷贝路径: 桩返回码/缓冲内容全部可注入断言
// ---------------------------------------------------------------------------

class CrackCleanerCopyTest : public ::testing::Test {
protected:
    void TearDown() override
    {
        g_stubMallocWsRet = nullptr;
        g_stubAclMemcpyRet = ACL_SUCCESS;
        g_stubAclMemcpyAsyncRet = ACL_SUCCESS;
    }
};

TEST_F(CrackCleanerCopyTest, RealCopyPathZeroesCrackData)
{
    // MallocWorkSpace 桩返回有效缓冲 → HcclRtMemSyncCopy(真源) 经 aclrtMemcpy 桩将零拷入
    // → HcclRtMemAsyncCopy(真源) 经 aclrtMemcpyAsync 桩真实搬运零 → crack 实际被清零
    static uint8_t crackZeroBuf[64] = {};
    g_stubMallocWsRet = crackZeroBuf;

    HcclOpState st;
    st.opType = "ut";
    st.ctx = reinterpret_cast<gert::EagerOpExecutionContext*>(0x10); // 任意非 null 指针 (触发桩)
    st.stream = reinterpret_cast<void*>(0x20);                       // 真源入口判 stream 非空
    CrackCleaner cleaner;

    uint8_t crackData[8];
    memset(crackData, 0xFF, sizeof(crackData)); // 预填 0xFF

    std::vector<void*> addrs = {reinterpret_cast<void*>(crackData)};
    std::vector<uint64_t> sizes = {sizeof(crackData)};
    ASSERT_EQ(ge::GRAPH_SUCCESS, cleaner.Clean(st, addrs, sizes));
    // 清零结果断言: 8 字节全部实际为 0 (H2D 零化 → D2D 真实搬运全链路)
    for (size_t i = 0; i < sizeof(crackData); i++) {
        EXPECT_EQ(0U, crackData[i]) << "crackData[" << i << "]";
    }
}

TEST_F(CrackCleanerCopyTest, MallocWorkSpaceFailureReturnsGraphFailed)
{
    g_stubMallocWsRet = nullptr; // MallocWorkSpace 桩返回 null → 分配失败
    HcclOpState st;
    st.opType = "ut";
    st.ctx = reinterpret_cast<gert::EagerOpExecutionContext*>(0x10);
    CrackCleaner cleaner;
    std::vector<void*> addrs = {reinterpret_cast<void*>(0x2000)};
    std::vector<uint64_t> sizes = {16};
    EXPECT_EQ(ge::GRAPH_FAILED, cleaner.Clean(st, addrs, sizes));
}

TEST_F(CrackCleanerCopyTest, MaxSizeZeroSkipsAllocation)
{
    g_stubMallocWsRet = nullptr; // 即使 MallocWorkSpace 失败也不应走到 (maxSize==0 提前 success)
    HcclOpState st;
    st.opType = "ut";
    st.ctx = nullptr; // maxSize==0 → 不触碰 ctx
    CrackCleaner cleaner;
    std::vector<void*> addrs = {reinterpret_cast<void*>(0x2000)};
    std::vector<uint64_t> sizes = {0};
    EXPECT_EQ(ge::GRAPH_SUCCESS, cleaner.Clean(st, addrs, sizes));
}
