/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// hccl_rt_utils.cc 真源 UT: kind 翻译缺省分支 (E_PARA) / count==0 短路 (先于指针判空) /
// ACL 失败映射 (E_RUNTIME) / ADDR 不支持 (E_NOT_SUPPORT) — aclrt 桩可注入返回码, 真实搬运
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl_rt.h"
#include "hccl_rt_utils.h"

using namespace hccl;

// 桩层可配置返回码 (op_deps_stub.cc 定义)
extern aclError g_stubAclMemcpyRet;
extern aclError g_stubAclMemcpyAsyncRet;
extern aclError g_stubAclAddrCopyRet;

namespace {
// 每用例复位注入, 防跨用例泄漏
class RtUtilsTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        g_stubAclMemcpyRet = ACL_SUCCESS;
        g_stubAclMemcpyAsyncRet = ACL_SUCCESS;
        g_stubAclAddrCopyRet = ACL_SUCCESS;
    }
    void TearDown() override
    {
        g_stubAclMemcpyRet = ACL_SUCCESS;
        g_stubAclMemcpyAsyncRet = ACL_SUCCESS;
        g_stubAclAddrCopyRet = ACL_SUCCESS;
    }
};
} // namespace

// ---------------------------------------------------------------------------
// HcclRtMemSyncCopy: 缺省 kind → E_PARA (同步路径不做 ADDR 裸 D2D 降级)
// ---------------------------------------------------------------------------
TEST_F(RtUtilsTest, SyncCopyReservedKindReturnsPara)
{
    uint8_t dst[8] = {};
    uint8_t src[8] = {};
    EXPECT_EQ(
        HCCL_E_PARA,
        HcclRtMemSyncCopy(dst, sizeof(dst), src, sizeof(src), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_RESERVED));
}

TEST_F(RtUtilsTest, SyncCopyAddrKindRejectedBySyncPath)
{
    // ADDR 仅异步路径支持, 同步路径显式报不支持 (避免同一枚举两路径语义分叉)
    uint8_t dst[8] = {};
    uint8_t src[8] = {};
    EXPECT_EQ(
        HCCL_E_PARA,
        HcclRtMemSyncCopy(dst, sizeof(dst), src, sizeof(src), HcclRtMemcpyKind::HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE));
}

// ---------------------------------------------------------------------------
// count==0 短路: 先于数据指针判空 — (空指针, 0) 无操作成功
// ---------------------------------------------------------------------------
TEST_F(RtUtilsTest, SyncCopyZeroCountWithNullPtrsSucceeds)
{
    EXPECT_EQ(
        HCCL_SUCCESS,
        HcclRtMemSyncCopy(nullptr, 0, nullptr, 0, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE));
}

TEST_F(RtUtilsTest, AsyncCopyZeroCountWithNullPtrsSucceeds)
{
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x20);
    EXPECT_EQ(
        HCCL_SUCCESS,
        HcclRtMemAsyncCopy(nullptr, 0, nullptr, 0, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream));
}

// ---------------------------------------------------------------------------
// 参数防御: stream 判空最前 (与拷贝长度无关) / 非零 count 空指针 → E_PARA
// ---------------------------------------------------------------------------
TEST_F(RtUtilsTest, AsyncCopyNullStreamReturnsPara)
{
    uint8_t dst[8] = {};
    uint8_t src[8] = {};
    EXPECT_EQ(
        HCCL_E_PARA,
        HcclRtMemAsyncCopy(
            dst, sizeof(dst), src, sizeof(src), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, nullptr));
}

TEST_F(RtUtilsTest, AsyncCopyNullPtrWithNonZeroCountReturnsPara)
{
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x20);
    uint8_t dst[8] = {};
    EXPECT_EQ(
        HCCL_E_PARA, HcclRtMemAsyncCopy(
                         dst, sizeof(dst), nullptr, 8, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream));
}

// ---------------------------------------------------------------------------
// ACL 失败映射: 同步/异步非特征类失败 → HCCL_E_RUNTIME
// ---------------------------------------------------------------------------
TEST_F(RtUtilsTest, SyncCopyAclFailureReturnsRuntime)
{
    uint8_t dst[8] = {};
    uint8_t src[8] = {};
    g_stubAclMemcpyRet = ACL_ERROR_INVALID_PARAM;
    EXPECT_EQ(
        HCCL_E_RUNTIME,
        HcclRtMemSyncCopy(dst, sizeof(dst), src, sizeof(src), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_HOST));
}

TEST_F(RtUtilsTest, AsyncCopyAclFailureReturnsRuntime)
{
    uint8_t dst[8] = {};
    uint8_t src[8] = {};
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x20);
    g_stubAclMemcpyAsyncRet = ACL_ERROR_INVALID_PARAM;
    EXPECT_EQ(
        HCCL_E_RUNTIME,
        HcclRtMemAsyncCopy(
            dst, sizeof(dst), src, sizeof(src), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream));
}

// ---------------------------------------------------------------------------
// ADDR kind: FEATURE_NOT_SUPPORT → HCCL_E_NOT_SUPPORT (非 E_RUNTIME)
// ---------------------------------------------------------------------------
TEST_F(RtUtilsTest, AsyncCopyAddrKindNotSupport)
{
    // ADDR 语义: dst/src 为地址槽 (二级指针)
    uint8_t bufA[8] = {};
    uint8_t bufB[8] = {};
    void* dstSlot = bufA;
    const void* srcSlot = bufB;
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x20);
    g_stubAclAddrCopyRet = ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    EXPECT_EQ(
        HCCL_E_NOT_SUPPORT, HcclRtMemAsyncCopy(
                                &dstSlot, sizeof(bufA), &srcSlot, sizeof(bufA),
                                HcclRtMemcpyKind::HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, stream));
}

TEST_F(RtUtilsTest, AddrAsyncCopyWithOffsetNotSupport)
{
    uint8_t bufA[8] = {};
    uint8_t bufB[8] = {};
    void* dstSlot = bufA;
    const void* srcSlot = bufB;
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x20);
    g_stubAclAddrCopyRet = ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    EXPECT_EQ(
        HCCL_E_NOT_SUPPORT,
        HcclRtMemAddrAsyncCopyWithOffset(&dstSlot, sizeof(bufA), 0, &srcSlot, sizeof(bufA), 0, stream));
}

// ---------------------------------------------------------------------------
// 正常路径: 真实搬运断言 (桩为真 memcpy)
// ---------------------------------------------------------------------------
TEST_F(RtUtilsTest, SyncCopyHostToHostMovesBytes)
{
    uint8_t dst[8] = {};
    const uint8_t src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    ASSERT_EQ(
        HCCL_SUCCESS,
        HcclRtMemSyncCopy(dst, sizeof(dst), src, sizeof(src), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_HOST));
    EXPECT_EQ(0, memcmp(dst, src, sizeof(src)));
}

TEST_F(RtUtilsTest, AsyncCopyDeviceToDeviceMovesBytes)
{
    uint8_t dst[8] = {};
    const uint8_t src[8] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22};
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x20);
    ASSERT_EQ(
        HCCL_SUCCESS,
        HcclRtMemAsyncCopy(
            dst, sizeof(dst), src, sizeof(src), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream));
    EXPECT_EQ(0, memcmp(dst, src, sizeof(src)));
}

TEST_F(RtUtilsTest, AddrAsyncCopyWithOffsetDoubleIndirection)
{
    // 二级指针语义: 实际搬运 *dst+dstOff ← *src+srcOff
    uint8_t bufA[8] = {};
    uint8_t bufB[8] = {};
    memset(bufB, 0x5A, sizeof(bufB));
    void* dstSlot = bufA;
    const void* srcSlot = bufB;
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x20);
    ASSERT_EQ(
        HCCL_SUCCESS, HcclRtMemAddrAsyncCopyWithOffset(&dstSlot, sizeof(bufA), 0, &srcSlot, sizeof(bufA), 0, stream));
    EXPECT_EQ(0, memcmp(bufA, bufB, sizeof(bufA)));
}

TEST_F(RtUtilsTest, AddrAsyncCopyZeroCountWithNullSlotsSucceeds)
{
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x20);
    EXPECT_EQ(HCCL_SUCCESS, HcclRtMemAddrAsyncCopyWithOffset(nullptr, 0, 0, nullptr, 0, 0, stream));
}
