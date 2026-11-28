/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 注册门控缺符号降级 UT: 同一文件编入两个目标 —
//   op_graph_utils_test    : 桩提供两接口 → 门控通过
//   op_graph_reg_gate_test : 隐藏桩定义 → 弱引用为 null → 门控拒绝 (降级路径可断言)
// Soc/GE 门控依赖运行环境 (HAL dlopen / GE 符号), 不在本 UT 范围。
#include <gtest/gtest.h>

#include "hccl_custom_op.h"

#ifdef HCCL_UT_HIDE_HCOMM_RES_APIS
TEST(HcclRegGateTest, MissingHcommResApisBlocksRegistration) { EXPECT_FALSE(hccl::HcclUnfoldStreamResApiAvailable()); }
#elif defined(HCCL_UT_HIDE_DEDICATED_ACQUIRE)
// 仅缺 HcclDedicatedThreadAcquire (9.2.0_beta.2 引入; 9.1.x toolkit 真实形态):
// 门控语义为任一接口缺席即 false — 该用例区分 && 与误写 || (|| 时 ThreadResGetInfo 在位会放行)
TEST(HcclRegGateTest, MissingOnlyDedicatedAcquireBlocksRegistration)
{
    EXPECT_FALSE(hccl::HcclUnfoldStreamResApiAvailable());
}
#elif defined(HCCL_UT_HIDE_THREAD_RES_INFO)
// 仅缺 HcclThreadResGetInfo (9.1.0_beta.1 引入): 同上, 单接口缺席必须 false
TEST(HcclRegGateTest, MissingOnlyThreadResGetInfoBlocksRegistration)
{
    EXPECT_FALSE(hccl::HcclUnfoldStreamResApiAvailable());
}
#else
TEST(HcclRegGateTest, PresentHcommResApisAllowsRegistration) { EXPECT_TRUE(hccl::HcclUnfoldStreamResApiAvailable()); }
#endif
