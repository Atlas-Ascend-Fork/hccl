/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// alltoall_nda_selector_test 目标的补充桩：
// - SelectorRegistry::Global/RegisterByOpType：alltoall_auto_selector.cc 末尾的
//   REGISTER_SELECTOR_BY_OPTYPE 静态注册在本 UT 中只要求"能注册成功"，不校验注册表内容。
// - HcclGetHcclBuffer 由 selector_engine/stub/selector_deps_stub.cc 统一提供，本文件不再重复定义，
//   避免同目标同时编入两个桩文件时链接期 multiple definition。
// 其余 auto_selector_base.cc 的依赖复用 selector_engine/stub/selector_deps_stub.cc。

#include "selector_registry.h"
#include "hccl_res.h"

namespace ops_hccl {

SelectorRegistry* SelectorRegistry::Global()
{
    static SelectorRegistry registry;
    return &registry;
}

HcclResult SelectorRegistry::RegisterByOpType(const HcclCMDType opType, u32 priority, AutoSelectorBase* selector)
{
    (void)opType;
    (void)priority;
    (void)selector;
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
