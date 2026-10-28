/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "adapter_acl.h"
#include "hccl_aiv_utils.h"

namespace ops_hccl {
// GetAlgResWithEngine同时引用非CCU分支；这些依赖仅用于链接，CCU用例不应调用。
HcclResult haclrtMemcpy(void*, size_t, const void*, size_t, aclrtMemcpyKind)
{
    ADD_FAILURE() << "CCU资源申请用例不应调用haclrtMemcpy";
    return HCCL_E_NOT_SUPPORT;
}

HcclResult haclrtMemset(void*, size_t, int32_t, size_t)
{
    ADD_FAILURE() << "CCU资源申请用例不应调用haclrtMemset";
    return HCCL_E_NOT_SUPPORT;
}

HcclResult ClearAivTagCb(HcclComm, HcclCommStatePhase, void*)
{
    ADD_FAILURE() << "CCU资源申请用例不应调用ClearAivTagCb";
    return HCCL_E_NOT_SUPPORT;
}
} // namespace ops_hccl
