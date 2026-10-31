/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_TEMPLATE_SELECTOR_TEST_ORDER_PRESERVED_COMMON_H
#define AIV_TEMPLATE_SELECTOR_TEST_ORDER_PRESERVED_COMMON_H

#include "alg_param.h"

namespace ops_hccl {

namespace aiv_selector_test {
    // 保序判定由用例控制，并统计调用次数以验证门控短路顺序。
    inline bool strictOrderRequired = false;
    inline u32 strictOrderQueries = 0;
} // namespace aiv_selector_test

inline bool IsNeedStrictModeForOrderPreserved(const OpParam&, u32)
{
    ++aiv_selector_test::strictOrderQueries;
    return aiv_selector_test::strictOrderRequired;
}

} // namespace ops_hccl

#endif // AIV_TEMPLATE_SELECTOR_TEST_ORDER_PRESERVED_COMMON_H
