/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_TEMPLATE_SELECTOR_TEST_CONFIG_LOG_H
#define AIV_TEMPLATE_SELECTOR_TEST_CONFIG_LOG_H

namespace ops_hccl {
namespace aiv_selector_test {
    inline void Log(...) {}
} // namespace aiv_selector_test
} // namespace ops_hccl

#define HCCL_ALG 0
#define HCCL_CONFIG_INFO(...) ops_hccl::aiv_selector_test::Log(__VA_ARGS__)
#define HCCL_DEBUG(...) ops_hccl::aiv_selector_test::Log(__VA_ARGS__)
#define HCCL_INFO(...) ops_hccl::aiv_selector_test::Log(__VA_ARGS__)
#define HCCL_WARNING(...) ops_hccl::aiv_selector_test::Log(__VA_ARGS__)
#define HCCL_ERROR(...) ops_hccl::aiv_selector_test::Log(__VA_ARGS__)
#define CHK_RET(call)                           \
    do {                                        \
        const auto result = (call);             \
        if (result != ops_hccl::HCCL_SUCCESS) { \
            return result;                      \
        }                                       \
    } while (false)
#define CHK_PRT_RET(condition, log, result) \
    do {                                    \
        if (condition) {                    \
            log;                            \
            return result;                  \
        }                                   \
    } while (false)

#endif // AIV_TEMPLATE_SELECTOR_TEST_CONFIG_LOG_H
