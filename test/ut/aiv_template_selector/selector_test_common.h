/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_TEMPLATE_SELECTOR_TEST_COMMON_H
#define AIV_TEMPLATE_SELECTOR_TEST_COMMON_H

#include <limits>
#include <vector>
#include "gtest/gtest.h"
#include "alg_data_trans_wrapper.h"
#include "alg_param.h"
#include "hccl_aiv_utils.h"
#include "order_preserved_common.h"

namespace ops_hccl {
namespace aiv_selector_test {

    class AivTemplateSelectorTest : public ::testing::Test {
    protected:
        void SetUp() override
        {
            param_.opMode = OpMode::OPBASE;
            param_.DataDes.count = 128;
            topo_.level0PcieMix = true;
            topo_.userRankSize = 8;
            sendCount_ = 0;
            kernelLaunches = 0;
            launchedArgs = AivOpArgs{};
            channelRequestRanks.clear();
            meshChannelRequests = 0;
            meshClosChannelRequests = 0;
            strictOrderRequired = false;
            strictOrderQueries = 0;
        }

        OpParam param_;
        TopoInfoWithNetLayerDetails topo_;
        const std::vector<std::vector<u32>> ranks_ = {{0, 1, 2, 3, 4, 5, 6, 7}};
        u64 sendCount_ = 0;
    };

} // namespace aiv_selector_test
} // namespace ops_hccl

#endif // AIV_TEMPLATE_SELECTOR_TEST_COMMON_H
