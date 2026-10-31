/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_TEMPLATE_SELECTOR_TEST_DATA_TRANS_WRAPPER_H
#define AIV_TEMPLATE_SELECTOR_TEST_DATA_TRANS_WRAPPER_H

#include "alg_param.h"

namespace ops_hccl {
namespace aiv_selector_test {
    inline std::vector<std::vector<u32>> channelRequestRanks;
    inline u32 meshChannelRequests = 0;
    inline u32 meshClosChannelRequests = 0;
} // namespace aiv_selector_test

inline HcclResult CalcChannelRequestMeshClosMultiJetty(
    HcclComm, const OpParam&, const TopoInfoWithNetLayerDetails*, const std::vector<std::vector<u32>>& subCommRanks,
    std::vector<HcclChannelDesc>&, bool)
{
    aiv_selector_test::channelRequestRanks = subCommRanks;
    ++aiv_selector_test::meshClosChannelRequests;
    return HCCL_SUCCESS;
}

inline HcclResult CalcChannelRequestMesh1D(
    HcclComm, const OpParam&, const TopoInfoWithNetLayerDetails*, const std::vector<std::vector<u32>>& subCommRanks,
    std::vector<HcclChannelDesc>&)
{
    aiv_selector_test::channelRequestRanks = subCommRanks;
    ++aiv_selector_test::meshChannelRequests;
    return HCCL_SUCCESS;
}
} // namespace ops_hccl

#endif // AIV_TEMPLATE_SELECTOR_TEST_DATA_TRANS_WRAPPER_H
