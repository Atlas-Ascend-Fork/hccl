/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_DPU_TOPOLOGY_H
#define ALLTOALL_DPU_TOPOLOGY_H

#include <algorithm>
#include "alg_param.h"

namespace ops_hccl {
// Select the outermost local DEVICE scope at or below the pod layer. HOST/rail scopes must not enlarge it.
inline HcclResult GetAlltoAllLocalPodRanks(const TopoInfoWithNetLayerDetails& topo, std::vector<u32>& ranks)
{
    ranks = {topo.userRank};
    const PhysicalLevelInfo* selected = nullptr;
    for (const auto& level : topo.physicalLevels) {
        if (!level.hasTopoInst || level.locType != EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE
            || level.ref.netLayer > POD_NET_LAYER) {
            continue;
        }
        if (level.localRanks.empty()
            || !std::binary_search(level.localRanks.begin(), level.localRanks.end(), topo.userRank)) {
            return HCCL_E_PARA;
        }
        if (selected == nullptr || level.localRanks.size() > selected->localRanks.size()) {
            selected = &level;
        }
    }
    if (selected != nullptr) {
        ranks = selected->localRanks;
    }
    // All local DEVICE ranges must belong to the selected pod, including same-size alternative topologies.
    for (const auto& level : topo.physicalLevels) {
        if (level.hasTopoInst && level.locType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE
            && level.ref.netLayer <= POD_NET_LAYER
            && !std::includes(ranks.begin(), ranks.end(), level.localRanks.begin(), level.localRanks.end())) {
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl

#endif // ALLTOALL_DPU_TOPOLOGY_H
