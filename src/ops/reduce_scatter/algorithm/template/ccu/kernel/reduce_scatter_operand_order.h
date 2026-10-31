/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_REDUCE_SCATTER_OPERAND_ORDER_H
#define HCCL_REDUCE_SCATTER_OPERAND_ORDER_H

#include <cstdint>
#include <vector>

namespace ops_hccl {

struct ReduceScatterOperandPlan {
    std::vector<uint32_t> remoteRanks;
    uint32_t localRankSlot;
};

inline ReduceScatterOperandPlan BuildReduceScatterOperandPlan(uint32_t rankSize, uint32_t rankId, bool orderPreserved)
{
    ReduceScatterOperandPlan plan;
    plan.localRankSlot = orderPreserved ? rankId : (rankSize > 0 ? rankSize - 1 : 0);
    if (rankSize > 0) {
        plan.remoteRanks.reserve(rankSize - 1);
    }
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        if (rank != rankId) {
            plan.remoteRanks.push_back(rank);
        }
    }
    return plan;
}

} // namespace ops_hccl

#endif // HCCL_REDUCE_SCATTER_OPERAND_ORDER_H
