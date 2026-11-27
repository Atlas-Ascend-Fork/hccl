/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_MESH_GROUP_H
#define ALLTOALL_MESH_GROUP_H

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ops_hccl {
const uint32_t ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE = 16;

// Pair both directions at the same distance so both endpoints select each other in the same round.
// Rank-list indices, rather than user rank IDs, define the circular order.
inline std::vector<uint32_t> CalcAlltoAllMeshGroup(
    const std::vector<uint32_t>& ranks, uint32_t myIndex, uint32_t roundIdx,
    const std::vector<uint32_t>* selectedRanks = nullptr)
{
    const uint64_t rankSize = ranks.size();
    if (rankSize <= 1 || myIndex >= rankSize) {
        return {};
    }
    const uint64_t concurrentSize = std::min<uint64_t>(ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE, rankSize - 1);
    const uint64_t pairsPerRound = (concurrentSize + 1) / 2;
    const uint64_t firstDistance = static_cast<uint64_t>(roundIdx) * pairsPerRound + 1;
    const uint64_t endDistance = std::min(firstDistance + pairsPerRound, rankSize / 2 + 1);
    std::vector<uint32_t> peers;
    for (uint64_t distance = firstDistance; distance < endDistance; ++distance) {
        const uint64_t left = (myIndex + rankSize - distance) % rankSize;
        const uint64_t right = (myIndex + distance) % rankSize;
        peers.push_back(ranks[left]);
        if (left != right) {
            peers.push_back(ranks[right]);
        }
    }
    if (selectedRanks != nullptr) {
        peers.erase(
            std::remove_if(
                peers.begin(), peers.end(),
                [selectedRanks](uint32_t peer) {
                    return std::find(selectedRanks->begin(), selectedRanks->end(), peer) == selectedRanks->end();
                }),
            peers.end());
    }
    return peers;
}
} // namespace ops_hccl

#endif // ALLTOALL_MESH_GROUP_H
