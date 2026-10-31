/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_REDUCE_SCATTER_BATCH_INVARIANT_H
#define HCCL_REDUCE_SCATTER_BATCH_INVARIANT_H

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "alg_param.h"

namespace ops_hccl {

constexpr const char* CCU_MS_RS_SOLE_MESH_ALG_NAME = "CcuMSReduceScatterSoleMesh";

// 与 reduce_scatter_auto_selector.cc 的 MAX_RANK_NUM_FOR_CCU_MS_REDUCE_SCATTER、
// ins_v2_reduce_scatter_sole_executor.cc 的 MAX_RANK_NUM_FOR_REDUCE_MS_ALGO 同源于内核
// msInterleave 约束（size = channelCount + 1 须不超过 CCU_MS_INTERLEAVE），须同步调整。
constexpr u32 MAX_RANK_NUM_FOR_CCU_MS_REDUCE_SCATTER_STRICT = 8;

inline bool IsCcuMsReduceScatterSoleMeshAlg(const char* algName)
{
    return algName != nullptr && std::strcmp(algName, CCU_MS_RS_SOLE_MESH_ALG_NAME) == 0;
}

inline bool IsReduceScatterInputOutputOverlap(const OpParam& opParam)
{
    if (opParam.inputPtr == nullptr || opParam.outputPtr == nullptr || opParam.inputSize == 0
        || opParam.outputSize == 0) {
        return false;
    }

    uintptr_t inputStart = reinterpret_cast<uintptr_t>(opParam.inputPtr);
    uintptr_t outputStart = reinterpret_cast<uintptr_t>(opParam.outputPtr);
    uintptr_t inputEnd = inputStart + opParam.inputSize - 1;
    uintptr_t outputEnd = outputStart + opParam.outputSize - 1;
    return inputStart <= outputEnd && outputStart <= inputEnd;
}

inline bool HasFullMeshForBatchInvariantCcuMs(const TopoInfoWithNetLayerDetails* topoInfo)
{
    if (topoInfo == nullptr) {
        return false;
    }
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        return topoInfo->level0MeshType == Level0MeshType::SINGLE_DIE;
    }
    if (topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS || topoInfo->netLayerDetails.localNetInsSizeOfLayer.empty()
        || topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] != topoInfo->userRankSize
        || topoInfo->topoInstDetailsOfLayer.empty()) {
        return false;
    }

    const auto& rankNumForTopoType = topoInfo->topoInstDetailsOfLayer[0].rankNumForTopoType;
    auto meshRankNums = rankNumForTopoType.find(CommTopo::COMM_TOPO_1DMESH);
    return meshRankNums != rankNumForTopoType.end()
           && std::find(meshRankNums->second.begin(), meshRankNums->second.end(), topoInfo->userRankSize)
                  != meshRankNums->second.end();
}

inline bool
CanSelectBatchInvariantCcuMsReduceScatter(const OpParam& opParam, const TopoInfoWithNetLayerDetails* topoInfo)
{
    if (topoInfo == nullptr || opParam.opExecuteConfig != OpExecuteConfig::CCU_MS || topoInfo->hostDpuOnly
        || topoInfo->topoLevelNums != 1 || topoInfo->userRankSize > MAX_RANK_NUM_FOR_CCU_MS_REDUCE_SCATTER_STRICT
        || opParam.reduceType != HcclReduceOp::HCCL_REDUCE_SUM || IsReduceScatterInputOutputOverlap(opParam)) {
        return false;
    }

    HcclDataType dataType = opParam.DataDes.dataType;
    if (dataType != HcclDataType::HCCL_DATA_TYPE_FP16 && dataType != HcclDataType::HCCL_DATA_TYPE_FP32
        && dataType != HcclDataType::HCCL_DATA_TYPE_BFP16) {
        return false;
    }

    return HasFullMeshForBatchInvariantCcuMs(topoInfo);
}

} // namespace ops_hccl

#endif // HCCL_REDUCE_SCATTER_BATCH_INVARIANT_H
