/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_MESH_1D_MEM2MEM
#define HCCL_CCU_KERNEL_REDUCE_MESH_1D_MEM2MEM

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

// KernelRun、LoadArgs 和 FastLaunch 共用的 taskArgs 头部索引。
constexpr uint32_t TASKARG_INPUT_ADDR = 0;
constexpr uint32_t TASKARG_OUTPUT_ADDR = 1;
constexpr uint32_t TASKARG_SCRATCH_ADDR = 2;
constexpr uint32_t TASKARG_TOKEN = 3;
constexpr uint32_t TASKARG_HEAD_SIZE = TASKARG_TOKEN + 1;
// FastLaunch 通过此索引读取缓存的 slice 大小。
constexpr uint32_t TASKARG_NORMAL_SLICE_SIZE = TASKARG_HEAD_SIZE + 2;

struct CcuKernelArgReduceMesh1DMem2Mem : CcuKernelArgBase {
    uint64_t rankSize;
    uint32_t rankId;
    uint32_t rootId;
    OpParam opParam;
    std::vector<std::vector<uint32_t>> subCommRanks;
};

struct ReduceMesh1DMem2MemContext : CcuKernelCtxBase {
    const CcuKernelArgReduceMesh1DMem2Mem* arg;

    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;

    std::vector<ccu::Variable> input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;

    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable normalSliceSize;
    ccu::Variable repeatNumVar;
    ccu::Variable flag;
    ccu::Variable scratchAddr;
    ccu::Event event;

    ccu::LocalAddr myInputAddr;
    ccu::RemoteAddr remoteInputAddr;
    ccu::LocalAddr dstAddr;
    ccu::LocalAddr scratchBase;
    ccu::LocalAddr scratchSlot;
    ccu::LocalAddr readDst;

    // 每个 rank 对应一个归约源；root 槽指向本卡 input。
    std::vector<ccu::LocalAddr> reduceSrc;

    GroupOpSizeVars reduceGoSize;
};

CcuResult CcuReduceMesh1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl
#endif // HCCL_CCU_KERNEL_REDUCE_MESH_1D_MEM2MEM
