/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_1D_H
#define HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH_1D_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "../ccu_temp_all_to_all_v_mesh_1D.h"

namespace ops_hccl {

struct CcuKernelArgAlltoAllVMesh1D : CcuKernelArgBase {
    uint64_t rankSize;
    uint32_t rankId;
    bool loadFromMem;
    OpParam opParam;
    std::vector<std::vector<uint32_t>> subCommRanks;
};

struct A2AsingleSendRecvInfo {
    ccu::Variable tailSize;
    ccu::Variable loopNum;
    ccu::Variable sendOffset;
    ccu::Variable recvOffset;
    GroupOpSizeVars tailGoSize;
};

struct AlltoAllVMesh1DContext : CcuKernelCtxBase {
    const CcuKernelArgAlltoAllVMesh1D* arg;

    uint64_t rankSize{0};
    uint32_t rankId{0};
    HcclDataType dataType;
    HcclDataType outputDataType;
    ccu::Variable repeatNum;
    std::vector<ChannelHandle> channels;
    std::vector<ccu::Variable> input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable sliceSize;
    ccu::Variable srcStride;
    ccu::Variable srcOffset;
    ccu::Variable dstOffset;
    ccu::Variable groupOpSize;
    ccu::Variable a2avXnAddr;
    ccu::Variable xnLength;
    ccu::Variable completedRankCount;
    ccu::Variable xnMaxTransportSize;
    GroupOpSizeVars xnMaxTransportGoSize;
    std::vector<A2AsingleSendRecvInfo> sendRecvInfo;
    ccu::LocalAddr myDst;
    std::vector<ccu::RemoteAddr> dst;
    std::vector<ccu::LocalAddr> src;
    // 单个 ccu::Event 的 16 位 mask 最多支持 16 个 rank 信号，改用 CcuEventGroup 支持 rankSize > 16
    CcuEventGroup eventGroup;
    ccu::Variable xnConst1;
    bool loadFromMem = false;
};

CcuResult CcuAlltoAllVMesh1DKernel(CcuKernelArg arg);
} // namespace ops_hccl
#endif // HCCL_CCU_KERNEL_ALL_TO_ALL_MESH_1D_H
