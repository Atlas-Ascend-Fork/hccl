/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_P2P_DFX_H
#define OPS_HCCL_P2P_DFX_H

#include <hccl/hccl_diag.h>
#include "common.h"

namespace ops_hccl_p2p {

// 填充 host/device 共用的 Dfx 算子信息
// 差异字段（device 侧的 cpuTsThread/cpuWaitAicpuNotifyIdx）由调用方自行补充
inline HcclResult FillDfxOpInfo(const OpParam& param, HcclDfxOpInfo& dfxOpInfo)
{
    int ret = sprintf_s(dfxOpInfo.algTag, sizeof(dfxOpInfo.algTag), "%s", param.tag);
    if (ret <= 0) {
        return HCCL_E_INTERNAL;
    }
    dfxOpInfo.opMode = 1; // 单算子模式，对应OpMode::OPBASE
    dfxOpInfo.opType = static_cast<uint32_t>(param.opType);
    dfxOpInfo.dataType = static_cast<uint32_t>(param.dataType);
    dfxOpInfo.dataCount = param.count;
    dfxOpInfo.engine = CommEngine::COMM_ENGINE_AICPU; // AICPU 通信引擎
    dfxOpInfo.inputMemAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    dfxOpInfo.inputMemSize = (param.inputPtr == nullptr) ? 0 : param.count * SIZE_TABLE[param.dataType];
    dfxOpInfo.outputMemAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    dfxOpInfo.outputMemSize = (param.outputPtr == nullptr) ? 0 : param.count * SIZE_TABLE[param.dataType];
    return HCCL_SUCCESS;
}

} // namespace ops_hccl_p2p

#endif // OPS_HCCL_P2P_DFX_H
