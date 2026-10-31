/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_TEMPLATE_SELECTOR_TEST_COST_MODEL_H
#define AIV_TEMPLATE_SELECTOR_TEST_COST_MODEL_H

#include "alg_param.h"

namespace ops_hccl {
class CostModelManager {
public:
    static CostModelManager* Global()
    {
        static CostModelManager instance;
        return &instance;
    }

    void CalcMeshParam(float, CommTopo, int, u32, float& value, bool) { value = 0.0f; }
    void CalcLocalCopyParams(float, EngineType, float& value) { value = 0.0f; }
    void CalcLocalReduceParams(float, EngineType, float& value) { value = 0.0f; }
    void CalcLatencyParams(int, EngineType, float& value) { value = 0.0f; }
    void CalcLaunchParams(int, EngineType, float& value) { value = 0.0f; }
};
} // namespace ops_hccl

#endif // AIV_TEMPLATE_SELECTOR_TEST_COST_MODEL_H
