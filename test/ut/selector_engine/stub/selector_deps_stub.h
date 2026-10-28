/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELECTOR_ENGINE_UT_SELECTOR_DEPS_STUB_H
#define SELECTOR_ENGINE_UT_SELECTOR_DEPS_STUB_H

#include <string>
#include <vector>

#include "alg_param.h"

namespace hccl_ut {
// CostModelManager::GenerateCostModel 捕获到的候选引擎列表, 即 GetEnginePriority 的实际输出
const std::vector<ops_hccl::OpExecuteConfig>& LastCandidateEngines();

// CostModelManager::GenerateCostModel 捕获到的候选前缀列表, 即 CandidateEnginesToPrefixes 的实际输出
const std::vector<std::string>& LastCandidatePrefixes();

// 注入 HcclTunerInit 返回值: 基模式用例注入哨兵错误, 证明 Run 越过 only 报错分支进入正常路径
void SetTunerInitResult(HcclResult ret);

// 注入"已注册算法"全集, GenerateCostModel 按候选前缀过滤后生成 count>0 模型; 空集保持空模型走负向路径
void SetCostModelAlgos(const std::vector<std::string>& algos);

void ResetSelectorStubState();
} // namespace hccl_ut

#endif // SELECTOR_ENGINE_UT_SELECTOR_DEPS_STUB_H
