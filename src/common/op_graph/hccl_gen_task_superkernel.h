/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_gen_task_superkernel.h
 * \brief
 */

#ifndef CANN_HCCL_GEN_TASK_SUPERKERNEL_H_
#define CANN_HCCL_GEN_TASK_SUPERKERNEL_H_

#include <string>
#include <vector>
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "graph/arg_desc_info.h"

namespace ops {
// A5 及以后代际判断（纯函数，便于 UT 覆盖）：Ascend950/Ascend960/... 及 A5 同代 Ascend350 系列（含带后缀型号）返回
// true。
bool IsA5OrLaterShortSocVersion(const std::string& shortSocVersion);

// 构造 HCCL SuperKernel 子算子的 args format：布局固定为 hidden input 在前，随后 input、output。
ge::AscendString BuildSuperKernelArgsFormat();

ge::Status
HcclSuperKernelGenTaskCallback(const gert::ExeResGenerationContext* context, std::vector<std::vector<uint8_t>>& tasks);

} // namespace ops
#endif // CANN_HCCL_GEN_TASK_SUPERKERNEL_H_
