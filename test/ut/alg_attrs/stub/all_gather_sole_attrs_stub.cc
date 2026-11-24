/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// UT 桩：all_gather_sole_attrs_test 的最小链接符号集。
// 被桩函数仅在 executor attrs 注册 lambda 体内或 alg_parse.cc 的环境变量/HCCL_ALGO
// 解析路径中引用（生产定义位于 auto_selector_base.cc / op_common.cc / CANN 运行时，
// 未编入本 UT 目标）。本 UT 只断言 AicpuAllGatherSoleNHR 的 opCustomCheck 真值契约，
// 不触发对其余被桩函数的运行时调用，桩实现仅需满足链接。

#include <string>
#include "auto_selector_base.h"
#include "hccl/hccl_res.h"
#include "op_common.h"

namespace ops_hccl {
// SoleMesh/SoleNHR attrs 的 topoCustomCheck lambda 引用（auto_selector_base.cc 未编入本 UT 目标）
bool AutoSelectorBase::IsLayerAllConnetedWithTopo(
    const TopoInfoWithNetLayerDetails* topoInfo, const u32 netLayer, const CommTopo topoType)
{
    (void)topoInfo;
    (void)netLayer;
    (void)topoType;
    return true;
}

// SoleMesh/SoleNHR attrs 的 topoCustomCheck MESH_1D_CLOS 分支引用；
// 本 UT 用例不断言该分支，桩仅满足链接，出参保持调用方初始化值。
HcclResult AutoSelectorBase::CheckMeshNumEqualToClosNum(const TopoInfoWithNetLayerDetails* topoInfo, bool& isEqual)
{
    (void)topoInfo;
    (void)isEqual;
    return HCCL_SUCCESS;
}

HcclResult
AutoSelectorBase::CheckClosNumMultipleOfMeshNum(const TopoInfoWithNetLayerDetails* topoInfo, bool& isMultiple)
{
    (void)topoInfo;
    (void)isMultiple;
    return HCCL_SUCCESS;
}

// alg_parse.cc 引用（定义在 op_common.cc / alg_env_config.cc，未编入本 UT 目标），
// 返回空串 = "用户未指定算法 / 环境变量未设置"。
HcclResult HcclGetHcclAlgo(HcclComm comm, std::string& hcclAlgo)
{
    (void)comm;
    hcclAlgo = "";
    return HCCL_SUCCESS;
}

std::string GetEnv(std::string idName)
{
    (void)idName;
    return "";
}
} // namespace ops_hccl

// Aiv attrs 的 opCustomCheck lambda 引用（真实定义在 CANN 运行时，声明见 hccl/hccl_res.h）
HcclResult HcclGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size)
{
    (void)comm;
    if (buffer != nullptr) {
        *buffer = nullptr;
    }
    if (size != nullptr) {
        *size = 0;
    }
    return HCCL_SUCCESS;
}
