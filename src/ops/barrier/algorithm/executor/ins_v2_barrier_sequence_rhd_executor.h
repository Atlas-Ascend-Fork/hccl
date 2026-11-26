/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_V2_BARRIER_SEQUENCE_RHD_EXECUTOR_H
#define INS_V2_BARRIER_SEQUENCE_RHD_EXECUTOR_H

#pragma once

#include <type_traits>
#include "executor_v2_base.h"
#include "alg_v2_template_base.h"
#include "topo_match_base_v2.h"
#include "topo_match_two_level.h"

namespace ops_hccl {

// RHD 专用编排：未指定第三层模板时执行两层，否则按 L2、L1、L0 顺序执行三层。
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2 = void>
class InsV2BarrierSequenceRhdExecutor : public InsCollAlgBase {
public:
    InsV2BarrierSequenceRhdExecutor() {}
    ~InsV2BarrierSequenceRhdExecutor() override {}

    HcclResult CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& algAttrs) override;

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    std::vector<CostModelParam> CalcCostCoeff(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param) override;
    AlgNetMeta GetAlgNetMeta(
        const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const override;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

private:
    static constexpr u32 LEVEL_NUM = std::is_void_v<InsAlgTemplate2> ? 2U : 3U;
};

} // namespace ops_hccl

#endif // INS_V2_BARRIER_SEQUENCE_RHD_EXECUTOR_H
