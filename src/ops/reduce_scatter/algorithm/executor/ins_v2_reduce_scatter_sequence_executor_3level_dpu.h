/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_REDUCE_SCATTER_SEQUENCE_EXECUTOR_3LEVEL_DPU_H
#define HCCLV2_INS_V2_REDUCE_SCATTER_SEQUENCE_EXECUTOR_3LEVEL_DPU_H

#include "executor_common_ops.h"
#include "ins_v2_reduce_scatter_sequence_3level_utils.h"
#include "topo_match_base.h"
#include "topo_match_multilevel.h"
#include "topo_match_base_v2.h"
#include "topo_match_two_level.h"
#include "topo_match_three_level.h"

namespace ops_hccl {
// DPU 版三级 Sequence 执行器: 复制设备侧三级骨架(InsV2ReduceScatterSequenceExecutor3Level),
// L2 模板实参换为 DPU Mesh 模板(InsTempReduceScatterMesh1dDpuInter, 编译期实例化, 无运行时 engine 分支)。
// 与底版的差异: L2 参数按 DpuInter 变长契约(allRank*)重写、GenTempResource 补 shmem 透传、
// cost 接口按家族口径配置(见 .cc 内注释)。
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
class InsV2ReduceScatterSequenceExecutor3LevelDpu : public InsV2ReduceScatterSequence3LevelBase {
public:
    explicit InsV2ReduceScatterSequenceExecutor3LevelDpu();
    ~InsV2ReduceScatterSequenceExecutor3LevelDpu() override = default;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& algAttrs) override;

    std::vector<CostModelParam> CalcCostCoeff(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param) override;
    AlgNetMeta GetAlgNetMeta(
        const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const override;

protected:
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);
    void GenInterTemplateParams2(
        TemplateDataParams& tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount,
        const u64 loop) const;
};
} // namespace ops_hccl

#endif
