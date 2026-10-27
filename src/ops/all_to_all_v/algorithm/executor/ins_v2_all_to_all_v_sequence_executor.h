/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_V2_ALL_TO_ALL_V_SEQUENCE_EXECUTOR_H
#define INS_V2_ALL_TO_ALL_V_SEQUENCE_EXECUTOR_H

#include "executor_common_ops.h"
#include "topo_match_base.h"
#include "aicpu/hier/ins_temp_all_to_all_hier_stage_base.h"
#include "aicpu/hier/alltoall_stage_template_registry.h"
#include "aicpu/hier/ins_temp_all_to_all_v_meta_stage.h"

namespace ops_hccl {

/*
 * 变长(variadic)分级alltoallv/alltoallvc序列executor(topoLevelNums=2，目标侧中转)。
 *
 * V/VC统一：alltoallvc在host侧已转换为4数组经all2AllVDataDes下发，device侧参数同构。
 *
 * 流程：
 *   Phase 0(meta)：MetaStage两级收集sendCounts全矩阵(N*N*u64，overlay在cclbuff头部)；
 *       KernelRun仅异步入队，executor先batch flush再HcommThreadJoin等待全部矩阵写入
 *       完成后扫描取max条目得maxCount；
 *   数据轮：chunkBytes = cclBuffSize/N/HCCL_MIN_SLICE_ALIGN*HCCL_MIN_SLICE_ALIGN，
 *       roundCount = ceil(maxCount/chunk)；每轮4数组传原始值，
 *       轮偏移(processedDataCount=round*chunk)经inBuffBaseOff/outBuffBaseOff承载。
 */
template <typename AlgTopoMatch>
class InsV2AlltoAllVSequenceExecutor : public InsCollAlgBase {
public:
    explicit InsV2AlltoAllVSequenceExecutor();
    ~InsV2AlltoAllVSequenceExecutor() override = default;
    HcclResult CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& algAttrs) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;
    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;
    std::vector<CostModelParam> CalcCostCoeff(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param) override;

protected:
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);
    // Phase 0：运行MetaStage收集全矩阵并扫描出全局最大单点count
    HcclResult RunMetaPhase(const OpParam& param, const AlgResourceCtxSerializable& resCtx, u64& maxCount);
    HcclResult SetStageParams(
        TemplateDataParams& tempAlgParams, u32 stageIndex, u32 totalStages, u64 loop, u64 currDataCount,
        u64 processedDataCount, u64 maxDataCountPerLoop, u32 totalRankSize, u32 stageRankSize);
    HcclResult FillTemplateResource(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateResource& templateRes, u32 stageIndex);
    // MetaStage资源：线程复用，channels合并intra/inter两组(rank全局唯一)
    HcclResult FillMetaTemplateResource(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateResource& templateRes);
    std::vector<std::string> GetDefaultStageAlgos(u32 levelNum);
    HcclResult BuildStageTemplates(const OpParam& param);
    HcclResult BuildMetaStage(const OpParam& param);

private:
    std::vector<std::shared_ptr<InsTempAlltoAllHierStageBase>> stageTemplates_;
    std::shared_ptr<InsTempAlltoAllVHierMetaStage> metaStage_;
    std::vector<std::string> stageAlgos_;
    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;
    u32 totalStages_{0};
    u32 totalRankSize_{0};
    u64 dataTypeSize_{0};
    HcclDataType dataType_{HCCL_DATA_TYPE_INT8};
};

} // namespace ops_hccl

#endif // !INS_V2_ALL_TO_ALL_V_SEQUENCE_EXECUTOR_H
