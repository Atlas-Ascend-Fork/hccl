/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_barrier_sequence_executor.h"
#include <algorithm>
#include "executor_common_ops.h"
#include "ins_temp_barrier_mesh_1D.h"
#include "ins_temp_barrier_nhr_dpu.h"
#include "coll_alg_v2_exec_registry.h"
#include "alg_attrs_registry.h"

namespace ops_hccl {
namespace {
    constexpr u32 BARRIER_NHR_LEVEL_NUM = 2;

    HcclResult CheckBarrierNhrHierarchy(const AlgHierarchyInfoForAllLevel& hierarchy, u32 myRank)
    {
        CHK_PRT_RET(
            hierarchy.infos.size() != BARRIER_NHR_LEVEL_NUM,
            HCCL_ERROR(
                "[InsV2BarrierSequenceExecutor] expected[%u] levels, got[%zu].", BARRIER_NHR_LEVEL_NUM,
                hierarchy.infos.size()),
            HCCL_E_INTERNAL);
        for (u32 level = 0; level < BARRIER_NHR_LEVEL_NUM; ++level) {
            const auto& subcomms = hierarchy.infos[level];
            CHK_PRT_RET(
                subcomms.size() != 1 || std::find(subcomms[0].begin(), subcomms[0].end(), myRank) == subcomms[0].end(),
                HCCL_ERROR("[InsV2BarrierSequenceExecutor] invalid subcomm for rank[%u] level[%u].", myRank, level),
                HCCL_E_PARA);
        }
        return HCCL_SUCCESS;
    }
} // namespace

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    CHK_PTR_NULL(topoInfo);
    CHK_RET(CheckBarrierNhrHierarchy(algHierarchyInfo, topoInfo->userRank));
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    algHierarchyInfo_ = algHierarchyInfo;

    InsAlgTemplate0 intraTempAlg(param, myRank_, algHierarchyInfo.infos[0]);
    InsAlgTemplate1 interTempAlg(param, myRank_, algHierarchyInfo.infos[1]);

    AlgResourceRequest resReqIntra;
    CHK_RET(intraTempAlg.CalcRes(comm, param, topoInfo, resReqIntra));
    AlgResourceRequest resReqInter;
    CHK_RET(interTempAlg.CalcRes(comm, param, topoInfo, resReqInter));

    resourceRequest.slaveThreadNum = std::max(resReqIntra.slaveThreadNum, resReqInter.slaveThreadNum);
    resourceRequest.notifyNumPerThread = resReqIntra.notifyNumPerThread; // DPU 目前无 notify
    resourceRequest.notifyNumOnMainThread
        = std::max(resReqIntra.notifyNumOnMainThread, resReqInter.notifyNumOnMainThread);

    resourceRequest.channels = {resReqIntra.channels[0], resReqInter.channels[0]};
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2BarrierSequenceExecutor][Orchestrate] Start");
    CHK_RET(CheckBarrierNhrHierarchy(resCtx.algHierarchyInfo, resCtx.topoInfo.userRank));
    CHK_PRT_RET(
        resCtx.channels.size() != BARRIER_NHR_LEVEL_NUM || resCtx.threads.empty(),
        HCCL_ERROR(
            "[InsV2BarrierSequenceExecutor] incomplete resources: channel groups[%zu], threads[%zu].",
            resCtx.channels.size(), resCtx.threads.size()),
        HCCL_E_INTERNAL);
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // Barrier 不依赖 dataCount，直接构造一份空的 TemplateDataParams 给两个模板。
    // 两个模板内部只用 channel/thread 资源，不读 buffInfo / sliceSize / count。
    TemplateDataParams interTempDataParams{};
    interTempDataParams.buffInfo.hcclBuff = resCtx.cclMem;
    interTempDataParams.repeatNum = 1;

    TemplateDataParams intraTempDataParams{};
    intraTempDataParams.buffInfo.hcclBuff = resCtx.cclMem;
    intraTempDataParams.repeatNum = 1;

    InsAlgTemplate1 interTempAlg(param, myRank_, algHierarchyInfo_.infos[1]);
    InsAlgTemplate0 intraTempAlg(param, myRank_, algHierarchyInfo_.infos[0]);

    TemplateResource templateResourceInter;
    SetTemplateCommResource(templateResourceInter, remoteRankToChannelInfo_[1], resCtx);
    templateResourceInter.isDpuThreadEnabled = resCtx.isDpuThreadEnabled;
    templateResourceInter.dpuThread = resCtx.dpuThread;
    templateResourceInter.dpuThreadNotifyIdxOnMain = resCtx.notifyNumOnMainThread + 1;
    templateResourceInter.isHcommChannelDrainSupported = resCtx.isHcommChannelDrainSupported;

    TemplateResource templateResourceIntra;
    SetTemplateCommResource(templateResourceIntra, remoteRankToChannelInfo_[0], resCtx);

    // 先框间（DPU）再框内（AICPU），与 AllGather SequenceExecutor 顺序一致。
    CHK_RET(interTempAlg.KernelRun(param, interTempDataParams, templateResourceInter));
    CHK_RET(intraTempAlg.KernelRun(param, intraTempDataParams, templateResourceIntra));

    HCCL_INFO("[InsV2BarrierSequenceExecutor][Orchestrate] End");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
std::vector<CostModelParam> InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)topoInfo;
    (void)algName;
    (void)param;
    return {{0.0f, 0.0f, 1.0f, 0.0f}};
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
AlgNetMeta InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)algName;
    (void)param;
    return BuildDefaultMeshNetMeta(topoInfo);
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_BARRIER, DpuBarrierSequenceMeshNHR, InsV2BarrierSequenceExecutor, TopoMatchTwoLevel,
    InsTempBarrierMesh1D, InsTempBarrierNHRDPU);
REGISTER_ALG_ATTRS(DpuBarrierSequenceMeshNHR, topo.isSupportLevel0PcieMix = true; topo.isHostDpuOnly = true;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS | LEVEL0_TOPO_CLOS;
                   topo.isSupportLevel1Nhr = true;);
} // namespace ops_hccl
