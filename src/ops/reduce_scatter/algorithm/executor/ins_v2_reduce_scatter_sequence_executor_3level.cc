/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_scatter_sequence_executor_3level.h"
#include "cost_model.h"
#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "alg_data_trans_wrapper.h"

#include "alg_attrs_registry.h"
#include "auto_selector_base.h"
namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2ReduceScatterSequenceExecutor3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InsV2ReduceScatterSequenceExecutor3Level()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    return RsSeq3LevelCalcRes<InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>(
        comm, param, topoInfo, algHierarchyInfo, resourceRequest, *this);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    CHK_RET(RsSeq3LevelRestoreOrchestrateCtx(param, resCtx, *this));
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2ReduceScatterSequenceExecutor3Level][Orchestrate] myRank[%u] errNo[0x%016llx] "
            "Reduce scatter executor kernel run failed",
            myRank_, HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenInterTemplateParams2(
        TemplateDataParams& tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount,
        const u64 loop) const
{
    tempAlgParamsInter.count = currDataCount;
    u64 symMemBaseOff2 = supportSymmetricMemory_ ? processedDataCount * dataTypeSize_ : 0;
    tempAlgParamsInter.buffInfo.inBuffBaseOff
        = symMemBaseOff2 + (rankIdxLevel0_ + (rankIdxLevel1_ * rankSizeLevel0_)) * currDataCount * dataTypeSize_;
    tempAlgParamsInter.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsInter.buffInfo.hcclBuffBaseOff
        = (rankIdxLevel0_ + (rankIdxLevel1_ * rankSizeLevel0_)) * currDataCount * dataTypeSize_;

    tempAlgParamsInter.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsInter.tailSize = tempAlgParamsInter.sliceSize;

    tempAlgParamsInter.inputSliceStride = rankSizeLevel0_ * rankSizeLevel1_ * currDataCount * dataTypeSize_;
    tempAlgParamsInter.outputSliceStride = 0;
    tempAlgParamsInter.repeatNum = 1;
    tempAlgParamsInter.inputRepeatStride = 0;
    tempAlgParamsInter.outputRepeatStride = 0;

    HCCL_INFO(
        "[InsV2ReduceScatterSequenceExecutor3Level] myRank[%u] loop[%llu] Inter2 inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] "
        "repeatNum[%llu] inputRepeatStride[%llu] outputRepeatStride[%llu]",
        myRank_, loop, tempAlgParamsInter.inputSliceStride, tempAlgParamsInter.outputSliceStride,
        tempAlgParamsInter.sliceSize, tempAlgParamsInter.buffInfo.inBuffBaseOff,
        tempAlgParamsInter.buffInfo.outBuffBaseOff, tempAlgParamsInter.repeatNum, tempAlgParamsInter.inputRepeatStride,
        tempAlgParamsInter.outputRepeatStride);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    RsSeq3LevelCtx rsCtx = RsSeq3LevelMakeCtx(*this);

    TemplateDataParams tempAlgParamsLevel0;
    SetTemplateBuffInfo(
        tempAlgParamsLevel0, BufferType::INPUT, BufferType::HCCL_BUFFER, param.inputPtr, resCtx.cclMem.addr,
        resCtx.cclMem);

    std::shared_ptr<InsAlgTemplate0> algTemplateLevel0
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);
    if (rankSizeLevel0_ > 1) {
        CHK_RET(algTemplateLevel0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));
    }

    TemplateDataParams tempAlgParamsLevel1;
    SetTemplateBuffInfo(
        tempAlgParamsLevel1, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, resCtx.cclMem.addr, resCtx.cclMem.addr,
        resCtx.cclMem);

    std::shared_ptr<InsAlgTemplate1> algTemplateLevel1
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);
    if (!skipLevel1_) {
        CHK_RET(algTemplateLevel1->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }

    TemplateDataParams tempAlgParamsLevel2;
    SetTemplateBuffInfo(
        tempAlgParamsLevel2, BufferType::HCCL_BUFFER, BufferType::OUTPUT, resCtx.cclMem.addr, param.outputPtr,
        resCtx.cclMem);

    std::shared_ptr<InsAlgTemplate2> algTemplateLevel2
        = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[2]);
    CHK_RET(algTemplateLevel2->SetchannelsPerRank(remoteRankToChannelInfo_[2]));

    u32 templateScratchMultiplier0 = algTemplateLevel0->CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplier1
        = skipLevel1_ ? 1 : algTemplateLevel1->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplier2
        = algTemplateLevel2->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::OUTPUT);
    u32 templateScratchMultiplier
        = templateScratchMultiplier0 * templateScratchMultiplier1 * templateScratchMultiplier2;

    TemplateResource templateResource0;
    CHK_RET(RsSeq3LevelGenTempResource(resCtx, 0, remoteRankToChannelInfo_, algTemplateLevel0, templateResource0));
    TemplateResource templateResource1;
    if (!skipLevel1_) {
        CHK_RET(RsSeq3LevelGenTempResource(resCtx, 1, remoteRankToChannelInfo_, algTemplateLevel1, templateResource1));
    }
    TemplateResource templateResource2;
    CHK_RET(RsSeq3LevelGenTempResource(resCtx, 2, remoteRankToChannelInfo_, algTemplateLevel2, templateResource2));

    if (templateScratchMultiplier == 0) {
        HCCL_ERROR("[%s] templateScratchMultiplier is 0, division by zero.", __func__);
        return HCCL_E_INTERNAL;
    }
    u64 maxCountPerLoop = tempAlgParamsLevel2.buffInfo.hcclBuff.size / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN
                          * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    if (maxCountPerLoop == 0) {
        HCCL_ERROR(
            "[InsV2ReduceScatterSequenceExecutor3Level] myRank[%u] maxCountPerLoop is 0, "
            "scratchMultiplier[%u] too large for cclBuffSize[%llu]",
            myRank_, templateScratchMultiplier, tempAlgParamsLevel2.buffInfo.hcclBuff.size);
        return HCCL_E_INTERNAL;
    }
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    if (supportSymmetricMemory_) {
        loopTimes = 1;
        tempAlgParamsLevel0.buffInfo.outputPtr = param.inputPtr;
        tempAlgParamsLevel0.buffInfo.outBuffType = BufferType::INPUT;
        tempAlgParamsLevel1.buffInfo.inputPtr = param.inputPtr;
        tempAlgParamsLevel1.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParamsLevel1.buffInfo.outputPtr = param.inputPtr;
        tempAlgParamsLevel1.buffInfo.outBuffType = BufferType::INPUT;
        tempAlgParamsLevel2.buffInfo.inputPtr = param.inputPtr;
        tempAlgParamsLevel2.buffInfo.inBuffType = BufferType::INPUT;
        HCCL_INFO(
            "[InsV2ReduceScatterSequenceExecutor3Level][OrchestrateLoop] %s: symmetric memory enabled", param.algName);
    }
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        RsSeq3LevelGenIntraParams(rsCtx, tempAlgParamsLevel0, processedDataCount, currDataCount, loop);
        CHK_RET(algTemplateLevel0->KernelRun(param, tempAlgParamsLevel0, templateResource0));

        if (!skipLevel1_) {
            RsSeq3LevelGenInter1Params(rsCtx, tempAlgParamsLevel1, processedDataCount, currDataCount, loop);
            CHK_RET(algTemplateLevel1->KernelRun(param, tempAlgParamsLevel1, templateResource1));
        }

        GenInterTemplateParams2(tempAlgParamsLevel2, processedDataCount, currDataCount, loop);
        CHK_RET(algTemplateLevel2->KernelRun(param, tempAlgParamsLevel2, templateResource2));
        processedDataCount += currDataCount;
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
std::vector<CostModelParam>
InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    CalcCostCoeff(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2ReduceScatterSequenceExecutor3Level][CalcCostCoeff]",
            TopoProbeScene::PROBE_CALC_COST_COEFF)) {
        return {};
    }
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = topoInfo->isPod;
    u32 rankSizeLevel0 = algHierarchyInfo.infos[0][0].size();
    u32 rankSizeLevel1 = (algHierarchyInfo.infos.size() > 1) ? algHierarchyInfo.infos[1][0].size() : 1;
    u32 rankSizeLevel2 = (algHierarchyInfo.infos.size() > 2) ? algHierarchyInfo.infos[2][0].size() : 1;
    // algo level 对应物理层(MeshConcur 为 2元组)逐层下传, 供跨物理层模板用; 层数不足时高层复用低层
    const auto& physIdx = algHierarchyInfo.physicalIdxForAlgoLevels;
    CHK_PRT_RET(
        physIdx.size() != TOPO_LEVEL_NUM_3,
        HCCL_WARNING(
            "[InsV2ReduceScatterSequenceExecutor3Level][CalcCostCoeff] physicalIdxForAlgoLevels size[%zu] != 3.",
            physIdx.size()),
        {});
    std::vector<std::vector<CommTopo>> phyLevelNetTypes(physIdx.size());
    std::vector<std::vector<std::vector<u32>>> phyLevelPortNums(physIdx.size());
    AppendPhysicalLevelInfo(topoInfo, physIdx, phyLevelNetTypes, phyLevelPortNums);
    CommTopo netTypeLevel0 = phyLevelNetTypes[0][0];
    CommTopo netTypeLevel1 = phyLevelNetTypes[1][0];
    CommTopo netTypeLevel2 = phyLevelNetTypes[2][0];
    const std::vector<u32>& portNumLevel0 = phyLevelPortNums[0][0];
    const std::vector<u32>& portNumLevel1 = phyLevelPortNums[1][0];
    const std::vector<u32>& portNumLevel2 = phyLevelPortNums[2][0];
    if (portNumLevel0.empty() || portNumLevel1.empty() || portNumLevel2.empty()) {
        HCCL_WARNING("[InsV2ReduceScatterSequenceExecutor3Level][CalcCostCoeff] portNum is empty");
        return {};
    }
    HCCL_INFO(
        "[InsV2ReduceScatterSequenceExecutor3Level][CalcCostCoeff] rankSize=%d, rankSizeLevel0=%d, rankSizeLevel1=%d, "
        "rankSizeLevel2=%d, "
        "portNumLevel0=%d, portNumLevel1=%d, portNumLevel2=%d, netTypeLevel0=%d, netTypeLevel1=%d, netTypeLevel2=%d",
        rankSize, rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, portNumLevel0.empty() ? 0 : portNumLevel0[0],
        portNumLevel1.empty() ? 0 : portNumLevel1[0], portNumLevel2.empty() ? 0 : portNumLevel2[0],
        static_cast<int>(netTypeLevel0), static_cast<int>(netTypeLevel1), static_cast<int>(netTypeLevel2));
    std::vector<CostModelParam> params = [rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, rankSize, portNumLevel0,
                                          portNumLevel1, portNumLevel2, netTypeLevel0, netTypeLevel1, netTypeLevel2,
                                          isPod, algName, comm, topoInfo, physIdx, phyLevelNetTypes, phyLevelPortNums] {
        std::vector<CostModelParam> v;
        // Step1: 框内 RS（level0 mesh，全量输入的 1/rankSizeLevel0）
        auto p0 = InsAlgTemplate0::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel0, 1.0f * rankSizeLevel1 * rankSizeLevel2 / rankSize, netTypeLevel0, BufferType::INPUT,
            BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, portNumLevel0, isPod, algName, comm, topoInfo,
            rankSizeLevel1 * rankSizeLevel2, physIdx[0], phyLevelNetTypes[0], phyLevelPortNums[0]});
        // Step2: 框间 RS（level1 NHR，level0 归约后的 1/level0 份）
        auto p1 = InsAlgTemplate1::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel1, 1.0f * rankSizeLevel2 / rankSize, netTypeLevel1, BufferType::HCCL_BUFFER,
            BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, portNumLevel1, isPod, algName, comm, topoInfo,
            rankSizeLevel0 * rankSizeLevel2, physIdx[1], phyLevelNetTypes[1], phyLevelPortNums[1]});
        // Step3: 跨 super-pod RS（level2 NHR，level1 归约后的 1/(level0*level1) 份）
        auto p2 = InsAlgTemplate2::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel2, 1.0f / rankSize, netTypeLevel2, BufferType::HCCL_BUFFER, BufferType::OUTPUT,
            BufferType::HCCL_BUFFER, portNumLevel2, isPod, algName, comm, topoInfo, rankSizeLevel0 * rankSizeLevel1,
            physIdx[2], phyLevelNetTypes[2], phyLevelPortNums[2]});
        // 任一 template 未实现 CalcCostCoeff（返回空）则整个算法不参与 CostModel
        if (p0.empty() || p1.empty() || p2.empty()) {
            HCCL_WARNING(
                "[InsV2ReduceScatterSequenceExecutor3Level] CalcCostCoeff incomplete, skip "
                "(p0=%zu p1=%zu p2=%zu).",
                p0.size(), p1.size(), p2.size());
            return v;
        }
        v.insert(v.end(), p0.begin(), p0.end());
        v.insert(v.end(), p1.begin(), p1.end());
        v.insert(v.end(), p2.begin(), p2.end());
        return v;
    }();
    return params;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
AlgNetMeta InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)param;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2ReduceScatterSequenceExecutor3Level][GetAlgNetMeta]",
            TopoProbeScene::PROBE_GET_ALG_NET_META)) {
        return {};
    }
    // 三段 dataRatio 为 ÷rankSize 口径 {d1*d2/R, d2/R, 1/R}(与 CalcCostCoeff 一致), utils 除数取 userRankSize
    return RsSeq3LevelBuildNetMeta(topoInfo, algHierarchyInfo, topoInfo->userRankSize, *this);
}

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AicpuReduceScatterSequenceMeshConcurNHRNHR,
    InsV2ReduceScatterSequenceExecutor3Level, TopoMatchThreeLevel, InsTempReduceScatterMesh1DZAxisDetour,
    InsTempReduceScatterNHR, InsTempReduceScatterNHR);
REGISTER_ALG_ATTRS(AicpuReduceScatterSequenceMeshConcurNHRNHR, topo.minTopoLevelNum = TOPO_LEVEL_NUM_3;
                   topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; op.isSupportProd = false;
                   op.unsupportedDataTypes = UNSUPPORTED_64BIT);

} // namespace ops_hccl
