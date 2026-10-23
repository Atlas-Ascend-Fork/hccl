/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_scatter_sequence_executor_3level_dpu.h"
#include "cost_model.h"
#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "ins_temp_reduce_scatter_mesh_1D_dpu_inter.h"
#include "alg_data_trans_wrapper.h"

#include "alg_attrs_registry.h"
#include "auto_selector_base.h"
namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2ReduceScatterSequenceExecutor3LevelDpu<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InsV2ReduceScatterSequenceExecutor3LevelDpu()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    return RsSeq3LevelCalcRes<InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>(
        comm, param, topoInfo, algHierarchyInfo, resourceRequest, *this);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    CHK_RET(RsSeq3LevelRestoreOrchestrateCtx(param, resCtx, *this));
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2ReduceScatterSequenceExecutor3LevelDpu][Orchestrate] myRank[%u] errNo[0x%016llx] "
            "Reduce scatter executor kernel run failed",
            myRank_, HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

// L2(DPU Mesh)参数按 InsTempReduceScatterMesh1dDpuInter 的 allRank* 变长契约重写(底版按设备 NHR 的
// stride 契约写, 不可照抄: 其 outputSliceStride=0 对 Mesh 非法, 所有源会写到对端同一槽位、规约失效)。
// 契约消费方: DPUKernelRun 发送侧按 allRankSliceSize/allRankDispls.at(rankIdx)、接收与本地规约侧
// (PostLocalReduce)按 .at(myAlgRank) 取源块位移/大小; sliceSize/tailSize/两个 sliceStride 不消费置 0,
// repeatNum/两个 repeatStride 被消费但置惰性值(见函数内注释)
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void InsV2ReduceScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenInterTemplateParams2(
        TemplateDataParams& tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount,
        const u64 loop) const
{
    const u64 dts = dataTypeSize_;
    const u64 C = currDataCount;
    const u64 symMemBaseOff2 = supportSymmetricMemory_ ? processedDataCount * dts : 0;
    tempAlgParamsInter.count = C;
    // 源块基址沿用底版(L1 NHR 输出布局): 本 rank 持 d2 个块, 第 j 块 = 发往 L2 组 j 的超节点归约片,
    // 位于 (r0 + r1*d0)*C + j*d0*d1*C; d1==1(skipLevel1)时 r1=0, 自动退化为 L0 直通布局, 无需特判
    tempAlgParamsInter.buffInfo.inBuffBaseOff
        = symMemBaseOff2 + (rankIdxLevel0_ + rankIdxLevel1_ * rankSizeLevel0_) * C * dts;
    tempAlgParamsInter.buffInfo.outBuffBaseOff = processedDataCount * dts;
    // 对端 j 的源块位移/大小, 均匀填法(L1 输出的 d2 个块恒等长, 与 SplitData 均匀退化情形一致)
    tempAlgParamsInter.allRankDispls.clear();
    tempAlgParamsInter.allRankSliceSize.clear();
    tempAlgParamsInter.allRankProcessedDataCount.clear();
    for (u32 j = 0; j < rankSizeLevel2_; j++) {
        tempAlgParamsInter.allRankDispls.push_back(j * rankSizeLevel0_ * rankSizeLevel1_ * C * dts);
        tempAlgParamsInter.allRankSliceSize.push_back(C * dts);
        tempAlgParamsInter.allRankProcessedDataCount.push_back(C);
    }
    // scratch 与 L1 输出区隔离(联调确认项①): DPU 对端 all-to-all 写入与本卡 PostLocalReduce 规约共需
    // d2*C 的连续 scratch, 须与源块区(L0/L1 原地规约区 [0, R*C))隔离, 否则读写冲突。非对称路径将
    // scratch 排在其后([R*C, (R+d2)*C), 容量由 OrchestrateLoop 的 templateScratchMultiplier 预算保证,
    // 基址各 rank 相同、与 DPUKernelRun 的 remoteCclBuffAddr 相对偏移口径一致); 对称内存路径 L0/L1
    // 直接在 input 对称窗口交换、不占 ccl, ccl 仅为 L2 scratch, 基址取 0(联调确认项②: 对称路径
    // 基址叠加口径)
    tempAlgParamsInter.buffInfo.hcclBuffBaseOff
        = supportSymmetricMemory_ ? 0 : rankSizeLevel0_ * rankSizeLevel1_ * rankSizeLevel2_ * C * dts;
    // sliceSize/tailSize/两个 sliceStride: DpuInter 不消费, 置 0(参照二级 AR StepTwo 的 "没用到" 填法);
    // repeatNum 与两个 repeatStride 会被 repeatIdx 循环消费, 须置惰性值: repeatNum=1 使 repeatIdx*stride
    // 恒为 0; 不可设 repeatNum=0(循环不执行)或配 repeatNum>1(stride=0 时多次叠写同槽位)
    tempAlgParamsInter.sliceSize = 0;
    tempAlgParamsInter.tailSize = 0;
    tempAlgParamsInter.inputSliceStride = 0;
    tempAlgParamsInter.outputSliceStride = 0;
    tempAlgParamsInter.repeatNum = 1;
    tempAlgParamsInter.inputRepeatStride = 0;
    tempAlgParamsInter.outputRepeatStride = 0;

    HCCL_INFO(
        "[InsV2ReduceScatterSequenceExecutor3LevelDpu] myRank[%u] loop[%llu] Inter2 allRankNum[%zu] "
        "inBuffBaseOff[%llu] outBuffBaseOff[%llu] hcclBuffBaseOff[%llu] displs[0][%llu] displs[last][%llu] "
        "sliceSizePerRank[%llu]",
        myRank_, loop, tempAlgParamsInter.allRankSliceSize.size(), tempAlgParamsInter.buffInfo.inBuffBaseOff,
        tempAlgParamsInter.buffInfo.outBuffBaseOff, tempAlgParamsInter.buffInfo.hcclBuffBaseOff,
        tempAlgParamsInter.allRankDispls.empty() ? 0 : tempAlgParamsInter.allRankDispls.front(),
        tempAlgParamsInter.allRankDispls.empty() ? 0 : tempAlgParamsInter.allRankDispls.back(),
        tempAlgParamsInter.allRankSliceSize.empty() ? 0 : tempAlgParamsInter.allRankSliceSize.front());
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
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
    // scratch 分区重算(联调确认项①, 底版为三层系数连乘, 不可照抄): L0/L1 为原地规约、共用
    // [0, m0*m1*m2*C)(即 R*C); L2 DPU scratch(对端写入 + 本地规约, 需 m2*C)须与该区隔离、排在其后,
    // 故总预算 = "L1 输出区 + L2 scratch" = m0*m1*m2 + m2, 与 GenInterTemplateParams2 的
    // hcclBuffBaseOff 取值配套
    u32 templateScratchMultiplier = templateScratchMultiplier0 * templateScratchMultiplier1 * templateScratchMultiplier2
                                    + templateScratchMultiplier2;

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
            "[InsV2ReduceScatterSequenceExecutor3LevelDpu] myRank[%u] maxCountPerLoop is 0, "
            "scratchMultiplier[%u] too large for cclBuffSize[%llu]",
            myRank_, templateScratchMultiplier, tempAlgParamsLevel2.buffInfo.hcclBuff.size);
        return HCCL_E_INTERNAL;
    }
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    if (supportSymmetricMemory_) {
        // 对称路径无法多 loop(L0/L1 在 input 对称窗口原地规约, DPU 对端按全量窗口读写), ccl 仅作
        // L2 scratch(d2*C): 超容不可分块、只能拒绝, 否则 DPU 对端写入/PostLocalReduce 越界且无报错
        u64 symL2ScratchBytes = rankSizeLevel2_ * dataCount_ * dataTypeSize_;
        CHK_PRT_RET(
            symL2ScratchBytes > tempAlgParamsLevel2.buffInfo.hcclBuff.size,
            HCCL_ERROR(
                "[InsV2ReduceScatterSequenceExecutor3LevelDpu][OrchestrateLoop] myRank[%u] symmetric memory "
                "L2 scratch[%llu] exceeds cclBuffSize[%llu] (rankSizeLevel2[%u] * dataCount[%llu] * "
                "dataTypeSize[%llu]), symmetric path cannot multi-loop, reject",
                myRank_, symL2ScratchBytes, tempAlgParamsLevel2.buffInfo.hcclBuff.size, rankSizeLevel2_, dataCount_,
                dataTypeSize_),
            HCCL_E_INTERNAL);
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
            "[InsV2ReduceScatterSequenceExecutor3LevelDpu][OrchestrateLoop] %s: symmetric memory enabled",
            param.algName);
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
InsV2ReduceScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    CalcCostCoeff(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2ReduceScatterSequenceExecutor3LevelDpu][CalcCostCoeff]",
            TopoProbeScene::PROBE_CALC_COST_COEFF)) {
        return {};
    }
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = topoInfo->isPod;
    // infos 外层非空且各已存在层的中层非空方可取 [i][0](MatchTopoForProbe 成功路径结构上恒满足,
    // 防御性收口, 与下方 phyLevel* 守卫同族): 第 0 层无条件访问, 1/2 层仅在外层 size 足够时访问
    if (algHierarchyInfo.infos.empty()) {
        HCCL_WARNING("[InsV2ReduceScatterSequenceExecutor3LevelDpu][CalcCostCoeff] infos empty, skip.");
        return {};
    }
    for (size_t lvl = 0; lvl < algHierarchyInfo.infos.size(); lvl++) {
        if (algHierarchyInfo.infos[lvl].empty()) {
            HCCL_WARNING(
                "[InsV2ReduceScatterSequenceExecutor3LevelDpu][CalcCostCoeff] infos level[%zu] group list empty, "
                "skip.",
                lvl);
            return {};
        }
    }
    u32 rankSizeLevel0 = algHierarchyInfo.infos[0][0].size();
    u32 rankSizeLevel1 = (algHierarchyInfo.infos.size() > 1) ? algHierarchyInfo.infos[1][0].size() : 1;
    u32 rankSizeLevel2 = (algHierarchyInfo.infos.size() > 2) ? algHierarchyInfo.infos[2][0].size() : 1;
    // algo level 对应物理层(MeshConcur 为 2元组)逐层下传, 供跨物理层模板用; 层数不足时高层复用低层
    const auto& physIdx = algHierarchyInfo.physicalIdxForAlgoLevels;
    CHK_PRT_RET(
        physIdx.size() != TOPO_LEVEL_NUM_3,
        HCCL_WARNING(
            "[InsV2ReduceScatterSequenceExecutor3LevelDpu][CalcCostCoeff] physicalIdxForAlgoLevels size[%zu] != 3.",
            physIdx.size()),
        {});
    std::vector<std::vector<CommTopo>> phyLevelNetTypes(physIdx.size());
    std::vector<std::vector<std::vector<u32>>> phyLevelPortNums(physIdx.size());
    AppendPhysicalLevelInfo(topoInfo, physIdx, phyLevelNetTypes, phyLevelPortNums);
    // phyLevel* 外层大小恒等于 physIdx.size()(上方已判 3), 内层按该算法层映射的物理层填充, 拓扑信息
    // 不完整时可能为空: 取 [i][0] 前须先判内层非空, 否则越界读且后续 empty 判断不可达(死判断)
    for (size_t lvl = 0; lvl < phyLevelNetTypes.size(); lvl++) {
        if (phyLevelNetTypes[lvl].empty() || phyLevelPortNums[lvl].empty()) {
            HCCL_WARNING(
                "[InsV2ReduceScatterSequenceExecutor3LevelDpu][CalcCostCoeff] phyLevel info empty at level[%zu] "
                "(netTypes[%zu] portNums[%zu]), skip.",
                lvl, phyLevelNetTypes[lvl].size(), phyLevelPortNums[lvl].size());
            return {};
        }
    }
    CommTopo netTypeLevel0 = phyLevelNetTypes[0][0];
    CommTopo netTypeLevel1 = phyLevelNetTypes[1][0];
    CommTopo netTypeLevel2 = phyLevelNetTypes[2][0];
    const std::vector<u32>& portNumLevel0 = phyLevelPortNums[0][0];
    const std::vector<u32>& portNumLevel1 = phyLevelPortNums[1][0];
    const std::vector<u32>& portNumLevel2 = phyLevelPortNums[2][0];
    if (portNumLevel0.empty() || portNumLevel1.empty() || portNumLevel2.empty()) {
        HCCL_WARNING("[InsV2ReduceScatterSequenceExecutor3LevelDpu][CalcCostCoeff] portNum is empty");
        return {};
    }
    // 准入已收敛至 REGISTER_ALG_ATTRS 的 topoCustomCheck(中间层 CLOS), 统一走 CheckAlgoMatchTopoWithReason
    // 留痕; d2>=2 无需在此重复检查: TopoMatchThreeLevel 对 d2==1 直接 not support, probe 阶段已拦截。
    HCCL_INFO(
        "[InsV2ReduceScatterSequenceExecutor3LevelDpu][CalcCostCoeff] rankSize=%d, rankSizeLevel0=%d, "
        "rankSizeLevel1=%d, rankSizeLevel2=%d, "
        "portNumLevel0=%d, portNumLevel1=%d, portNumLevel2=%d, netTypeLevel0=%d, netTypeLevel1=%d, netTypeLevel2=%d",
        rankSize, rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, portNumLevel0.empty() ? 0 : portNumLevel0[0],
        portNumLevel1.empty() ? 0 : portNumLevel1[0], portNumLevel2.empty() ? 0 : portNumLevel2[0],
        static_cast<int>(netTypeLevel0), static_cast<int>(netTypeLevel1), static_cast<int>(netTypeLevel2));
    std::vector<CostModelParam> params
        = [rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, portNumLevel0, portNumLevel1, portNumLevel2, netTypeLevel0,
           netTypeLevel1, netTypeLevel2, isPod, algName, comm, topoInfo, physIdx, phyLevelNetTypes, phyLevelPortNums] {
              std::vector<CostModelParam> v;
              // 三段 dataRatio 为家族口径 {d1*d2, d2, 1}(底版为 ÷rankSize 口径, 不可照抄): 与二级 DPU RS 的
              // {d1', 1} 同基准; 照抄底版会使同拓扑下三级各段 cost 被低估 R 倍、与二级竞争失真。
              // 尾部三参数 physIdx[lvl]/phyLevelNetTypes[lvl]/phyLevelPortNums[lvl] 不可丢: ZAxisDetour 依赖
              // phyLevel* 做双物理层配对, 漏传则回退 level1PortTotal=8 默认值、L0 段 cost 与运行态口径不一致
              // Step1: 框内 RS（level0 ZAxisDetour 绕路, 全量输入的 rankSizeLevel1*rankSizeLevel2 份）
              auto p0 = InsAlgTemplate0::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel0, 1.0f * rankSizeLevel1 * rankSizeLevel2, netTypeLevel0, BufferType::INPUT,
                  BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, portNumLevel0, isPod, algName, comm, topoInfo,
                  rankSizeLevel1 * rankSizeLevel2, physIdx[0], phyLevelNetTypes[0], phyLevelPortNums[0]});
              // Step2: 框间 RS（level1 NHR, level0 归约后的 1/rankSizeLevel0 份）
              auto p1 = InsAlgTemplate1::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel1, 1.0f * rankSizeLevel2, netTypeLevel1, BufferType::HCCL_BUFFER,
                  BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, portNumLevel1, isPod, algName, comm, topoInfo,
                  rankSizeLevel0 * rankSizeLevel2, physIdx[1], phyLevelNetTypes[1], phyLevelPortNums[1]});
              // Step3: 跨 super node RS（level2 DPU Mesh, level1 归约后的 1/(rankSizeLevel0*rankSizeLevel1) 份）
              auto p2 = InsAlgTemplate2::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel2, 1.0f, netTypeLevel2, BufferType::HCCL_BUFFER, BufferType::OUTPUT,
                  BufferType::HCCL_BUFFER, portNumLevel2, isPod, algName, comm, topoInfo,
                  rankSizeLevel0 * rankSizeLevel1, physIdx[2], phyLevelNetTypes[2], phyLevelPortNums[2]});
              // 任一 template 未实现 CalcCostCoeff（返回空）则整个算法不参与 CostModel
              if (p0.empty() || p1.empty() || p2.empty()) {
                  HCCL_WARNING(
                      "[InsV2ReduceScatterSequenceExecutor3LevelDpu] CalcCostCoeff incomplete, skip "
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
AlgNetMeta
InsV2ReduceScatterSequenceExecutor3LevelDpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)param;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2ReduceScatterSequenceExecutor3LevelDpu][GetAlgNetMeta]",
            TopoProbeScene::PROBE_GET_ALG_NET_META)) {
        return {};
    }
    // 家族口径 {d1*d2, d2, 1}, 与 CalcCostCoeff 的三段 dataRatio 一致: utils 除数取 1
    AlgNetMeta meta = RsSeq3LevelBuildNetMeta(topoInfo, algHierarchyInfo, 1, *this);
    // 显式逐段 algoTypes(名字按层序解析产出 [MeshConcur, NHR, Mesh])
    // CalcAlgCost 现查 meta,本函数正确性直接影响 cost 与 util 查表口径, 故显式设置
    meta.algoTypes = {AlgoType::MESH_CONCUR, AlgoType::NHR, AlgoType::MESH};
    return meta;
}

// DPU 版三级 Sequence: 与设备侧三级同一执行器形态, 仅 L2 模板实参换为 DPU Mesh 模板(编译期实例化,
// 无运行时 engine 分支)。命名硬约束(违反则算法无法被选中): 名字逐段须落在 GetAlgoNameToTypeMap
// 既有键内; L0 必须写 MeshConcur(TopoMatchThreeLevel 依赖解析出的 algoTypes 驱动 AnchorMeshLevels,
// 未匹配到则 not support); L2 的 Mesh 使 AnchorHostDpu 把最高算法层锚到 HOST 物理层(host-DPU 场景)
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, DpuReduceScatterSequenceMeshConcurNHRMesh,
    InsV2ReduceScatterSequenceExecutor3LevelDpu, TopoMatchThreeLevel, InsTempReduceScatterMesh1DZAxisDetour,
    InsTempReduceScatterNHR, InsTempReduceScatterMesh1dDpuInter);
// 与设备侧三级 PipeLine 对齐
REGISTER_ALG_ATTRS(
    DpuReduceScatterSequenceMeshConcurNHRMesh, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
    topo.minTopoLevelNum = TOPO_LEVEL_NUM_3; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; topo.isHostDpuOnly = true;
    topo.isSupportLevel0PcieMix = true; op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_64BIT;
    // 准入(原 CalcCostCoeff 内嵌, 收敛至 attrs 统一走 CheckAlgoMatchTopoWithReason, 过滤原因可查):
    // 中间层网型须为 CLOS(目标场景为超节点内 clos)且为设备侧(host 侧 CLOS 不适配 L1 的设备侧 NHR 模板)。
    // hostdpu 标准三层布局物理层自底向上为 [框内mesh, 框间, HOST DPU], 锚点把 L0/L1 定为首/中/末层
    // (FindAnchors), L1 即 physicalLevels[1]。非三层布局不在目标域, 保守过滤(回退二级 DPU 承接)。
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->physicalLevels.size() == TOPO_LEVEL_NUM_3
               && topo->physicalLevels[1].topoType == CommTopo::COMM_TOPO_CLOS
               && topo->physicalLevels[1].locType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    };);

} // namespace ops_hccl
