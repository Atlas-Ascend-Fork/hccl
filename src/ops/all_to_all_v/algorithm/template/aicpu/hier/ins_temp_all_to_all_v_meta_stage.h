/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_TO_ALL_V_META_STAGE_H
#define INS_TEMP_ALL_TO_ALL_V_META_STAGE_H

#include "ins_temp_all_to_all_hier_stage_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

/*
 * 变长(variadic)分级Meta stage：alltoallv/alltoallvc两级分级的Phase 0元数据收集。
 *
 * 目标：把sendCounts全矩阵(N*N个u64)收集到每卡cclbuffer头部(overlay复用)，
 * 供executor扫描取max条目计算roundCount。
 *
 * 矩阵布局(slot-major行号)：第(b*serverNum+s)行 = rank(s,b)的sendCounts行(N个u64)，
 * 行大小rowSize = N*sizeof(u64)。rank = s*serverSize + b(s=server序号，b=server内slot)。
 *
 * 交换流程(与数据stage一致的恒定wrapper约定，占位slice保证wrapper每peer必被调用一次)：
 *   本地：本卡sendCounts行(AICPU直读param)直写本卡cclbuff第(b*serverNum+s)行；
 *   0a inter交换(SendRecvWrite)：与同位组(同b跨server)各rank整行交换，同偏移写对端；
 *   0b intra收集(SendRecvRead)：从server内各peer(同server跨b')读其0a结果的连续serverNum行，
 *       0b的ACK-wait天然保证对端已完成0a(进入0b前必先完成0a)。
 *
 * executor直接构造(不经registry)，subCommRanks = {intraRanks, interRanks}双组传入，
 * 基类自动推导templateRankSize_ = serverSize * serverNum = N。
 *
 * KernelRun仅向设备流异步追加SQE后即返回，矩阵完成性由executor侧batch flush +
 * HcommThreadJoin保证后才扫描(见RunMetaPhase)，本stage无需自带完成标志。
 */

class InsTempAlltoAllVHierMetaStage : public InsTempAlltoAllHierStageBase {
public:
    InsTempAlltoAllVHierMetaStage(
        const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks, u32 stageIndex);

    ~InsTempAlltoAllVHierMetaStage() override = default;

    std::string Describe() const override
    {
        return "Hierarchical AllToAllV Meta Stage, rankSize=" + std::to_string(templateRankSize_);
    }

    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;
    std::string GetAlgoType() const override { return "MetaStage"; }

private:
    static constexpr u32 INVALID_ALG_RANK = 0xFFFFFFFF;

    static u32 FindIdxInGroup(const std::vector<u32>& group, u32 rankId);
    HcclResult ValidateAndInit();
    HcclResult LocalCopyMatrixRow(
        const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource);
    HcclResult RunInterExchange(const TemplateDataParams& tempAlgParams, TemplateResource& templateResource);
    HcclResult RunIntraGather(const TemplateDataParams& tempAlgParams, TemplateResource& templateResource);
    HcclResult PreSyncThreads(const TemplateResource& templateResource);
    HcclResult PostSyncThreads(const TemplateResource& templateResource);

    u32 serverSize_{0};                 // server内组大小(intra组, subCommRanks_[0])
    u32 serverNum_{0};                  // server数(inter组, subCommRanks_[1])
    u32 myServerIdx_{INVALID_ALG_RANK}; // s：本rank所在server序号(inter组内下标)
    u32 mySlotIdx_{INVALID_ALG_RANK};   // b：本rank在server内slot下标(intra组内下标)
};

} // namespace ops_hccl

#endif // !INS_TEMP_ALL_TO_ALL_V_META_STAGE_H
