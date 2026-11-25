/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_barrier_rhd_aicpu.h"
#include "barrier_rhd_utils.h"
#include "alg_data_trans_wrapper.h"
#include "alg_v2_template_register.h"

namespace ops_hccl {
InsTempBarrierRhdAicpu::InsTempBarrierRhdAicpu(
    const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

HcclResult InsTempBarrierRhdAicpu::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    return CalcBarrierRhdResources(comm, param, topoInfo, subCommRanks_, false, resourceRequest);
}

HcclResult InsTempBarrierRhdAicpu::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& resource)
{
    (void)param;
    (void)tempAlgParams;
    std::vector<BarrierRhdStep> steps;
    CHK_RET(BuildBarrierRhdSteps(myRank_, subCommRanks_, steps));
    if (steps.empty()) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(resource.threads.empty(), HCCL_ERROR("[BarrierRHD] missing main thread."), HCCL_E_INTERNAL);
    CHK_RET(CheckBarrierRhdChannels(steps, resource.channels, ENDPOINT_LOC_TYPE_DEVICE));
    // 所有步骤使用同一通信线程，前一步的 notify wait 约束下一步及下一层的执行顺序。
    // 公共 AICPU kernel 结束批量模式并统一提交，模板只负责入队。
    const std::vector<DataSlice> emptySlices;
    for (const auto& step : steps) {
        const auto& channel = resource.channels.at(step.peerRank)[0];
        TxRxChannels pair(channel, channel);
        TxRxSlicesList slices({emptySlices, emptySlices}, {emptySlices, emptySlices});
        SendRecvInfo info(pair, slices);
        if (step.phase == RhdPhase::MAIN_AG) {
            HCCL_DEBUG(
                "[BarrierRHD] rank[%u] phase[%u] step[%u] peer[%u] virtualPeer[%u] algRank[%u] "
                "userRank[%u] payloadBytes=0.",
                myRank_, static_cast<u32>(step.phase), step.step, step.peerRank, step.virtualPeer, step.algPeer,
                step.peerRank);
        } else {
            HCCL_DEBUG(
                "[BarrierRHD] rank[%u] phase[%u] step[%u] peer[%u] payloadBytes=0.", myRank_,
                static_cast<u32>(step.phase), step.step, step.peerRank);
        }
        CHK_RET(SendRecvWrite(info, resource.threads[0]));
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempBarrierRhdAicpu", InsTempBarrierRhdAicpu);
} // namespace ops_hccl
