/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include "all_reduce_biar_executor.h"
#include "all_reduce_biar.h"
#include "topo_experimental.h"

namespace ops_hccl_experimental {

AllReduceBIARExecutor::AllReduceBIARExecutor() : AllReduceExecutorBase()
{
    desc_.level1SupportedAlgos
        = {AlgTypeLevel1::ALG_LEVEL1_NHR, AlgTypeLevel1::ALG_LEVEL1_NB, AlgTypeLevel1::ALG_LEVEL1_RING};
    desc_.level2SupportedAlgos
        = {AlgTypeLevel2::ALG_LEVEL2_NHR, AlgTypeLevel2::ALG_LEVEL2_NB, AlgTypeLevel2::ALG_LEVEL2_RING};
}

HcclResult AllReduceBIARExecutor::CalcResRequest(
    HcclComm comm, const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    AlgResourceRequest& resourceRequest, AlgType& algType)
{
    if (topoInfo->serverNum == 1) {
        CHK_RET(CalcGeneralTopoInfoForA3(comm, topoInfo, algHierarchyInfo));
    } else {
        CHK_RET(CalcGeneralTopoInfoInterServer(comm, topoInfo, algHierarchyInfo));
    }
    CHK_RET(RefreshAlgType(algType));
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;

    resourceRequest.slaveThreadNum = BIAR_THREAD_NUM;
    for (u32 index = 0; index < BIAR_THREAD_NUM; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = BIAR_THREAD_NUM;

    // level0 channel
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcLevel0ChannelRequest(param, topoInfo, algHierarchyInfo, algType, level0Channels));
    resourceRequest.channels.push_back(level0Channels);

    HCCL_INFO(
        "[AllReduceBIARExecutor][CalcResRequest]slaveThreadNum[%u] notifyNumPerThread[%zu]"
        " notifyNumOnMainThread[%u] level0Channels[%zu]",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread.size(),
        resourceRequest.notifyNumOnMainThread, level0Channels.size());
    return HCCL_SUCCESS;
}

HcclResult AllReduceBIARExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[AllReduceBIARExecutor][KernelRun] starts.");

    CHK_RET(KernelRunLevel0(param, execMem));

    HCCL_INFO("AllReduce BIAR run success");
    return HCCL_SUCCESS;
}

HcclResult AllReduceBIARExecutor::SelectAndPrepareBirsTemplate(
    u32 localRank, u32 localRankSize, std::unique_ptr<AlgTemplateBase>& templatePtr)
{
    TemplateType RStplType = TEMPLATE_ALL_REDUCE_BIAR;
    templatePtr = AlgTemplateRegistry::Instance().GetAlgTemplate(RStplType);
    CHK_SMART_PTR_NULL(templatePtr);
    CHK_RET(templatePtr->Prepare(localRank, localRankSize));
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_BIAR in COMM_LEVEL0", __func__);

    return HCCL_SUCCESS;
}

HcclResult AllReduceBIARExecutor::KernelRunLevel0(const OpParam& param, ExecMem& execMem)
{
    SubCommInfo level0CommInfo;
    CHK_RET(GetSubCommInfo(COMM_LEVEL0, level0CommInfo));
    u32 level0LocalRank = level0CommInfo.localRank;
    u32 level0LocalRankSize = level0CommInfo.localRankSize;
    u32 sliceNum = level0LocalRankSize;
    std::vector<Slice> dataSegsSlice;
    CHK_RET(PrepareDataSlice(execMem.count, unitSize_, sliceNum, dataSegsSlice));

    std::unique_ptr<AlgTemplateBase> tempAlg = nullptr;
    CHK_RET(SelectAndPrepareBirsTemplate(level0LocalRank, level0LocalRankSize, tempAlg));

    HcclMem UsrInputMem{HCCL_MEM_TYPE_DEVICE, execMem.inputPtr, execMem.count * unitSize_};
    HcclMem UsrOutputMem{HCCL_MEM_TYPE_DEVICE, execMem.outputPtr, execMem.count * unitSize_};

    if (auto Exp = dynamic_cast<AlgTemplateBaseExperimental*>(tempAlg.get())) {
        CHK_RET(Exp->Prepare(
            UsrInputMem, UsrOutputMem, execMem.inputMem, execMem.count, param.DataDes.dataType, thread_, slaveThreads_,
            param.reduceType, 0, dataSegsSlice, 0, false));

        CHK_RET(Exp->RunAsync(level0LocalRank, level0LocalRankSize, channels_[COMM_LEVEL0]));
    } else {
        HCCL_ERROR("[AllReduceBIARExecutor] template is not AlgTemplateBaseExperimental");
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceBIARExecutor", AllReduceBIAR, AllReduceBIARExecutor);
} // namespace ops_hccl_experimental
