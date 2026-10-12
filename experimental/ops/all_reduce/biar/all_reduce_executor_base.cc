/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_executor_base.h"

namespace ops_hccl_experimental {

AllReduceExecutorBase::AllReduceExecutorBase() : CollExecutorBaseExperimental() {}

HcclResult AllReduceExecutorBase::CalcResRequest(
    HcclComm comm, const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    AlgResourceRequest& resourceRequest, AlgType& algType)
{
    // 子类必须 override 完成资源申请；默认返回不支持错误，避免静默跳过。
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AllReduceExecutorBase::PrepareDataSlice(u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice>& dataSlice)
{
    CHK_PRT_RET((sliceNum == 0), HCCL_ERROR("[AllReduceExecutorBase][PrepareDataSlice]sliceNum is zero."), HCCL_E_PARA);

    dataSlice.resize(sliceNum);
    // V-style slicing: distribute remainder elements to the first 'remainder' ranks
    u64 baseCount = dataCount / sliceNum;
    u64 remainder = dataCount % sliceNum;
    u64 offsetCount = 0;
    for (u32 i = 0; i < sliceNum; i++) {
        u64 curCount = baseCount + (i < remainder ? 1 : 0);
        dataSlice[i].size = curCount * unitSize;
        dataSlice[i].offset = offsetCount * unitSize;
        offsetCount += curCount;
        HCCL_DEBUG(
            "[AllReduceExecutorBase][PrepareDataSlice] slice[%u]: offset[%llu] size[%llu] count[%llu]", i,
            dataSlice[i].offset, dataSlice[i].size, curCount);
    }
    return HCCL_SUCCESS;
}

u64 AllReduceExecutorBase::CalcScratchDemand(u64 curRecvSize) const { return curRecvSize; }

} // namespace ops_hccl_experimental
