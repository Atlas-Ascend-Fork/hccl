/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_executor_base.h"

namespace ops_hccl_experimental {

ReduceScatterExecutorBase::ReduceScatterExecutorBase() : CollExecutorBaseExperimental() {}

HcclResult ReduceScatterExecutorBase::CalcResRequest(
    HcclComm comm, const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    AlgResourceRequest& resourceRequest, AlgType& algType)
{
    return HCCL_SUCCESS;
}

HcclResult
ReduceScatterExecutorBase::PrepareDataSlice(u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice>& dataSlice)
{
    CHK_PRT_RET(
        (sliceNum == 0), HCCL_ERROR("[ReduceScatterExecutorBase][PrepareDataSlice]sliceNum is zero."), HCCL_E_PARA);

    dataSlice.resize(sliceNum);
    u64 sliceSize = dataCount * unitSize;
    for (u32 i = 0; i < sliceNum; i++) {
        dataSlice[i].size = sliceSize;
        dataSlice[i].offset = (i * sliceSize);
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl_experimental
