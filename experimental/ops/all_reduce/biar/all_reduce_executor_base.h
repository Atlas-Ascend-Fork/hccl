/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_EXECUTOR_H
#define ALL_REDUCE_EXECUTOR_H

#include "alg_param.h"
#include "executor_base.h"
#include "config_log.h"
#include "alg_template_base_experimental.h"
#include "coll_executor_base_experimental.h"

namespace ops_hccl_experimental {
using ops_hccl::AlgHierarchyInfo;
using ops_hccl::AlgResourceRequest;
using ops_hccl::AlgType;
using ops_hccl::OpParam;
using ops_hccl::Slice;
using ops_hccl::TopoInfo;

class AllReduceExecutorBase : public CollExecutorBaseExperimental {
public:
    explicit AllReduceExecutorBase();
    ~AllReduceExecutorBase() override = default;

    /* *************** 资源计算 *************** */
    HcclResult CalcResRequest(
        HcclComm comm, const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
        AlgResourceRequest& resourceRequest, AlgType& algType) override;

protected:
    /* *************** 通用工具 *************** */
    virtual HcclResult PrepareDataSlice(u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice>& dataSlice);

    // AllReduce(BIAR) 为 RS+AG 分解: 切片总和为单 rank 数据量, scratch 需求按单 rank 量校验
    u64 CalcScratchDemand(u64 curRecvSize) const override;
};

} // namespace ops_hccl_experimental

#endif
