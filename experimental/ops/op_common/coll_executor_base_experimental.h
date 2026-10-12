/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_EXECUTOR_BASE_EXPERIMENTAL_H
#define COLL_EXECUTOR_BASE_EXPERIMENTAL_H

#include "alg_param.h"
#include "executor_base.h"
#include "config_log.h"

namespace ops_hccl_experimental {
using ops_hccl::AlgResourceCtx;
using ops_hccl::ChannelInfo;
using ops_hccl::ExecutorBase;
using ops_hccl::OpParam;

// 集合通信 executor 公共 orchestration 基类: 统一 Orchestrate/RunLoop/IsHugeData 与线程/通道成员,
// 消除各算子 executor 基类的逐行重复; RunLoop 失败路径保证 HcommBatchModeEnd/HcommReleaseComm 释放。
// 数据切片语义(均匀/V式)由派生类自行实现 PrepareDataSlice。
class CollExecutorBaseExperimental : public ExecutorBase {
public:
    explicit CollExecutorBaseExperimental();
    ~CollExecutorBaseExperimental() override = default;

    HcclResult Orchestrate(const OpParam& param, AlgResourceCtx* resCtx) override;

protected:
    // 用于需要Loop的Executor
    virtual HcclResult RunLoop(const OpParam& param);

    bool IsHugeData(u64 curSize) const;

    // 单位数据大小: 默认取 DataDes.dataType; 变长算子(如 ReduceScatterV) override 取 vDataDes.dataType
    virtual u32 CalcUnitSize(const OpParam& param) const;

    // RunLoop 的 scratch 容量校验值: 默认按 ReduceScatter 语义取全量(userRankSize*单rank接收量,
    // 全量数据驻留 CCL buffer 前半区); AllReduce(RS+AG 分解, 切片总和为单 rank 数据量)等
    // 单 rank 量需求的算子由派生类 override
    virtual u64 CalcScratchDemand(u64 curRecvSize) const;

    ThreadHandle thread_ = 0;
    std::vector<ThreadHandle> slaveThreads_;
    std::vector<std::vector<ChannelInfo>> channels_;
    u32 unitSize_;
};

} // namespace ops_hccl_experimental

#endif
