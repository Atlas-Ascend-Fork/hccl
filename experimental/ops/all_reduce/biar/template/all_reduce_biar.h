/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_BIAR_H
#define ALL_REDUCE_BIAR_H

#include "alg_template_base.h"
#include "alg_data_trans_wrapper.h"
#include "alg_template_base_experimental.h"

namespace ops_hccl_experimental {
using ops_hccl::AlgTemplateBase;
using ops_hccl::AlgTemplateRegistry;
using ops_hccl::ChannelInfo;
using ops_hccl::CUSTOM_TIMEOUT;
using ops_hccl::DefaultTemplateCreator;
using ops_hccl::HCCL_MIN_SLICE_ALIGN_910B;
using ops_hccl::NOTIFY_IDX_ACK;
using ops_hccl::NOTIFY_IDX_DATA_SIGNAL;
using ops_hccl::PostSyncInterThreads;
using ops_hccl::PreSyncInterThreads;
using ops_hccl::RoundUpWithDivisor;
using ops_hccl::Slice;
using ops_hccl::TemplateType;

// BIAR 从线程数单一事实源: executor 侧据此申请 slaveThreadNum/notify 资源, 模板侧据此生成 notifyIdx
// 并索引 subThreads_, 两侧必须一致
constexpr u32 BIAR_THREAD_NUM = 2;

class AllReduceBIAR : public AlgTemplateBaseExperimental {
public:
    explicit AllReduceBIAR();

    ~AllReduceBIAR() override;

    // should be called soon after template AllReduceBIAR instance created
    HcclResult Prepare(u32 interRank, u32 interRankSize) override;

    HcclResult Prepare(
        HcclMem& inputMem, HcclMem& outputMem, HcclMem& scratchMem, const u64 count, const HcclDataType dataType,
        ThreadHandle thread, const std::vector<ThreadHandle>& slaveThreads, const HcclReduceOp reductionOp,
        const u32 root, const std::vector<Slice>& slices, const u64 baseOffset, const bool disableDMAReduce) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels) override;

protected:
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub);
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain);

    virtual HcclResult Preprocess(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels);
    HcclResult CheckScratchCapacity(u32 rankSize, u64 localStrideSize);
    HcclResult
    RunReduceScatter(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels, u64 localStrideSize);
    HcclResult
    RunAllGather(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels, u64 localStrideSize);

    HcclResult HCCSProcessRsLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize);
    HcclResult SIOProcessRsLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize);
    HcclResult LocalCopyRsLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize);
    HcclResult TreeReduceRs(const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize);
    HcclResult LocalReduceCCLToCCL(u64 srcOffset, u64 dstOffset, u64 size, ThreadHandle thread);

    HcclResult HCCSProcessAgLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize);
    HcclResult SIOProcessAgLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize);
    HcclResult LocalCopyAgLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 localStrideSize);

    u32 interRank_;
    u32 interRankSize_;
    u32 sio_rank_;
    std::vector<u32> hccs_ranks;
    std::vector<u32> hccs_ranks_reversed;
    std::vector<u32> hccs_neighbour_rank;

    std::vector<ChannelInfo> hccs_links;
    std::vector<ChannelInfo> hccs_links_reversed;
    ChannelInfo sio_link;

    u32 unitSize_;
    ThreadHandle mainThread_;
    std::vector<ThreadHandle> subThreads_;
    std::vector<u32> notifyIdxMainToSub_;
    std::vector<u32> notifyIdxSubToMain_;
    u32 rankSizeX_;
    u32 rankSizeY_;

private:
};
} // namespace ops_hccl_experimental

#endif /* ALL_REDUCE_BIAR_H */
