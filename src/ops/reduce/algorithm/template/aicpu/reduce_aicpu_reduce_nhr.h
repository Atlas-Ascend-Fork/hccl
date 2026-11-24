/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_REDUCE_AICPU_REDUCE_NHR_H
#define AICPU_REDUCE_AICPU_REDUCE_NHR_H

#include <cstring>
#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class ReduceAicpuReduceNHR : public InsAlgTemplateBase {
public:
    ReduceAicpuReduceNHR() = default;
    explicit ReduceAicpuReduceNHR(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);

    ~ReduceAicpuReduceNHR() override;

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter Mesh with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    // 现在的Kernel就是之前的GenExtIns
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    void SetRoot(u32 root);
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

    u64 GetThreadNum() const override;

private:
    HcclResult CalcSlice(const u64 chunkSize);
    HcclResult
    PreCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads, u32 channelIdx);
    HcclResult RunGather(
        const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels,
        u32 channelIdx);
    HcclResult RunReduceInitCopy(
        const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads, u32 channelIdx);
    HcclResult
    RunReduceLocal(const std::vector<ThreadHandle>& threads, const TemplateDataParams& tempAlgParams, u32 channelIdx);

    HcclResult GetStepInfoList(std::vector<AicpuNHRStepInfo>& stepInfoList);
    HcclResult GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo);
    HcclResult GetAlgRank(u32 rank, u32& algRank) const;

    u64 processSize_{0};
    u64 count_{0};
    u32 myIdx_ = UINT32_MAX; // 本rank在通信域内的索引

    RankSliceInfo sliceInfoVec_;
    bool isDmaRead_{false};
    std::vector<u64> sizeOut_;    // 每通道子区大小(字节), 多通道数据切分
    std::vector<u64> elemOffset_; // 每通道子区起始偏移(字节)
    std::vector<u32> notifyIdxMainToSub_;
    std::vector<u32> notifyIdxSubToMain_;
};

} // namespace ops_hccl

#endif // AICPU_REDUCE_AICPU_REDUCE_NHR_H
