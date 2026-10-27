/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_REDUCE_HD_H
#define INS_TEMP_ALL_REDUCE_HD_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"
#include <set>

namespace ops_hccl {

// 全域 Halving-Doubling AllReduce：数据划分为 N 个块（块 b 归属 rank b，尾块归 N-1），
// RS 阶段按对齐区段递减距离折半归约（伙伴 = algRank XOR D），AG 阶段递增距离倍增收集；
// 要求 rank 数为 2 的整数次幂（由 TopoMatchOneLevel 硬门控）
class InsTempAllReduceHD : public InsAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::HD};
    InsTempAllReduceHD() = default;
    explicit InsTempAllReduceHD(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempAllReduceHD() override;

    std::string Describe() const override
    {
        std::string info = "Template of all reduce HD with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 GetThreadNum() const override;

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;

private:
    HcclResult ValidateChannels(const std::map<u32, std::vector<ChannelInfo>>& channels) const;
    HcclResult PrepareDataSplitForMultiChannel(const TemplateResource& templateResource);

    HcclResult
    PreCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads, u32 channelIdx) const;
    HcclResult RunReduceScatterHD(
        const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const std::vector<ThreadHandle>& threads, u32 channelIdx);
    HcclResult RunAllGatherHD(
        const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const std::vector<ThreadHandle>& threads, u32 channelIdx);
    HcclResult RunLastStepReadToOutput(
        const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const std::vector<ThreadHandle>& threads, u32 channelIdx);
    HcclResult
    PostCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads, u32 channelIdx) const;

    bool CanReadLastStepToOutput() const;
    bool IsLastStepReadSlice(u32 algRank) const;

    HcclResult GetReduceScatterStepInfoList(std::vector<AicpuNHRStepInfo>& stepInfoList) const;
    HcclResult GetAllGatherStepInfoList(std::vector<AicpuNHRStepInfo>& stepInfoList) const;
    u32 GetHDStepNum() const;

    TemplateDataParams tempAlgParams_;
    std::vector<AicpuNHRStepInfo> reduceScatterSteps_;
    std::vector<AicpuNHRStepInfo> allGatherSteps_;
    u32 dataTypeSize_{0};
    u64 count_{0};
    u64 processSize_{0};
    u64 sliceSize_{0};
    u64 tailSize_{0};

    u32 myRankIdx_{0};
    std::vector<u32> rankList_;

    bool isDmaRead_{false};
    bool readLastStepToOutput_{false};
    std::set<u32> lastStepReadSliceIdxs_;
    bool skipStep0TxPreCopy_{false};
    std::set<u32> step0TxSliceIdxs_;
    std::vector<u64> dataSplit_;
    std::vector<u64> dataOffset_;
    std::vector<u64> dataSplitTail_;
    std::vector<u64> dataOffsetTail_;
    bool supportSymmetricMemAccess_{false};
};

} // namespace ops_hccl

#endif // INS_TEMP_ALL_REDUCE_HD_H
