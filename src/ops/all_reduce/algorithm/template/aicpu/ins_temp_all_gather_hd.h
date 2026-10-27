/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_HD_H
#define INS_TEMP_ALL_GATHER_HD_H

#include "alg_v2_template_base.h"
#include "executor_base.h"

namespace ops_hccl {

// 框间 HD AllGather：框间维度按对齐区段递增距离倍增交换，最终每 rank 持有全部块；
// 伙伴 = algRank XOR 2^k，框间 rank 数必须为 2 的整数次幂（由 TopoMatchTwoLevel 硬门控）
class InsTempAllGatherHD : public InsAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::HD};
    InsTempAllGatherHD() = default;
    explicit InsTempAllGatherHD(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempAllGatherHD() override;

    std::string Describe() const override
    {
        std::string info = "Template of all gather HD with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;

    HcclResult SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

private:
    HcclResult GetHDStepInfo(u32 step, AicpuNHRStepInfo& stepInfo);
    u32 GetRankFromMap(const u32 algRankIdx) const;
    HcclResult PrepareDataSplitForMultiChannel(const TemplateResource& templateResource);
    HcclResult LocalDataCopy(const std::vector<ThreadHandle>& threads, const u32& channelIdx);
    HcclResult PostLocalCopy(const ThreadHandle& thread, const u32& channelIdx);
    HcclResult RunAllGatherHD(
        const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const u32& channelIdx);

    TemplateDataParams tempAlgParams_;
    bool isDmaRead_{false};
    bool isOutputCclBuff_{false};
    u64 dataTypeSize_{0};
    std::vector<u64> dataSplit_;
    std::vector<u64> dataOffset_;
    std::vector<u64> dataSplitTail_;
    std::vector<u64> dataOffsetTail_;
};

} // namespace ops_hccl

#endif // INS_TEMP_ALL_GATHER_HD_H
