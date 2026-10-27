/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_HD_H
#define INS_TEMP_REDUCE_SCATTER_HD_H

#include <set>
#include "alg_v2_template_base.h"
#include "executor_v2_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

// 框间 HD ReduceScatter：框间维度按对齐区段递减距离折半归约，每 rank 最终持有自己块的完整归约结果；
// 伙伴 = algRank XOR 2^k，框间 rank 数必须为 2 的整数次幂（由 TopoMatchTwoLevel 硬门控）
class InsTempReduceScatterHD : public InsAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::HD};
    InsTempReduceScatterHD() = default;
    explicit InsTempReduceScatterHD(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempReduceScatterHD() override;

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter HD with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

private:
    HcclResult GetHDStepInfoList(std::vector<AicpuNHRStepInfo>& stepInfoList);
    HcclResult CollectStep0TxSliceIdxs();
    HcclResult LocalDataCopy(const std::vector<ThreadHandle>& threads, u32 channelIdx);
    HcclResult RunHD(const std::vector<ThreadHandle>& threads, u32 channelIdx);
    HcclResult PostLocalCopy(const std::vector<ThreadHandle>& threads, u32 channelIdx);
    TemplateDataParams tempAlgParams_;
    std::map<u32, std::vector<ChannelInfo>> channels_;
    u64 dataTypeSize_{0};
    std::vector<u64> sizeOut_;
    std::vector<u64> elemOffset_;
    std::vector<u64> sizeOutTail_;
    std::vector<u64> elemOffsetTail_;
    bool isDmaRead_{false};
    bool doPreCopy_{false};
    bool skipStep0TxPreCopy_{false};
    std::set<u32> step0TxSliceIdxs_;
    bool supportSymmetricMemAccess_{false};
};

} // namespace ops_hccl

#endif // !INS_TEMP_REDUCE_SCATTER_HD_H
