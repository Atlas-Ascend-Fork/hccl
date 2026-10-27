/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_TO_ALL_V_HIER_MESH_STAGE_H
#define INS_TEMP_ALL_TO_ALL_V_HIER_MESH_STAGE_H

#include "ins_temp_all_to_all_hier_mesh_stage.h"
#include "ins_temp_all_to_all_hier_stage_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

/*
 * 变长(variadic)分级Mesh数据stage：alltoallv/alltoallvc两级分级的inter/intra数据搬运。
 *
 * 数据流为目标侧中转(inter-first)：
 *   inter stage(first, SendRecvWrite)：本卡input中发往对端server各slot的段 -> 对端cclbuff block[t][s]；
 *   intra stage(last,  SendRecvRead) ：对端cclbuff block[b][i] -> 本卡output直写(rdispls定位)。
 *
 * 关键约定：
 *   1. 变长参数(sendCounts/recvCounts/sdispls/rdispls)经TemplateDataParams传入原始值，
 *      轮内偏移由buffInfo.inBuffBaseOff/outBuffBaseOff(=round*chunk*dt)承载；
 *   2. 本轮slot传输量 = (count > processed) ? min(count - processed, chunk) : 0，防止u64下溢；
 *   3. 恒定wrapper：live侧只push非零slice，死代码侧恒push占位(addr,0,0,0)；
 *      live侧全零时也补占位，保证wrapper被调用(对端可能有数据发往本卡)，避免notify不匹配死锁。
 */
class InsTempAlltoAllVHierMeshStage : public InsTempAlltoAllHierStageBase {
public:
    InsTempAlltoAllVHierMeshStage(
        const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks, u32 stageIndex);

    ~InsTempAlltoAllVHierMeshStage() override = default;

    std::string Describe() const override
    {
        return "Hierarchical AllToAllV Mesh Stage, stageIndex=" + std::to_string(stageIndex_)
               + ", rankSize=" + std::to_string(templateRankSize_);
    }

    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;
    std::string GetAlgoType() const override { return "MeshVStage"; }

private:
    static constexpr u32 INVALID_ALG_RANK = 0xFFFFFFFF;

    u32 FindMyAlgRank() const;
    void InitStageParams(const TemplateDataParams& tempAlgParams, StageParams& params);
    // 本轮slot传输元素数：count > processed ? min(count - processed, chunk) : 0
    u64 CalcSlotCount(u64 count, u64 processedDataCount) const;
    HcclResult LocalCopyForMyGroup(
        const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource, u32 myAlgRank,
        u32 targetGroupSize, u32 subSlotCount);
    HcclResult LocalCopyForMyRank(
        const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource, u32 myAlgRank,
        u32 srcGroupSize, u32 subSlotCount);
    void BuildSendSlices(
        const TemplateDataParams& tempAlgParams, u32 myAlgRank, u32 peerAlgRank, void* remoteCclBuffAddr,
        u32 targetGroupSize, u32 subSlotCount, std::vector<DataSlice>& txSrcSlices,
        std::vector<DataSlice>& txDstSlices);
    void BuildRecvSlices(
        const TemplateDataParams& tempAlgParams, u32 myAlgRank, u32 peerAlgRank, void* remoteCclBuffAddr,
        const StageParams& params, std::vector<DataSlice>& rxSrcSlices, std::vector<DataSlice>& rxDstSlices);
    HcclResult ExecuteSendRecv(
        const ChannelInfo& linkSend, const ChannelInfo& linkRecv, const ThreadHandle& thread,
        const std::vector<DataSlice>& txSrcSlices, const std::vector<DataSlice>& txDstSlices,
        const std::vector<DataSlice>& rxSrcSlices, const std::vector<DataSlice>& rxDstSlices, u32 remoteRank);
    HcclResult RunSendRecv(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource,
        u32 myAlgRank, const StageParams& params);
    HcclResult PreSyncThreads(const TemplateResource& templateResource);
    HcclResult PostSyncThreads(const TemplateResource& templateResource);
    HcclResult ValidateAndInit(const TemplateDataParams& tempAlgParams, u32& myAlgRank, StageParams& params);

    u64 dataTypeSize_{0};
    HcclDataType dataType_{HCCL_DATA_TYPE_INT8};
    u64 cclBufferCountPerRank_{0}; // cclbuff内每slot容量(元素数)，即每轮chunk
};

} // namespace ops_hccl

#endif // !INS_TEMP_ALL_TO_ALL_V_HIER_MESH_STAGE_H
