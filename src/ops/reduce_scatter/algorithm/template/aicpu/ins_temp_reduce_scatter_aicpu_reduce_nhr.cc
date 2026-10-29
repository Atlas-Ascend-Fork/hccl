/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_aicpu_reduce_nhr.h"
#include "cost_model.h"

namespace ops_hccl {

std::vector<CostModelParam> InsTempReduceScatterAicpuReduceNHR::CalcCostCoeff(CalcCostCoeffParam param)
{
    // 口径: cost 乘子 dataSize 为 RS 单 rank 输出份额; dataRatio×rankSize=本段单 rank 输入全量占 dataSize 的份数
    //   (sole 段 dataRatio=1.0, 两级框内段 dataRatio=框间 rank 数, 框间段 dataRatio=1.0, 以调用方 executor 传参为准)
    //   A: rankSize 轮整域 NHR AllGather, 总发送 (rankSize-1)×dataRatio×rankSize 份
    //   B: rankSize 轮入拷贝(输入全量, BufferType 相同跳过) + 本 rank 份额出拷贝
    //   C: 每轮 (nSteps 收发 + 拷贝/规约折算 2 task) + 每轮一次强制同步(折算 SYNC_EQUIV_TASKS 个 2us task)
    //   64bit/PROD 的规约吞吐由框架 per-call 按 count 折算进 B, 此处不建模
    CommTopo netType = param.netType;
    int portNum = 1;
    if (!param.portNum.empty()) {
        portNum = (param.netType == CommTopo::COMM_TOPO_CLOS && param.portNum.size() >= 2) ?
                      (param.portNum[0] + param.portNum[1]) :
                      param.portNum[0];
    }
    if (portNum <= 0) {
        portNum = 1;
    }
    float A = 0.0f;
    // RS 1dpod64 单 server pod 内 CLOS 无跨 pod 端口预留, 不做 isPod 端口减半(减半使 A 斜率高估 ~1.74x),
    // 与 RS 家族 ins_temp_reduce_scatter_nhr.cc / ins_temp_reduce_scatter_hd.cc 同口径
    CostModelManager::Global()->CalcNHRParams(
        param.dataRatio * param.rankSize, netType, portNum, param.rankSize, A, param.isPod, false);

    // B: 入拷贝(输入全量) + 出拷贝(本 rank 份额)
    float B1 = 0.0f;
    float B2 = 0.0f;
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio * param.rankSize, EngineType::AICPU, B1);
    }
    if (param.outputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B2);
    }
    const float B = B1 + B2;

    // AICPU 规约工作量: myIdx 轮串行规约 (rankSize-1) 个 slice(每份 dataRatio×dataSize 字节),
    // 64bit/PROD 时由框架 per-call 按 count×1.2ns 折算进 B
    const float reduceCopies = static_cast<float>(param.rankSize - 1) * param.dataRatio;

    constexpr int SYNC_EQUIV_TASKS = 50;
    int rounds = static_cast<int>(param.rankSize);
    int nSteps = static_cast<int>(GetNHRStepNum(param.rankSize));
    int taskNum = rounds * (nSteps + 2 + SYNC_EQUIV_TASKS);
    float C = 0.0f;
    CostModelManager::Global()->CalcLatencyParams(taskNum, EngineType::AICPU, C);

    // D: 任务下发固定项(不随数据量变化)
    float D = 1e-6f * taskNum;

    HCCL_INFO(
        "[InsTempReduceScatterAicpuReduceNHR][CalcCostCoeff] algName[%s] rankSize[%u] "
        "A=%e B=%e(copy=%e) C=%e D=%e reduceCopies=%f.",
        param.algName != nullptr ? param.algName : "(null)", param.rankSize, A, B, B1 + B2, C, D, reduceCopies);
    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D, reduceCopies});
    return params;
}

InsTempReduceScatterAicpuReduceNHR::InsTempReduceScatterAicpuReduceNHR(
    const OpParam& param, const u32 rankId, // 传通信域的u32，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempReduceScatterAicpuReduceNHR::~InsTempReduceScatterAicpuReduceNHR() {}

HcclResult InsTempReduceScatterAicpuReduceNHR::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // NHR 需要的 que Num 为 1
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, channels));
    resourceRequest.channels.push_back(channels);
    HCCL_INFO(
        "[InsTempReduceScatterAicpuReduceNHR][CalcRes] slaveThreadNum: [%u], notifyNumOnMainThread: [%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceNHR::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterAicpuReduceNHR::GetThreadNum() const { return 1; }

HcclResult InsTempReduceScatterAicpuReduceNHR::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempReduceScatterAicpuReduceNHR] KernelRun start");

    tempAlgParams_ = tempAlgParams;
    channels_ = templateResource.channels;
    dataType_ = param.DataDes.dataType;
    bool isPcieProtocol = IsPcieProtocol(channels_); // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocol;                     // 是否使用Read模式

    for (u32 sliceIdx = 0; sliceIdx < templateRankSize_; ++sliceIdx) {
        // Step 1: 本地拷贝，将本rank的数据拷贝到自己hcclBuffer上相应的位置
        CHK_RET(LocalDataCopy(templateResource.threads, sliceIdx));
        if (sliceIdx == myRank_) {
            CHK_RET(LocalCopyToOutput(templateResource.threads, sliceIdx));
        }
        // Step 2: 数据分发，每个rank通过write方式将对端rank需要的数据写到其hcclBuffer对应的槽位
        CHK_RET(RunAllGather(templateResource.threads));

        // 必须确保所有通信任务完成，因为接下来的 AICPU Reduce 运行在 CPU 上，不感知任务队列同步
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
        for (const auto& thread : templateResource.threads) {
            CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
        }

        // Step 3: 在output上做本地规约
        if (sliceIdx == myRank_) {
            CHK_RET(PostLocalReduce(templateResource.threads));
        }
    }

    HCCL_INFO("[InsTempReduceScatterAicpuReduceNHR] KernelRun end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceNHR::LocalDataCopy(const std::vector<ThreadHandle>& threads, u32 sliceIdx)
{
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterAicpuReduceNHR][LocalDataCopy] empty threads"),
        HcclResult::HCCL_E_INTERNAL);

    ThreadHandle q = threads[0];
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);

    // 获取本rank在算法中的索引
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));

    // 把属于自己的slice从input拷贝到hcclBuff的对应位置
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        // 计算input上的偏移：使用inputSliceStride计算本rank的数据偏移
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 inOff = inBaseOff + tempAlgParams_.inputSliceStride * sliceIdx;

        // 计算hcclBuffer上的偏移
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scOff = scratchBase + tempAlgParams_.sliceSize * myAlgIdx;

        DataSlice src
            = DataSlice(tempAlgParams_.buffInfo.inputPtr, inOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
        DataSlice dst
            = DataSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scOff, tempAlgParams_.sliceSize, tempAlgParams_.count);

        HCCL_INFO(
            "[InsTempReduceScatterAicpuReduceNHR][LocalDataCopy] rpt[%u] inOff[%llu] scOff[%llu] sliceSize[%llu]", rpt,
            inOff, scOff, tempAlgParams_.sliceSize);

        // 如果源地址和目标地址相同，则不需要做拷贝
        if (tempAlgParams_.buffInfo.inBuffType != tempAlgParams_.buffInfo.hcclBuffType || inOff != scOff) {
            CHK_RET(LocalCopy(q, src, dst));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceNHR::LocalCopyToOutput(const std::vector<ThreadHandle>& threads, u32 sliceIdx)
{
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterAicpuReduceNHR][LocalCopyToOutput] empty threads"),
        HcclResult::HCCL_E_INTERNAL);
    ThreadHandle q = threads[0];
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 inOff = inBaseOff + tempAlgParams_.inputSliceStride * sliceIdx;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        DataSlice src
            = DataSlice(tempAlgParams_.buffInfo.inputPtr, inOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
        DataSlice dst
            = DataSlice(tempAlgParams_.buffInfo.outputPtr, outBaseOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
        HCCL_INFO(
            "[InsTempReduceScatterAicpuReduceNHR][LocalCopyToOutput] rpt[%u] inOff[%llu] outBaseOff[%llu] "
            "sliceSize[%llu]",
            rpt, inOff, outBaseOff, tempAlgParams_.sliceSize);
        if (tempAlgParams_.buffInfo.inBuffType != tempAlgParams_.buffInfo.outBuffType || inOff != outBaseOff) {
            CHK_RET(LocalCopy(q, src, dst));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceNHR::PostLocalReduce(const std::vector<ThreadHandle>& threads)
{
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterAicpuReduceNHR][PostLocalReduce] empty threads"),
        HcclResult::HCCL_E_INTERNAL);

    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));
    ThreadHandle q = threads[0];

    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;

        // 然后将其余rank的数据规约到output
        for (u32 rankIdx = 0; rankIdx < templateRankSize_; ++rankIdx) {
            if (rankIdx == myAlgIdx) {
                continue; // 跳过本rank，已经拷贝过了
            }

            const u64 otherScratchOff = scratchBase + tempAlgParams_.sliceSize * rankIdx;
            DataSlice reduceSrcSlice(
                tempAlgParams_.buffInfo.hcclBuff.addr, otherScratchOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
            DataSlice reduceDstSlice(
                tempAlgParams_.buffInfo.outputPtr, outBaseOff, tempAlgParams_.sliceSize, tempAlgParams_.count);

            HCCL_INFO(
                "[InsTempReduceScatterAicpuReduceNHR][PostLocalReduce] reduce from rank[%u] otherScratchOff[%llu]",
                rankIdx, otherScratchOff);

            // 使用AicpuReduce进行规约
            CHK_RET(LocalReduce(q, reduceSrcSlice, reduceDstSlice, dataType_, reduceOp_));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceNHR::RunAllGather(const std::vector<ThreadHandle>& threads)
{
    const u32 nSteps = GetNHRStepNum(templateRankSize_); // NHR 通信步数， ceil(log2(rankSize))

    for (u32 step = 0; step < nSteps; ++step) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo)); // 计算当前step要通信的卡，数据

        const ChannelInfo& channelRecv = channels_.at(GetRankFromMap(stepInfo.fromRank))[0];
        const ChannelInfo& channelSend = channels_.at(GetRankFromMap(stepInfo.toRank))[0];
        // 构造SendRecv， 都是Scratch到Scratch的传输，没有DMA消减
        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        void* sendCclBuffAddr = channelSend.remoteCclMem.addr;
        void* recvCclBuffAddr = channelRecv.remoteCclMem.addr;

        HCCL_DEBUG(
            "[InsTempReduceScatterAicpuReduceNHR] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] "
            "nSlices[%u]",
            myRank_, templateRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

        for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
            const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
            const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

            for (u32 i = 0; i < stepInfo.nSlices; ++i) {
                const u32 txIdx = stepInfo.txSliceIdxs[i];
                const u32 rxIdx = stepInfo.rxSliceIdxs[i];

                const u64 txScratchOff = scratchBase + tempAlgParams_.sliceSize * txIdx;

                const u64 rxScratchOff = scratchBase + tempAlgParams_.sliceSize * rxIdx;

                txSrcSlicesAll.emplace_back(
                    tempAlgParams_.buffInfo.hcclBuff.addr, txScratchOff, tempAlgParams_.sliceSize,
                    tempAlgParams_.count);
                txDstSlicesAll.emplace_back(
                    sendCclBuffAddr, txScratchOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
                rxSrcSlicesAll.emplace_back(
                    recvCclBuffAddr, rxScratchOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
                rxDstSlicesAll.emplace_back(
                    tempAlgParams_.buffInfo.hcclBuff.addr, rxScratchOff, tempAlgParams_.sliceSize,
                    tempAlgParams_.count);
            }
        }
        // write模式使用tx,rx地址不生效，仅使用对端link做Post/Wait
        // read 模式使用rx, tx地址不生效，仅使用对端link做Post/Wait
        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(channelSend, channelRecv);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        if (isDmaRead_) {
            CHK_PRT_RET(
                SendRecvRead(sendRecvInfo, threads[0]),
                HCCL_ERROR("[InsTempReduceScatterAicpuReduceNHR] SendRecv failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                SendRecvWrite(sendRecvInfo, threads[0]),
                HCCL_ERROR("[InsTempReduceScatterAicpuReduceNHR] SendRecv failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterAicpuReduceNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    HCCL_INFO(
        "[InsTempReduceScatterAicpuReduceNHR][CalcScratchMultiple] templateScratchMultiplier[%llu]", templateRankSize_);
    return templateRankSize_;
}

u32 InsTempReduceScatterAicpuReduceNHR::GetRankFromMap(const u32 algRankIdx) { return subCommRanks_[0].at(algRankIdx); }

//  计算每轮收发的对端以及slice编号
HcclResult InsTempReduceScatterAicpuReduceNHR::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = myAlgRank;

    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (myAlgRank + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 sendTo = (myAlgRank + deltaRank) % templateRankSize_;

    // ReduceNHR 数据份数和数据编号增量， NHR是一个传输数据变化的
    u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = myAlgRank;
    u32 rxSliceIdx = (myAlgRank - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.fromRank = recvFrom;
    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG(
            "[InsTempReduceScatterAicpuReduceNHR][ReduceScatterAicpu][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]",
            i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempReduceScatterAicpuReduceNHR::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub)
{
    (void)notifyIdxMianToSub;
}

void InsTempReduceScatterAicpuReduceNHR::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    (void)notifyIdxSubToMain;
}
} // namespace ops_hccl
