/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_executor_base_experimental.h"

namespace ops_hccl_experimental {
using ops_hccl::AlgHierarchyInfo;
using ops_hccl::ExecMem;
using ops_hccl::HCCL_INTERNODE_MAX_DATA_RATE;
using ops_hccl::RDMA_SEND_MAX_SIZE;
using ops_hccl::SDMA_SEND_MAX_SIZE;

CollExecutorBaseExperimental::CollExecutorBaseExperimental() : ExecutorBase() {}

u32 CollExecutorBaseExperimental::CalcUnitSize(const OpParam& param) const
{
    return HCCL_SIZE_TABLE[param.DataDes.dataType];
}

u64 CollExecutorBaseExperimental::CalcScratchDemand(u64 curRecvSize) const
{
    return topoInfo_->userRankSize * curRecvSize;
}

// 执行入口
HcclResult CollExecutorBaseExperimental::Orchestrate(const OpParam& param, AlgResourceCtx* resCtx)
{
    HcclUs startut = TIME_NOW();
    topoInfo_ = &(resCtx->topoInfo);
    algResource_ = resCtx;
    tag_ = std::string(param.tag);
    algType_ = resCtx->algType;
    unitSize_ = CalcUnitSize(param);

    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);

    // 做参数的还原
    ThreadHandle* threadHandlePtr
        = reinterpret_cast<ThreadHandle*>(reinterpret_cast<char*>(algResource_) + sizeof(AlgResourceCtx));
    ChannelInfo* channelInfoPtr = reinterpret_cast<ChannelInfo*>(
        reinterpret_cast<char*>(threadHandlePtr) + sizeof(ThreadHandle) * (algResource_->slaveThreadNum + 1));

    HCCL_DEBUG("[CollExecutorBaseExperimental][Orchestrate] slaveThreadNum[%u]", algResource_->slaveThreadNum);
    for (u32 i = 0; i < algResource_->slaveThreadNum + 1; i++) {
        HCCL_DEBUG("[CollExecutorBaseExperimental][Orchestrate] threadHandle[%u]=[%llu]", i, threadHandlePtr[i]);
        if (i == 0) {
            thread_ = threadHandlePtr[i];
        } else {
            slaveThreads_.push_back(threadHandlePtr[i]);
        }
    }
    AlgHierarchyInfo& algHierarchyInfo = resCtx->algHierarchyInfo;
    channels_.resize(algHierarchyInfo.levels);
    for (u32 level = 0; level < algHierarchyInfo.levels; level++) {
        u32 curLevelRankSize = algHierarchyInfo.infos[level].localRankSize;
        channels_[level].resize(curLevelRankSize);
        for (u32 rank = 0; rank < curLevelRankSize; rank++) {
            channels_[level][rank] = channelInfoPtr[rank];
        }
        channelInfoPtr += curLevelRankSize;
    }

    HcclResult ret = RunLoop(param);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollExecutorBaseExperimental][Orchestrate]errNo[0x%016llx]collective executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    HCCL_INFO(
        "[CollExecutorBaseExperimental][Orchestrate]tag[%s] collective executor orchestrate success, take time "
        "[%lld]us.",
        param.tag, DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

bool CollExecutorBaseExperimental::IsHugeData(u64 curSize) const
{
    bool hugeData = curSize * topoInfo_->userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE
                    || curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult CollExecutorBaseExperimental::RunLoop(const OpParam& param)
{
    u64 totalRecvCount = param.DataDes.count;

    u8* curUserInputPtr = static_cast<u8*>(param.inputPtr);
    u8* curUserOutputPtr = static_cast<u8*>(param.outputPtr);
    auto cclInputMem = algResource_->cclInputMem;
    auto cclOutputMem = algResource_->cclOutputMem;
    CHK_PRT_RET(
        (cclInputMem.size == 0), HCCL_ERROR("[CollExecutorBaseExperimental][RunLoop]cclBuffer size is zero"),
        HCCL_E_PARA);

    u64 curRecvCount = totalRecvCount;
    u64 curRecvSize = curRecvCount * unitSize_;
    u64 curSendSize = topoInfo_->userRankSize * curRecvSize;

    // 资源申请(HcommAcquireComm/HcommBatchModeStart)前先做容量校验, 校验失败直接返回不持有任何资源。
    // scratch 需求随算子语义不同: ReduceScatter 为全量(userRankSize*单rank量), AllReduce(RS+AG 分解,
    // 切片总和为单 rank 数据量)为单 rank 量, 由 CalcScratchDemand 统一提供
    u64 scratchDemand = CalcScratchDemand(curRecvSize);
    CHK_PRT_RET(
        scratchDemand > cclInputMem.size,
        HCCL_ERROR(
            "[CollExecutorBaseExperimental][RunLoop]scratch demand[%llu] exceeds cclInputMem size[%llu]", scratchDemand,
            cclInputMem.size),
        HCCL_E_PARA);

    if (param.engine == CommEngine::COMM_ENGINE_CPU_TS || param.engine == CommEngine::COMM_ENGINE_CPU) {
        int32_t ret = HcommAcquireComm(param.commName);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] [%s] HcommAcquireComm failed ", __func__, param.commName),
            static_cast<HcclResult>(ret));
    }

#ifndef AICPU_COMPILE
    if (!IsHugeData(curRecvSize)) {
        HcclResult startRet = static_cast<HcclResult>(HcommBatchModeStart(param.algTag));
        if (startRet != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] [%s] HcommBatchModeStart failed", __func__, param.algTag);
            if (param.engine == CommEngine::COMM_ENGINE_CPU_TS || param.engine == CommEngine::COMM_ENGINE_CPU) {
                int32_t relRet = HcommReleaseComm(param.commName);
                if (relRet != HCCL_SUCCESS) {
                    HCCL_ERROR("[%s] [%s] HcommReleaseComm failed ", __func__, param.commName);
                }
            }
            return startRet;
        }
    }
#endif
    // scratch 实际可用区域为整个 cclInputMem(CCL buffer 前半区), 声明大小必须传真实容量而非 curSendSize:
    // 模板容量校验(如 CheckScratchCapacity)以 scratchMem_.size 判断, 声明失真会误报或漏判
    HcclMem curInputMem{cclInputMem.type, cclInputMem.addr, cclInputMem.size};
    HcclMem curOutputMem{cclOutputMem.type, cclOutputMem.addr, curRecvSize};

    ExecMem execMem;
    execMem.count = curRecvCount;
    execMem.inputMem = curInputMem;
    execMem.outputMem = curOutputMem;
    execMem.inputPtr = curUserInputPtr;
    execMem.outputPtr = curUserOutputPtr;

    HCCL_DEBUG(
        "[CollExecutorBaseExperimental][RunLoop] curUserInputPtr[%p], curUserOutputPtr[%p], "
        "curRecvCount[%llu], curRecvSize[%llu], curSendSize[%llu], inputPtr[%p], outputPtr[%p]",
        curUserInputPtr, curUserOutputPtr, curRecvCount, curRecvSize, curSendSize, curInputMem.addr, curOutputMem.addr);

    HcclResult runRet = KernelRun(param, execMem);

    HcclResult releaseRet = HCCL_SUCCESS;
#ifndef AICPU_COMPILE
    if (!IsHugeData(curRecvSize)) {
        releaseRet = static_cast<HcclResult>(HcommBatchModeEnd(param.algTag));
        if (releaseRet != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] [%s] HcommBatchModeEnd failed", __func__, param.algTag);
        }
    }
#endif
    if (param.engine == CommEngine::COMM_ENGINE_CPU_TS || param.engine == CommEngine::COMM_ENGINE_CPU) {
        int32_t ret = HcommReleaseComm(param.commName);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] [%s] HcommReleaseComm failed ", __func__, param.commName);
            if (releaseRet == HCCL_SUCCESS) {
                releaseRet = static_cast<HcclResult>(ret);
            }
        }
    }
    if (runRet != HCCL_SUCCESS) {
        return runRet;
    }
    return releaseRet;
}

} // namespace ops_hccl_experimental
