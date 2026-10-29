/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_mesh1d_mem2mem.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0 = 0;

static CcuResult ParseKernelArg(ReduceMesh1DMem2MemContext& ctx, CcuKernelArgReduceMesh1DMem2Mem* kernelArg)

{
    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;

        HCCL_DEBUG(
            "[CcuKernelReduceMesh1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]", ctx.dataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceMesh1DMem2MemContext& ctx)
{
    const auto* arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceMesh1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] channels.size: [%u]", arg->channelCount);

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求算法返回的Link同样是按顺序排列的
    ctx.input.resize(arg->rankSize);
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId != arg->rankId) {
            ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    ctx.reduceSrc.resize(arg->rankSize);

    ctx.moConfig.loopCount = CCU_M2M_LOCAL_COPY_LOOP_COUNT;
    ctx.moConfig.msInterleave = CCU_MS_INTERLEAVE;
    ctx.moConfig.memSlice = CCU_MS_SIZE;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceMesh1DMem2MemContext& ctx)
{
    const auto* arg = ctx.arg;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId], TASKARG_INPUT_ADDR));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], TASKARG_OUTPUT_ADDR));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, TASKARG_SCRATCH_ADDR));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], TASKARG_TOKEN));
    uint32_t cnt = TASKARG_HEAD_SIZE;
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumVar, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.reduceGoSize.addrOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.reduceGoSize.loopParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.reduceGoSize.parallelParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.reduceGoSize.residual, cnt++));
    return CCU_SUCCESS;
}

static void PreSync(ReduceMesh1DMem2MemContext& ctx)
{
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMeshMem2Mem1D PreSync begin");
    const auto* arg = ctx.arg;
    // 远端 output 无消费者，仅交换 input/token。
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.input[arg->rankId], INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.token[arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMeshMem2Mem1D PreSync end");
}

static void PostSync(ReduceMesh1DMem2MemContext& ctx)
{
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMeshMem2Mem1D post sync start");
    const auto* arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMeshMem2Mem1D post sync end");
}

static CcuResult
DoRepeatReduce(ReduceMesh1DMem2MemContext& ctx, const std::vector<ccu::Variable>& srcAddr, const ccu::Variable& dstAddr)
{
    const auto* arg = ctx.arg;

    ctx.dstAddr.addr = dstAddr;
    ctx.dstAddr.token = ctx.token[arg->rankId];

    ctx.myInputAddr.addr = srcAddr[arg->rankId];
    ctx.myInputAddr.token = ctx.token[arg->rankId];

    // scratch 槽按 rank 编号；root 输入直接从 input 读取。
    ctx.scratchBase.addr = ctx.scratchAddr;
    ctx.scratchBase.token = ctx.token[arg->rankId];
    ctx.scratchSlot.token = ctx.token[arg->rankId];
    ctx.readDst.token = ctx.token[arg->rankId];

    CCU_IF(ctx.flag != 0)
    {
        // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
        ctx.dstAddr.addr += ctx.outputRepeatStride;
        ctx.myInputAddr.addr += ctx.inputRepeatStride;
    }
    // 每个远端 rank 的当前 slice 读入同号 scratch 槽；root 槽保留空位。
    ctx.scratchSlot.addr = ctx.scratchBase.addr;
    for (uint16_t rmtId = 0; rmtId < arg->rankSize; ++rmtId) {
        if (rmtId > 0) {
            ctx.scratchSlot.addr += ctx.normalSliceSize;
        }
        if (rmtId == arg->rootId) {
            continue;
        }
        ctx.remoteInputAddr.addr = srcAddr[rmtId];
        ctx.remoteInputAddr.token = ctx.token[rmtId];
        CCU_IF(ctx.flag != 0) { ctx.remoteInputAddr.addr += ctx.inputRepeatStride; }
        // channel 索引跳过本卡，scratch 槽索引不跳过。
        uint16_t channelId = rmtId < arg->rootId ? rmtId : rmtId - 1;
        ctx.readDst.addr = ctx.scratchSlot.addr;
        ccu::Read(
            arg->channels[channelId], ctx.readDst, ctx.remoteInputAddr, ctx.normalSliceSize, ctx.event, 1 << rmtId);
    }
    if (arg->rankSize > 1) {
        uint16_t allBit = ((1 << arg->rankSize) - 1) & (~(1 << arg->rankId));
        ccu::EventWait(ctx.event, allBit);
    }
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMeshMem2Mem1D remote read end, begin block-wise local reduce");

    // 归约源按 rank 排列，root 槽由本卡 input 代替。
    for (uint32_t r = 0; r < arg->rankSize; r++) {
        ctx.reduceSrc[r].token = ctx.scratchBase.token;
        if (r == 0) {
            ctx.reduceSrc[0].addr = ctx.scratchBase.addr;
        } else {
            ctx.reduceSrc[r].addr = ctx.reduceSrc[r - 1].addr;
            ctx.reduceSrc[r].addr += ctx.normalSliceSize;
        }
    }
    ctx.reduceSrc[arg->rootId].addr = ctx.myInputAddr.addr;
    ctx.reduceSrc[arg->rootId].token = ctx.myInputAddr.token;

    // 固定槽序归约，结果直接写入 root output。
    CCU_CHK_RET(GroupLocalReduce(
        ctx, ctx.dstAddr, ctx.reduceSrc, ctx.reduceGoSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp));
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMeshMem2Mem1D block-wise local reduce end");
    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuReduceMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto* kernelArg = static_cast<CcuKernelArgReduceMesh1DMem2Mem*>(arg);

    ReduceMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMesh1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    PreSync(ctx);

    CCU_IF(ctx.normalSliceSize != 0)
    {
        if (kernelArg->rankId == kernelArg->rootId) {
            ccu::Variable repeatNumAdd;
            ctx.flag = 0;
            repeatNumAdd = 1;
            CCU_WHILE(ctx.repeatNumVar != UINT64_MAX)
            { // 循环repeatNum_次
                CCU_CHK_RET(DoRepeatReduce(ctx, ctx.input, ctx.output[kernelArg->rankId]));
                ctx.repeatNumVar += repeatNumAdd;
                ctx.flag = 1;
            }
        }
    }

    PostSync(ctx);
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMesh1D end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl
