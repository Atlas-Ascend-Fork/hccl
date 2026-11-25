/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_REDUCE_SEQUENCE_3LEVEL_UTILS_H
#define HCCLV2_INS_V2_ALL_REDUCE_SEQUENCE_3LEVEL_UTILS_H

#include "executor_v2_base.h"
#include "executor_common_ops.h"
#include "alg_param.h"
#include "channel.h"
#include "coll_alg_v2_exec_registry.h"
#include "template_utils.h"
#include "cost_model.h"
#include "alg_template_base.h"
#ifndef AICPU_COMPILE
#include "hccl/hccl_res.h"
#endif

namespace ops_hccl {

/* 聚参 struct: 打包自由函数依赖的成员, 充当"成员 → 自由函数"的桥 (OmniPipeCostAxes 同款思路) */
struct ArSeq3LevelCtx {
    u64 dataTypeSize;
    u64 rankSizeLevel0;
    u64 rankSizeLevel1;
    u64 rankIdxLevel0;
    bool supportSymmetricMemory;
    u64 rsResultBuffOffset;
    u64 meshCommBuffOffset;
};

/* AllReduce 三级 Sequence executor 公共基类: 收敛 AICPU/DPU 两实现完全同名的成员,
 * 派生类成员访问名保持不变, .cc 实现无需改动。参照 omnipipe_executor_utils.h 先例。 */
class InsV2AllReduceSequence3LevelBase : public InsCollAlgBase {
public:
    InsV2AllReduceSequence3LevelBase() = default;
    ~InsV2AllReduceSequence3LevelBase() override = default;

protected:
    HcclResult SequenceInitCommInfo(
        const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, const char* executorName)
    {
        myRank_ = topoInfo->userRank;
        rankSize_ = topoInfo->userRankSize;
        reduceOp_ = param.reduceType;
        dataType_ = param.DataDes.dataType;
        dataCount_ = param.DataDes.count;
        dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
        algHierarchyInfo_ = algHierarchyInfo;
        HCCL_INFO(
            "[%s][InitCommInfo] myRank [%u], rankSize [%u], redOp [%u], "
            "dataType [%u] dataTypeSize [%llu]",
            executorName, myRank_, rankSize_, reduceOp_, dataType_, dataTypeSize_);
        return HCCL_SUCCESS;
    }

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};
    uint64_t rankSizeLevel2_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};
    uint64_t rankIdxLevel2_{0};

    bool skipLevel1_{false};

    uint64_t cclBuffSliceSize_{0};
    uint64_t rsResultBuffSize_{0};
    uint64_t meshCommBuffSize_{0};
    uint64_t rsResultBuffOffset_{0};
    uint64_t meshCommBuffOffset_{0};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;

    ArSeq3LevelCtx MakeArSeq3LevelCtx() const
    {
        return ArSeq3LevelCtx{dataTypeSize_,           rankSizeLevel0_,     rankSizeLevel1_,    rankIdxLevel0_,
                              supportSymmetricMemory_, rsResultBuffOffset_, meshCommBuffOffset_};
    }
};

/* GenBaseTempAlgParams: 6 次 SetTemplateBuffInfo, 两实现逐字相同 */
inline void ArSeq3LevelGenBaseBuffInfo(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateDataParams& tempAlgParamsRSL0,
    TemplateDataParams& tempAlgParamsRSL1, TemplateDataParams& tempAlgParamsRSL2, TemplateDataParams& tempAlgParamsAGL2,
    TemplateDataParams& tempAlgParamsAGL1, TemplateDataParams& tempAlgParamsAGL0)
{
    SetTemplateBuffInfo(
        tempAlgParamsRSL0, BufferType::INPUT, BufferType::HCCL_BUFFER, param.inputPtr, resCtx.cclMem.addr,
        resCtx.cclMem);
    SetTemplateBuffInfo(
        tempAlgParamsRSL1, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, resCtx.cclMem.addr, resCtx.cclMem.addr,
        resCtx.cclMem);
    SetTemplateBuffInfo(
        tempAlgParamsRSL2, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, resCtx.cclMem.addr, resCtx.cclMem.addr,
        resCtx.cclMem);
    SetTemplateBuffInfo(
        tempAlgParamsAGL2, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, resCtx.cclMem.addr, resCtx.cclMem.addr,
        resCtx.cclMem);
    SetTemplateBuffInfo(
        tempAlgParamsAGL1, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, resCtx.cclMem.addr, resCtx.cclMem.addr,
        resCtx.cclMem);
    SetTemplateBuffInfo(
        tempAlgParamsAGL0, BufferType::HCCL_BUFFER, BufferType::OUTPUT, resCtx.cclMem.addr, param.outputPtr,
        resCtx.cclMem);
}

/* GenTempAlgParamsRSL0: 两实现逐字相同 (仅日志 tag 不同) */
inline void ArSeq3LevelGenRSL0Params(
    const ArSeq3LevelCtx& ctx, const u64 loop, const u64 currDataCount, const u64 processedDataCount,
    TemplateDataParams& tempAlgParamsRSL0, const char* tag)
{
    tempAlgParamsRSL0.count = currDataCount;
    tempAlgParamsRSL0.buffInfo.inBuffBaseOff = processedDataCount * ctx.dataTypeSize;
    tempAlgParamsRSL0.buffInfo.outBuffBaseOff
        = ctx.supportSymmetricMemory ? processedDataCount * ctx.dataTypeSize : ctx.rsResultBuffOffset;
    tempAlgParamsRSL0.buffInfo.hcclBuffBaseOff = ctx.meshCommBuffOffset;

    tempAlgParamsRSL0.sliceSize = currDataCount / ctx.rankSizeLevel0 * ctx.dataTypeSize;
    tempAlgParamsRSL0.tailSize
        = (currDataCount / ctx.rankSizeLevel0 + currDataCount % ctx.rankSizeLevel0) * ctx.dataTypeSize;

    tempAlgParamsRSL0.inputSliceStride = tempAlgParamsRSL0.sliceSize;
    tempAlgParamsRSL0.outputSliceStride = ctx.supportSymmetricMemory ? tempAlgParamsRSL0.sliceSize : 0;

    HCCL_INFO(
        "[%s] loop [%llu] RSL0.inputSliceStride [%llu], "
        "RSL0.outputSliceStride [%llu], RSL0.sliceSize [%llu], RSL0.tailSize [%llu], "
        "RSL0.inBuffBaseOff [%llu], RSL0.outBuffBaseOff [%llu]",
        tag, loop, tempAlgParamsRSL0.inputSliceStride, tempAlgParamsRSL0.outputSliceStride, tempAlgParamsRSL0.sliceSize,
        tempAlgParamsRSL0.tailSize, tempAlgParamsRSL0.buffInfo.inBuffBaseOff,
        tempAlgParamsRSL0.buffInfo.outBuffBaseOff);

    tempAlgParamsRSL0.repeatNum = 1;
    tempAlgParamsRSL0.inputRepeatStride = 0;
    tempAlgParamsRSL0.outputRepeatStride = 0;
}

/* GenTempAlgParamsRSL1: 两实现逐字相同 (仅日志 tag 不同) */
inline void ArSeq3LevelGenRSL1Params(
    const ArSeq3LevelCtx& ctx, const u64 loop, const u64 currDataCount, const u64 sliceSizeRSL0, const u64 tailSizeRSL0,
    const u64 processedDataCount, TemplateDataParams& tempAlgParamsRSL1, const char* tag)
{
    tempAlgParamsRSL1.count = currDataCount;
    if (ctx.rankIdxLevel0 == ctx.rankSizeLevel0 - 1) {
        u64 tailCountRSL0 = tailSizeRSL0 / ctx.dataTypeSize;
        tempAlgParamsRSL1.sliceSize = tailCountRSL0 / ctx.rankSizeLevel1 * ctx.dataTypeSize;
        tempAlgParamsRSL1.tailSize
            = tempAlgParamsRSL1.sliceSize + tailCountRSL0 % ctx.rankSizeLevel1 * ctx.dataTypeSize;
    } else {
        u64 sliceCountRSL0 = sliceSizeRSL0 / ctx.dataTypeSize;
        tempAlgParamsRSL1.sliceSize = sliceCountRSL0 / ctx.rankSizeLevel1 * ctx.dataTypeSize;
        tempAlgParamsRSL1.tailSize
            = tempAlgParamsRSL1.sliceSize + sliceCountRSL0 % ctx.rankSizeLevel1 * ctx.dataTypeSize;
    }
    u64 symMemBaseOff
        = ctx.supportSymmetricMemory ? processedDataCount * ctx.dataTypeSize + ctx.rankIdxLevel0 * sliceSizeRSL0 : 0;
    tempAlgParamsRSL1.buffInfo.inBuffBaseOff = symMemBaseOff;
    tempAlgParamsRSL1.buffInfo.outBuffBaseOff = symMemBaseOff;
    tempAlgParamsRSL1.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsRSL1.inputSliceStride = tempAlgParamsRSL1.sliceSize;
    tempAlgParamsRSL1.outputSliceStride = tempAlgParamsRSL1.sliceSize;

    HCCL_INFO(
        "[%s] loop [%llu] RSL1.inputSliceStride [%llu], "
        "RSL1.outputSliceStride [%llu], RSL1.sliceSize [%llu], RSL1.tailSize [%llu], "
        "RSL1.inBuffBaseOff [%llu], RSL1.outBuffBaseOff [%llu]",
        tag, loop, tempAlgParamsRSL1.inputSliceStride, tempAlgParamsRSL1.outputSliceStride, tempAlgParamsRSL1.sliceSize,
        tempAlgParamsRSL1.tailSize, tempAlgParamsRSL1.buffInfo.inBuffBaseOff,
        tempAlgParamsRSL1.buffInfo.outBuffBaseOff);

    tempAlgParamsRSL1.repeatNum = 1;
    tempAlgParamsRSL1.inputRepeatStride = 0;
    tempAlgParamsRSL1.outputRepeatStride = 0;
}

/* GenTempAlgParamsAGL1: 两实现逐字相同 (仅日志 tag 不同) */
inline void ArSeq3LevelGenAGL1Params(
    const ArSeq3LevelCtx& ctx, const u64 loop, const u64 currDataCount, const u64 sliceSize, const u64 tailSize,
    const u64 symMemBaseOff, TemplateDataParams& tempAlgParamsAGL1, const char* tag)
{
    tempAlgParamsAGL1.count = currDataCount;
    u64 baseOff = ctx.supportSymmetricMemory ? symMemBaseOff : 0;
    tempAlgParamsAGL1.buffInfo.inBuffBaseOff = baseOff;
    tempAlgParamsAGL1.buffInfo.outBuffBaseOff = baseOff;
    tempAlgParamsAGL1.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsAGL1.sliceSize = sliceSize;
    tempAlgParamsAGL1.tailSize = tailSize;

    tempAlgParamsAGL1.inputSliceStride = tempAlgParamsAGL1.sliceSize;
    tempAlgParamsAGL1.outputSliceStride = tempAlgParamsAGL1.sliceSize;

    HCCL_INFO(
        "[%s] loop [%llu] AGL1.inputSliceStride [%llu], "
        "AGL1.outputSliceStride [%llu], AGL1.sliceSize [%llu], AGL1.tailSize [%llu], "
        "AGL1.inBuffBaseOff [%llu], AGL1.outBuffBaseOff [%llu]",
        tag, loop, tempAlgParamsAGL1.inputSliceStride, tempAlgParamsAGL1.outputSliceStride, tempAlgParamsAGL1.sliceSize,
        tempAlgParamsAGL1.tailSize, tempAlgParamsAGL1.buffInfo.inBuffBaseOff,
        tempAlgParamsAGL1.buffInfo.outBuffBaseOff);

    tempAlgParamsAGL1.repeatNum = 1;
    tempAlgParamsAGL1.inputRepeatStride = 0;
    tempAlgParamsAGL1.outputRepeatStride = 0;
}

/* GenTempAlgParamsAGL0: 两实现逐字相同 (仅日志 tag 不同) */
inline void ArSeq3LevelGenAGL0Params(
    const ArSeq3LevelCtx& ctx, const u64 loop, const u64 currDataCount, const u64 processedDataCount,
    const u64 sliceSize, const u64 tailSize, const u64 symMemBaseOff, TemplateDataParams& tempAlgParamsAGL0,
    const char* tag)
{
    tempAlgParamsAGL0.count = currDataCount;
    tempAlgParamsAGL0.buffInfo.inBuffBaseOff = ctx.supportSymmetricMemory ? symMemBaseOff : 0;
    tempAlgParamsAGL0.buffInfo.outBuffBaseOff = processedDataCount * ctx.dataTypeSize;
    tempAlgParamsAGL0.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsAGL0.sliceSize = sliceSize;
    tempAlgParamsAGL0.tailSize = tailSize;

    tempAlgParamsAGL0.inputSliceStride = 0;
    tempAlgParamsAGL0.outputSliceStride = tempAlgParamsAGL0.sliceSize;

    HCCL_INFO(
        "[%s] loop [%llu] AGL0.inputSliceStride [%llu], "
        "AGL0.outputSliceStride [%llu], AGL0.sliceSize [%llu], AGL0.tailSize [%llu], "
        "AGL0.inBuffBaseOff [%llu], AGL0.outBuffBaseOff [%llu]",
        tag, loop, tempAlgParamsAGL0.inputSliceStride, tempAlgParamsAGL0.outputSliceStride, tempAlgParamsAGL0.sliceSize,
        tempAlgParamsAGL0.tailSize, tempAlgParamsAGL0.buffInfo.inBuffBaseOff,
        tempAlgParamsAGL0.buffInfo.outBuffBaseOff);

    tempAlgParamsAGL0.repeatNum = 1;
    tempAlgParamsAGL0.inputRepeatStride = 0;
    tempAlgParamsAGL0.outputRepeatStride = 0;
}

/* 多轮 loop 固定开销折算（host-only）：运行态单轮容量有限，中大消息被切成 L=⌈dataSize/P⌉ 轮，
 * 每轮完整付出全部模板固定时延 C+D；cost 模型 C/D 只按单轮计一次 → 低估 L 倍。
 * HcclGetHcclBuffer 按 HCOMM 分层为 host-only API（设备镜像不可引用），与 host-only 方法
 * 守卫惯例一致用 AICPU_COMPILE 裁剪；设备侧算法名由 host 选定后传入，不重选，无需折算。 */
#ifndef AICPU_COMPILE
// 单轮处理量 P（字节），镜像 PrepareLoopBuffers 的布局与 maxCountPerLoop 公式：
// slot = cclMem/cclBuffSliceNum（对齐 dts，布局按 d0+1 切），单轮 = rsResult(1 slot)×d0；
// 失败/退化返回 0（不折算）。reservedBytes：最坏情况预扣（RHD 尾块 2·slot），
// 先扣再算，保证 P ≤ 运行态容量
inline u64 ArSeq3LevelCalcPerLoopDataSize(
    HcclComm comm, u64 cclBuffSliceNum, u64 rankSizeLevel0, u64 dataTypeSize, u64 totalRankAlign, u64 reservedBytes = 0)
{
    if (comm == nullptr || cclBuffSliceNum == 0 || rankSizeLevel0 == 0 || dataTypeSize == 0 || totalRankAlign == 0) {
        return 0;
    }
    void* cclBuffAddr = nullptr;
    uint64_t cclBuffSize = 0; // hccl_res.h 形参为 uint64_t（与 u64 同宽但不同类型）
    if (HcclGetHcclBuffer(comm, &cclBuffAddr, &cclBuffSize) != HCCL_SUCCESS || cclBuffSize <= reservedBytes) {
        return 0;
    }
    cclBuffSize -= reservedBytes;
    u64 rsResultBuffSize = cclBuffSize / cclBuffSliceNum / dataTypeSize * dataTypeSize;
    u64 maxLoopBytes = rsResultBuffSize * rankSizeLevel0;
    u64 maxCountPerLoop
        = maxLoopBytes / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / dataTypeSize / totalRankAlign * totalRankAlign;
    return maxCountPerLoop * dataTypeSize;
}

// 多轮固定开销折算进斜率：B += (C+D)/P，模型隐含轮数 1+x/P，对真实 ⌈x/P⌉ 的偏差恒 ≤ 一轮 C+D；
// 单轮区间（x≤P）模型值有界抬高 (C+D)·x/P，方向安全——高估使本算法小消息更难被选中，属期望行为。
// C/D 角色不变：C 保留在求和项，D 仍作 max 地板
inline void ArSeq3LevelFoldPerLoopFixedCost(std::vector<CostModelParam>& params, u64 perLoopDataSize)
{
    if (perLoopDataSize == 0) {
        return;
    }
    for (CostModelParam& p : params) {
        p.B += (p.C + p.D) / static_cast<float>(perLoopDataSize);
    }
}
#endif // AICPU_COMPILE

} // namespace ops_hccl

#endif // HCCLV2_INS_V2_ALL_REDUCE_SEQUENCE_3LEVEL_UTILS_H
