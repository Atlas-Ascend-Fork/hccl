/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AG_RHD_TEST_ACCESS_H
#define AG_RHD_TEST_ACCESS_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

// 被测 KernelRun 消费的域成员为基类保护成员，UT 目标通过 -fno-access-control 放开访问权限
#include "ins_temp_all_gather_rhd_dpu.h"

namespace ag_rhd_ut {
using namespace ops_hccl;

// LocalCopy 桩记录的一次拷贝调用：断言 LocalDataCopy 已完成（到达 End 调用点的旁证）
struct CopyCall {
    void* srcAddr = nullptr;
    u64 srcOffset = 0;
    u64 srcSize = 0;
    void* dstAddr = nullptr;
    u64 dstOffset = 0;
    u64 dstSize = 0;
};

std::vector<CopyCall>& CopyCalls();

// batch 模式窗口计数：BatchModeGuard 行为的核心断言面（恢复恰 1 次 Start / 无重复 Start）
struct BatchModeStats {
    u32 startCount = 0;
    u32 endCount = 0;
};

BatchModeStats& BatchModeCounts();

// HCOMM 原语返回值注入：覆盖 KernelRun 的 5 个中间错误出口
void SetBatchModeStartResult(s32 result);
void SetBatchModeEndResult(s32 result);
void SetThreadSyncResult(s32 result);
void SetSendRequestResult(s32 result);
void SetWaitResponseResult(s32 result);
void SetSendMsgId(u32 msgId);
void SetWaitResponseMsgId(u32 msgId);
void ResetStubState();

// 构造最小被测实例：默认构造（注册表路径同款）+ 直写域成员（对齐 nhr UT 的 SetupTemplate 手法）
inline void SetupTemplate(InsTempAllGatherRHDDPU& tpl, u32 myRank, const std::vector<u32>& rankIds)
{
    tpl.myRank_ = myRank;
    tpl.subCommRanks_ = {rankIds};
    tpl.templateRankSize_ = static_cast<u32>(rankIds.size());
}

inline std::vector<u32> MakeRankIds(u32 rankSize)
{
    std::vector<u32> rankIds(rankSize);
    for (u32 i = 0; i < rankSize; ++i) {
        rankIds[i] = i;
    }
    return rankIds;
}

// 最小可跑参数：sliceSize 非零（跳过零切片早退）、输入与 CCL 缓冲不同址（预拷贝必发生）、
// outBuffType 保持 DEFAULT（真拷贝绑定，PostLocalCopy 走恒等回搬分支）
inline TemplateDataParams MakeParams()
{
    TemplateDataParams params;
    params.sliceSize = 16;
    params.count = 4;
    params.inputSliceStride = 32;
    params.outputSliceStride = 16;
    params.inputRepeatStride = 1000;
    params.outputRepeatStride = 70;
    params.repeatNum = 2;
    params.buffInfo.inputPtr = reinterpret_cast<void*>(0x1000);
    params.buffInfo.outputPtr = reinterpret_cast<void*>(0x3000);
    params.buffInfo.hcclBuff.type = HcclMemType::HCCL_MEM_TYPE_DEVICE;
    params.buffInfo.hcclBuff.addr = reinterpret_cast<void*>(0x2000);
    params.buffInfo.hcclBuff.size = 4096;
    params.buffInfo.inBuffBaseOff = 8;
    params.buffInfo.outBuffBaseOff = 4;
    params.buffInfo.hcclBuffBaseOff = 40;
    return params;
}

} // namespace ag_rhd_ut

#endif // AG_RHD_TEST_ACCESS_H
