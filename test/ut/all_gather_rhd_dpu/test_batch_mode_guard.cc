/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// InsTempAllGatherRHDDPU KernelRun 的 batch 模式守卫（BatchModeGuard）覆盖：
// 5 个中间错误出口（HcommBatchModeEnd/HcommThreadSynchronize/HcommSendRequest/
// HcommWaitResponse 失败、recvMsgId 不一致）任一命中时，KernelRun 返回 HCCL_E_INTERNAL
// 且守卫析构恰恢复 1 次 HcommBatchModeStart；成功路径 Dismiss 后显式恢复恰 1 次，
// 无重复 Start。守卫构造在 HcommBatchModeEnd 之前（Issue #960 修复增强，dev-r18 意见 1），
// End 失败出口位于保护窗口内——该用例在守卫构造位于 End 之后时会以 startCount==0 失败。

#include "ag_rhd_test_access.h"

#include <cstdio>

namespace ag_rhd_ut {
namespace {

    constexpr u32 kRankSize = 2;

    InsTempAllGatherRHDDPU MakeTemplate()
    {
        InsTempAllGatherRHDDPU tpl;
        SetupTemplate(tpl, 0, MakeRankIds(kRankSize));
        return tpl;
    }

    TemplateResource MakeResource()
    {
        TemplateResource resource;
        resource.threads = {1};
        resource.npu2DpuShmemPtr = reinterpret_cast<void*>(0x4000);
        resource.dpu2NpuShmemPtr = reinterpret_cast<void*>(0x5000);
        return resource;
    }

    OpParam MakeOpParam()
    {
        OpParam param;
        std::snprintf(param.algTag, sizeof(param.algTag), "%s", "ag_rhd_ut_tag");
        param.DataDes.dataType = HCCL_DATA_TYPE_FP16;
        param.opConfig.execTimeout = 7;
        return param;
    }

    // 5 个错误出口共享的守卫断言口径：E_INTERNAL + 恢复 Start 恰 1 次 + End 已被调用恰 1 次
    void ExpectErrorExitRestoresBatch(HcclResult ret, const TemplateDataParams& params)
    {
        EXPECT_EQ(ret, HCCL_E_INTERNAL);
        EXPECT_EQ(BatchModeCounts().startCount, 1u);
        EXPECT_EQ(BatchModeCounts().endCount, 1u);
        // LocalDataCopy 已完成（每 rpt 份 1 条预拷贝）→ 用例实际到达 End 调用点而非更早出口
        EXPECT_EQ(CopyCalls().size(), params.repeatNum);
    }

    // 出口 1：HcommBatchModeEnd 自身失败——守卫构造前置于 End 后此出口进入保护窗口
    // （HCOMM SetLaunchMode 先置 mode 后收尾，End 失败时 thread_local 已残留 EAGER，
    // 析构恢复 Start 恰 1 次）。守卫若仍在 End 之后构造，本用例 startCount 将为 0。
    TEST(AgRhdBatchModeGuardTest, KernelRunRestoresBatchWhenBatchModeEndFails)
    {
        InsTempAllGatherRHDDPU tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeParams();
        TemplateResource resource = MakeResource();
        ResetStubState();
        SetBatchModeEndResult(HCCL_E_INTERNAL);

        ExpectErrorExitRestoresBatch(tpl.KernelRun(param, params, resource), params);
    }

    // 出口 2：HcommThreadSynchronize 失败（End 已成功切换 eager）
    TEST(AgRhdBatchModeGuardTest, KernelRunRestoresBatchWhenThreadSyncFails)
    {
        InsTempAllGatherRHDDPU tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeParams();
        TemplateResource resource = MakeResource();
        ResetStubState();
        SetThreadSyncResult(HCCL_E_INTERNAL);

        ExpectErrorExitRestoresBatch(tpl.KernelRun(param, params, resource), params);
    }

    // 出口 3：HcommSendRequest 失败（DPURunInfo 序列化后下发失败）
    TEST(AgRhdBatchModeGuardTest, KernelRunRestoresBatchWhenSendRequestFails)
    {
        InsTempAllGatherRHDDPU tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeParams();
        TemplateResource resource = MakeResource();
        ResetStubState();
        SetSendRequestResult(HCCL_E_INTERNAL);

        ExpectErrorExitRestoresBatch(tpl.KernelRun(param, params, resource), params);
    }

    // 出口 4：HcommWaitResponse 失败（等待 DPU 回写失败）
    TEST(AgRhdBatchModeGuardTest, KernelRunRestoresBatchWhenWaitResponseFails)
    {
        InsTempAllGatherRHDDPU tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeParams();
        TemplateResource resource = MakeResource();
        ResetStubState();
        SetWaitResponseResult(HCCL_E_INTERNAL);

        ExpectErrorExitRestoresBatch(tpl.KernelRun(param, params, resource), params);
    }

    // 出口 5：recvMsgId != sendMsgId（注入 WaitResponse 写入不匹配的 msgId，两原语均返回成功）
    TEST(AgRhdBatchModeGuardTest, KernelRunRestoresBatchWhenMsgIdMismatch)
    {
        InsTempAllGatherRHDDPU tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeParams();
        TemplateResource resource = MakeResource();
        ResetStubState();
        SetSendMsgId(1);
        SetWaitResponseMsgId(2);

        ExpectErrorExitRestoresBatch(tpl.KernelRun(param, params, resource), params);
    }

    // 成功路径：Dismiss 后显式恢复恰 1 次 Start（无重复 Start），返回 HCCL_SUCCESS；
    // PostLocalCopy 走真拷贝回搬（每 rpt 份 kRankSize 条）→ 预拷贝 + 回搬拷贝总数旁证走完全程
    TEST(AgRhdBatchModeGuardTest, KernelRunHappyPathRestoresBatchExactlyOnce)
    {
        InsTempAllGatherRHDDPU tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeParams();
        TemplateResource resource = MakeResource();
        ResetStubState();

        EXPECT_EQ(tpl.KernelRun(param, params, resource), HCCL_SUCCESS);
        // 显式恢复恰 1 次；Dismiss 后析构不再补 Start（重复 Start 会破坏 Start/End 配对）
        EXPECT_EQ(BatchModeCounts().startCount, 1u);
        EXPECT_EQ(BatchModeCounts().endCount, 1u);
        EXPECT_EQ(CopyCalls().size(), params.repeatNum * (1 + kRankSize));
    }

} // namespace
} // namespace ag_rhd_ut
