/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "order_launch.h"
#include <atomic>
#include "hcomm_dlsym.h"
#include "hcomm_primitives_dl.h"
#include "hccl_res_dl.h"
#include "dlhcomm_function.h"
#include "rt_external.h"
#include "acl/acl_rt.h"

namespace ops_hccl {

static HcclResult OpLaunchGetHostOrderStream(ThreadHandle hostOrderThread, aclrtStream& resolvedStream)
{
    void* hostOrderStream = nullptr;
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    if (!HcclThreadResGetInfoFunc.dlHcommThreadResGetInfo) {
        resolvedStream = nullptr;
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcommThreadResGetInfo is invalid.");
        return HCCL_SUCCESS;
    }
    HcclResult ret
        = HcclThreadResGetInfoFunc.dlHcommThreadResGetInfo(hostOrderThread, 0, sizeof(void*), &hostOrderStream);
    if (ret == HCCL_E_NOT_SUPPORT) {
        resolvedStream = nullptr;
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcommThreadResGetInfo is not supported.");
        return HCCL_SUCCESS;
    } else if (ret != HCCL_SUCCESS) {
        resolvedStream = nullptr;
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcommThreadResGetInfo not success.");
        return HCCL_SUCCESS;
    }
    resolvedStream = hostOrderStream;
    return HCCL_SUCCESS;
}

static const char* GetOrderLaunchModeName(OrderLaunchMode mode)
{
    if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {
        return "Aclgraph";
    } else if (mode == OrderLaunchMode::ORDER_LAUNCH_GE) {
        return "GE";
    }
    return "Opbase";
}

static HcclDedicatedThreadType GetOrderLaunchHostThreadType(OrderLaunchMode mode)
{
    if (mode == OrderLaunchMode::ORDER_LAUNCH_GE) {
        return HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE;
    } else if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {
        return HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH;
    }
    return HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE;
}

// 查询进程级缓存：-1未查询/0关闭/1开启，一次查询后粘滞
static std::atomic<int8_t>& GetAicpuOrderLaunchEnableCache()
{
    static std::atomic<int8_t> cache{-1};
    return cache;
}

// 查询HCOMM侧开关（进程级环境变量解析结果），dlsym缺失或查询失败fail-open默认开启
static bool IsAicpuOrderLaunchEnabled(HcclComm comm)
{
    int8_t cached = GetAicpuOrderLaunchEnableCache().load(std::memory_order_relaxed);
    if (cached >= 0) {
        return cached != 0;
    }
    auto& hcommFunction = DlHcommFunction::GetInstance();
    if (!hcommFunction.dlHcclConfigGetInfo) {
        // dlsym缺失不写缓存，保持后续补齐语义
        static std::atomic<bool> hasDlMissingLogged{false};
        if (!hasDlMissingLogged.exchange(true)) {
            HCCL_INFO("[%s] HcclConfigGetInfo is not supported, AICPU OrderLaunch use default enable.", __func__);
        }
        return true;
    }
    uint32_t aicpuOrderLaunchEnable = 0;
    HcclResult ret = hcommFunction.dlHcclConfigGetInfo(
        comm, static_cast<HcclConfigType>(HCCL_CONFIG_TYPE_AICPU_ORDER_LAUNCH_ENABLE), sizeof(aicpuOrderLaunchEnable),
        &aicpuOrderLaunchEnable);
    if (ret != HCCL_SUCCESS) {
        // fail-open并粘滞，避免混布旧HCOMM逐算子报内部ERROR
        HCCL_WARNING("[%s] query AICPU_ORDER_LAUNCH_ENABLE failed, ret[%d], use default enable.", __func__, ret);
        GetAicpuOrderLaunchEnableCache().store(1, std::memory_order_relaxed);
        return true;
    }
    int8_t enable = (aicpuOrderLaunchEnable != 0) ? 1 : 0;
    GetAicpuOrderLaunchEnableCache().store(enable, std::memory_order_relaxed);
    HCCL_DEBUG("[%s] AICPU OrderLaunch enable[%u].", __func__, aicpuOrderLaunchEnable);
    return enable != 0;
}

// 仅UT重置用，勿在生产路径调用
void ResetAicpuOrderLaunchCacheForTest() { GetAicpuOrderLaunchEnableCache().store(-1, std::memory_order_relaxed); }

// 保序跳过公共动作：句柄置零并关闭Phase2
void OrderLaunch::MarkOrderLaunchSkipped(OpParam& param)
{
    param.exportHostOrderThread = 0;
    param.deviceOrderThread = 0;
    orderLaunchRequired_ = false;
}

// Phase1保序准入：HCOMM版本（未知/过低）、开关、展开线程依次校验，任一不满足即跳过
static bool OrderLaunchAdmissionCheck(HcclComm comm, ThreadHandle unfoldThread, OrderLaunchMode mode)
{
    const char* modeName = GetOrderLaunchModeName(mode);
    // 版本号单次取值，避免条件与日志间取值漂移
    int hcommVersion = GetHcommVersion();
    if (hcommVersion == 0) {
        // 版本未知（查询失败）：无法判定枚举是否交付，保守跳过；去重WARNING供环境侧排查
        static std::atomic<bool> hasUnknownVersionLogged{false};
        if (!hasUnknownVersionLogged.exchange(true)) {
            HCCL_WARNING(
                "[%s]. hcommVersion is unknown (query failed), %s OrderLaunch is skipped for safety.", __func__,
                modeName);
        }
        return false;
    }
    // 保序线程枚举自9.2.0_beta.2起交付，低版本HCOMM不识别、禁止传递
    if (hcommVersion < CANN_VERSION(9, 2, 0, 2)) {
        // 进程级恒定状态，去重打印防刷屏
        static std::atomic<bool> hasLowVersionLogged{false};
        if (!hasLowVersionLogged.exchange(true)) {
            HCCL_INFO(
                "[%s]. hcommVersion[%d] is lower than 9.2.0_beta.2 (CANN_VERSION(9,2,0,2)), %s OrderLaunch is not "
                "supported, skipped.",
                __func__, hcommVersion, modeName);
        }
        return false;
    }

    // 开关关闭不做保序
    if (!IsAicpuOrderLaunchEnabled(comm)) {
        // 进程级恒定状态，去重打印防刷屏
        static std::atomic<bool> hasDisabledLogged{false};
        if (!hasDisabledLogged.exchange(true)) {
            HCCL_WARNING(
                "[%s]. AICPU OrderLaunch is disabled by HCCL_AICPU_ORDER_LAUNCH_ENABLE, %s OrderLaunch is "
                "skipped.",
                __func__, modeName);
        }
        return false;
    }

    // 未提前展开或不支持专用线程时无需保序
    if (unfoldThread == 0 || !HcommIsSupportHcclDedicatedThreadAcquire()) {
        HCCL_WARNING(
            "[%s]. unfoldThread is null or HcclDedicatedThreadAcquire not supported, %s OrderLaunch is skipped.",
            __func__, modeName);
        return false;
    }
    return true;
}

/**
 * @brief ACLGRAPH模式：将hostOrderThread对应的hostOrderStream经rtStreamAddToModel显式加入capture模型。
 * 替代原event0的归属传播入图：显式入图的流不参与EndCapture的join-back拓扑校验，且hostOrderStream上的
 * notify任务仍可被采集为图action（replay保序依赖）。实现模式对齐op_common.cc的CaptureSlaveStreams()。
 * @param stream          算子执行流（用于查询capture状态与rtModel）
 * @param hostOrderThread Host侧保序线程句柄
 * @return HcclResult
 */
static HcclResult AclgraphOrderLaunchAddHostOrderStreamToModel(aclrtStream stream, ThreadHandle hostOrderThread)
{
    // 1、反查hostOrderThread对应的hostOrderStream
    aclrtStream hostOrderStream = nullptr;
    CHK_RET(OpLaunchGetHostOrderStream(hostOrderThread, hostOrderStream));
    CHK_PRT_RET(
        hostOrderStream == nullptr,
        HCCL_ERROR(
            "[%s] failed to get hostOrderStream, hostOrderThread[0x%llx], dlsym missing or resGetInfo failed.",
            __func__, hostOrderThread),
        HCCL_E_RUNTIME);

    // 2、查询算子执行流的capture状态，取rtModel（isCapture由同流实时查询置位，此处非ACTIVE属状态异常：
    // 升级告警留痕供replay问题定位；仍按成功降级——若capture恰在本op下发前结束，握手与kernel按eager执行可正确完成）
    aclmdlRI rtModel = nullptr;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclError ret = aclmdlRICaptureGetInfo(stream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[%s]Stream capture is not supported, skip adding hostOrderStream to model.", __func__);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s]aclmdlRICaptureGetInfo fail. return[%d].", __func__, ret), HCCL_E_RUNTIME);
    if (captureStatus != aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
        HCCL_WARNING(
            "[%s]captureStatus[%d] is not active in ACLGRAPH mode, skip adding hostOrderStream to model; notify "
            "handshake still proceeds, replay may hang if stream keeps capturing.",
            __func__, captureStatus);
        return HCCL_SUCCESS;
    }

    // 3、显式入图（幂等：同流重复add同一模型为no-op，与CaptureSlaveStreams每op重复add一致）
    rtError_t addRet = rtStreamAddToModel(hostOrderStream, rtModel);
    CHK_PRT_RET(
        addRet != RT_ERROR_NONE, HCCL_ERROR("[%s]rtStreamAddToModel fail. return[%d].", __func__, addRet),
        HCCL_E_RUNTIME);
    HCCL_INFO("[%s]add hostOrderStream to model success, stream[%p], rtModel[%p]", __func__, hostOrderStream, rtModel);
    return HCCL_SUCCESS;
}

/**
 * @brief OrderLaunch第一阶段资源准备：获取超时时间、确定启动模式、GE模式覆盖展开线程。
 * @param comm          通信域
 * @param param         算子参数
 * @param unfoldThread  展开线程句柄，GE模式下可能被覆盖
 * @return HcclResult
 */
HcclResult OrderLaunch::HcclOrderLaunchPrepare(HcclComm comm, const OpParam& param, ThreadHandle& unfoldThread)
{
    // 超时取自param；ExecTimeoutManager仅在AICPU侧被Set，host侧读取恒为默认值
    execTimeout_ = param.opConfig.execTimeout;

    launchMode_ = param.isCapture ? OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH :
                                    (param.opMode == OpMode::OFFLOAD ? OrderLaunchMode::ORDER_LAUNCH_GE :
                                                                       OrderLaunchMode::ORDER_LAUNCH_OPBASE);
    const char* modeName = GetOrderLaunchModeName(launchMode_);

    // GE模式通过HcclDedicatedThreadAcquire获取GE专用展开线程，覆盖unfoldThread；
    // 不支持时置0，跳过语义由Phase1的unfoldThread==0判空兜底，不提前return
    if (launchMode_ == OrderLaunchMode::ORDER_LAUNCH_GE) {
        if (!HcommIsSupportHcclDedicatedThreadAcquire()) {
            unfoldThread = 0;
            HCCL_WARNING("[%s]. HcclDedicatedThreadAcquire not supported, GE OrderLaunch is skipped.", __func__);
        } else {
            ThreadHandle geUnfoldThread = 0;
            CHK_RET(HcclDedicatedThreadAcquire(
                comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, ORDER_UNFOLD_THREAD_NOTIFY_NUM, &geUnfoldThread));
            unfoldThread = geUnfoldThread;
            HCCL_INFO("[%s]. GE unfoldThread[0x%llx]", __func__, unfoldThread);
        }
    }

    HCCL_INFO("[%s]. %s OrderLaunch Prepare success, unfoldThread[0x%llx]", __func__, modeName, unfoldThread);
    return HCCL_SUCCESS;
}

/**
 * @brief 按序launch第一阶段：将通信算子按序launch到OrderStream。
 * OPBASE / ACLGRAPH / HCOMM 三模式统一入口，流程如下：
 * 1. 获取并导出Host侧保序流（hostOrderThread），写入param.exportHostOrderThread；
 * 2. 仅ACLGRAPH模式：hostOrderStream经rtStreamAddToModel显式加入capture模型；
 * 3. 获取Device侧保序流（deviceOrderThread），写入param.deviceOrderThread；
 * 4. hostOrderThread在unfoldThread上record notify(idx)，unfoldThread等待该notify，完成第一阶段同步。
 * @param comm          通信域
 * @param param         算子参数，函数会写入exportHostOrderThread和deviceOrderThread
 * @param unfoldThread  展开线程句柄
 * @param notifyIdx     notify索引
 * @return HcclResult
 */
HcclResult
OrderLaunch::HcclOrderLaunchToOrderStream(HcclComm comm, OpParam& param, ThreadHandle unfoldThread, u32 notifyIdx)
{
    const char* modeName = GetOrderLaunchModeName(launchMode_);
    HcclDedicatedThreadType hostThreadType = GetOrderLaunchHostThreadType(launchMode_);
    HCCL_INFO("[%s]. %s OrderLaunch Phase1 Start, Comm[%p], timeout[%u s].", __func__, modeName, comm, execTimeout_);

    // 准入校验不通过即跳过保序
    if (!OrderLaunchAdmissionCheck(comm, unfoldThread, launchMode_)) {
        MarkOrderLaunchSkipped(param);
        return HCCL_SUCCESS;
    }

    // 1.1、获取Host侧保序流
    ThreadHandle hostOrderThread;
    ThreadHandle exportHostOrderThread;
    CHK_RET(HcclDedicatedThreadAcquire(comm, hostThreadType, HOST_ORDER_THREAD_NOTIFY_NUM, &hostOrderThread));
    HCCL_INFO(
        "[%s]. %s After HcclDedicatedThreadAcquire hostOrderThread [0x%llx]", __func__, modeName, hostOrderThread);
    if (hostOrderThread == 0) {
        MarkOrderLaunchSkipped(param);
        HCCL_INFO(
            "[%s]. Communication domains Number is less than cores Number, %s OrderLaunch is not Required.", __func__,
            modeName);
        return HCCL_SUCCESS;
    }
    // 缓存句柄供Phase2复用，避免二次acquire结果漂移
    hostOrderThread_ = hostOrderThread;

    // 1.2、导出Host侧保序流
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &hostOrderThread, COMM_ENGINE_AICPU_TS, &exportHostOrderThread));
    param.exportHostOrderThread = exportHostOrderThread;
    HCCL_INFO(
        "[%s]. %s After HcclThreadExportToCommEngine hostOrderThread [0x%llx], exportHostOrderThread[0x%llx]", __func__,
        modeName, hostOrderThread, exportHostOrderThread);

    // 2、仅 ACLGRAPH 模式：hostOrderStream 经 rtStreamAddToModel 显式加入 capture 模型
    if (launchMode_ == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {
        CHK_RET(AclgraphOrderLaunchAddHostOrderStreamToModel(param.stream, hostOrderThread));
    }

    // 3、获取Device侧保序流
    ThreadHandle deviceOrderThread;
    CHK_RET(HcclDedicatedThreadAcquire(
        comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, DEVICE_ORDER_THREAD_NOTIFY_NUM, &deviceOrderThread));
    if (deviceOrderThread == 0) {
        MarkOrderLaunchSkipped(param);
        HCCL_INFO(
            "[%s]. HcclDedicatedThreadAcquire unable to obtain deviceOrderThread, %s OrderLaunch is not Required.",
            __func__, modeName);
        return HCCL_SUCCESS;
    }
    param.deviceOrderThread = deviceOrderThread;
    HCCL_INFO(
        "[%s]. %s After HcclDedicatedThreadAcquire deviceOrderThread [0x%llx]", __func__, modeName, deviceOrderThread);

    // 4、notify0
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(hostOrderThread, unfoldThread, notifyIdx)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(unfoldThread, notifyIdx, execTimeout_)));
    HCCL_INFO("[%s]. %s OrderLaunch Phase1 Success, timeout[%u s].", __func__, modeName, execTimeout_);
    return HCCL_SUCCESS;
}

/**
 * @brief 按序launch第二阶段：将通信算子按序launch到KernelStream。
 * OPBASE / ACLGRAPH / HCOMM 三模式统一入口，流程如下：
 * 1. 复用Phase1缓存的Host侧保序线程句柄（hostOrderThread），不再二次acquire；
 * 2. hostOrderThread等待第一阶段record的notify(idx)，完成第二阶段同步。
 * @param comm          通信域
 * @param notifyIdx     notify索引
 * @return HcclResult
 */
HcclResult OrderLaunch::HcclOrderLaunchToKernelStream(HcclComm comm, u32 notifyIdx)
{
    const char* modeName = GetOrderLaunchModeName(launchMode_);
    HCCL_INFO("[%s]. %s OrderLaunch Phase2 Start, Comm[%p], timeout[%u s].", __func__, modeName, comm, execTimeout_);

    // 第一阶段已判定无需保序，直接跳过
    if (!orderLaunchRequired_) {
        HCCL_INFO("[%s]. %s OrderLaunch Phase2 is skipped, orderLaunchRequired is false.", __func__, modeName);
        return HCCL_SUCCESS;
    }

    // 1、复用Phase1缓存的Host侧保序线程句柄，避免二次acquire结果漂移
    ThreadHandle hostOrderThread = hostOrderThread_;
    CHK_PRT_RET(
        hostOrderThread == 0,
        HCCL_ERROR(
            "[%s]. %s hostOrderThread cached in Phase1 is null while orderLaunchRequired is true, internal state "
            "error.",
            __func__, modeName),
        HCCL_E_INTERNAL);
    HCCL_INFO("[%s]. %s reuse Phase1 hostOrderThread [0x%llx]", __func__, modeName, hostOrderThread);

    // 2、wait notify1
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(hostOrderThread, notifyIdx, execTimeout_)));

    HCCL_INFO("[%s]. %s OrderLaunch Phase2 Success.timeout[%u s].", __func__, modeName, execTimeout_);
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
