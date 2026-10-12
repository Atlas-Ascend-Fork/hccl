/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_op_experimental.h"
#include "reduce_scatter_op_experimental.h"
#include "topo_host.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>
#include "op_common_experimental.h"
#include "param_check.h"
#include "all_reduce.h"
#include "load_kernel.h"
#include "hcomm_dlsym.h"
#include "hcomm_host_profiling_dl.h"

using namespace std;

extern "C" HcclResult HcclAllReduceInner(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream);

namespace ops_hccl_experimental {
using ops_hccl::AllReduceEntryLog;
using ops_hccl::DATATYPE_SIZE_TABLE;
using ops_hccl::HcclNetLayer;
using ops_hccl::LoadAICPUKernel;
using ops_hccl::LogHcclExit;
using ops_hccl::OpMode;

static bool MatchBIAR(HcclComm comm)
{
    if (!MatchBIRS()) {
        return false;
    }
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    if (HcclGetRankSize(comm, &rankSize) != HCCL_SUCCESS) {
        return false;
    }
    HcclDevType deviceType = HcclDevType::DEV_TYPE_COUNT;
    if (HcclGetDeviceType(deviceType) != HCCL_SUCCESS) {
        return false;
    }
    uint32_t* instSizeList = nullptr;
    uint32_t serverNum = 0;
    if (HcclRankGraphGetInstSizeListByLayer(
            comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0), &instSizeList, &serverNum)
        != HCCL_SUCCESS) {
        return false;
    }
    // 不适用条件统一回退老路径, 须与 DecideAllReduceBiarAlg 的拒绝条件保持一致(selector 仅作兜底),
    // 避免奇数规模静默回退而小规模偶数硬报错的分裂处置
    if (rankSize == 1 || deviceType != HcclDevType::DEV_TYPE_910_93 || rankSize % 2 != 0 || serverNum == 0
        || rankSize / serverNum < 4) {
        HCCL_INFO(
            "AllReduceBIAR not applicable: deviceType[%d] rankSize[%u] serverNum[%u], fallback to the default "
            "AllReduce",
            deviceType, rankSize, serverNum);
        return false;
    }
    return true;
}

HcclResult AllReduceExperimental(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    if (!MatchBIAR(comm)) {
        return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);
    }

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return all reduce success"), HCCL_SUCCESS);

    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内
    OpParam param;
    CHK_RET(AllReduceInitAndCheck(comm, sendBuf, recvBuf, count, dataType, op, stream, param));

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));

    /* 接口交互信息日志 */
    CHK_RET(AllReduceEntryLog(sendBuf, recvBuf, count, dataType, op, stream, param.tag, "HcclAllReduce"));

    CHK_RET(AllReduceOutPlaceCustom(param, sendBuf, recvBuf, count, dataType, op, comm, stream, rankSize));

    CHK_RET(LogHcclExit("HcclAllReduce", param.tag, startut));

    return HCCL_SUCCESS;
}

static HcclResult PrepareAllReduceParam(
    OpParam& param, void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream, OpMode opMode)
{
    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 outputSize = count * perDataSize;
    u64 inputSize = outputSize;

    param.stream = stream;
    param.reduceType = op;
    param.opMode = opMode;

    if (param.commName[0] == '\0') {
        CHK_RET(HcclGetCommName(comm, param.commName));
    }
    HcclDevType deviceType = HcclDevType::DEV_TYPE_COUNT;
    CHK_RET(HcclGetDeviceType(deviceType));

    param.inputSize = inputSize;
    param.inputPtr = sendBuf;
    param.outputSize = outputSize;
    param.outputPtr = recvBuf;
    param.DataDes.dataType = dataType;
    param.DataDes.count = count;
    param.enableDetour = false;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    param.deviceType = deviceType;

    return HCCL_SUCCESS;
}

HcclResult AllReduceOutPlaceCustom(
    OpParam& param, void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream, u32 userRankSize)
{
    HCCL_INFO("Start to execute AllReduceOutPlaceCustom");
    CHK_RET(PrepareAllReduceParam(param, sendBuf, recvBuf, count, dataType, op, comm, stream, OpMode::OPBASE));

    if (IsAiCpuMode(param.deviceType, userRankSize)) {
        HCCL_DEBUG("is aicpu mode");
        CHK_RET(LoadAICPUKernel());
        param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    } else {
        HCCL_DEBUG("is host mode");
        param.engine = CommEngine::COMM_ENGINE_CPU_TS;
    }

    uint64_t beginTime = 0;
    if (HcommIsProfilingSupported()) {
        beginTime = HcommGetProfilingSysCycleTime();
    }

    CHK_RET(ProcessA3(comm, param, beginTime));

    HCCL_INFO("Execute AllReduceOutPlaceCustom success.");
    return HCCL_SUCCESS;
}

} // namespace ops_hccl_experimental
