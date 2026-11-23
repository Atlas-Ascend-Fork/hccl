/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "adapter_acl.h"
#include "auto_selector_base.h"
#include "hccl_aiv_utils.h"
#include "hccl_sym_win_dl.h"

// op_common.cc IsSymWinLsaCovered 引用，真实实现定义在 hccl_sym_win_dl.cc（本 UT 不链接）。
// AIV 引擎分支会校验 LSA 覆盖：按旧版 hcomm（rank == memberId）视为已覆盖。
extern "C" bool RefreshSymWinRankToMember(HcclComm comm, HcclCommSymWindow winHandle, uint32_t rankSize)
{
    (void)comm;
    (void)winHandle;
    (void)rankSize;
    return true;
}

// op_common.cc IsSymWinLsaCovered 引用，真实实现由 hcomm 动态库提供（本 UT 不链接）。
// 桩返回固定 rankSize，配合上面的 memberId 换算把 AIV 用例判定为 LSA 已覆盖。
extern "C" HcclResult HcclGetRankSize(HcclComm comm, uint32_t* rankSize)
{
    (void)comm;
    if (rankSize == nullptr) {
        return HCCL_E_PARA;
    }
    *rankSize = 8;
    return HCCL_SUCCESS;
}

// op_common.cc FillSymPeerAddrArray 引用（同属 hccl_sym_win_dl.cc，本 UT 不链接）：
// 门禁用例不构造对端对称 VA 表，按映射未登记处理。
extern "C" bool GetSymWinRankToMemberTable(HcclCommSymWindow winHandle, uint32_t rankSize, uint32_t* memberIds)
{
    (void)winHandle;
    (void)rankSize;
    (void)memberIds;
    return false;
}

// op_common.cc FillSymPeerAddrArray 引用：视为旧版 hcomm，未导出 host 侧对端取址接口。
extern "C" bool HcommIsSupportHcclSymWinGetPeerDevicePointer(void) { return false; }

// op_common.cc HcclAivCacheCheckAndReplay 引用（acl runtime 提供）：UT 无 device 上下文，固定返回可用核数。
extern "C" aclError aclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t* value)
{
    (void)type;
    *value = 8;
    return ACL_SUCCESS;
}

namespace ops_hccl {
// op_common.cc 同时引用 AIV/CCU 分支的依赖；对称内存门禁用例不应触发，仅用于链接。
HcclResult haclrtMemcpy(void*, size_t, const void*, size_t, aclrtMemcpyKind)
{
    ADD_FAILURE() << "对称内存门禁用例不应调用haclrtMemcpy";
    return HCCL_E_NOT_SUPPORT;
}

HcclResult haclrtMemset(void*, size_t, int32_t, size_t)
{
    ADD_FAILURE() << "对称内存门禁用例不应调用haclrtMemset";
    return HCCL_E_NOT_SUPPORT;
}

HcclResult ClearAivTagCb(HcclComm, HcclCommStatePhase, void*)
{
    ADD_FAILURE() << "对称内存门禁用例不应调用ClearAivTagCb";
    return HCCL_E_NOT_SUPPORT;
}

// op_common.cc CheckNdaMainboardSupported 引用，定义在 adapter_acl.cc（本 UT 不链接），按取不到主板信息处理
HcclResult hcalrtGetDeviceInfo(u32 deviceId, aclrtDevAttr devAttr, s64& val, bool quiet)
{
    (void)deviceId;
    (void)devAttr;
    (void)val;
    (void)quiet;
    return HCCL_E_NOT_SUPPORT;
}

// op_common.cc IsUbxAicpuTsAllToAllSymAlg 引用，真实实现定义在 auto_selector_base.cc（本 UT 不链接）。
// 门禁用例不构造 UBX alltoall 拓扑，按非 concurrent 拓扑处理。
HcclResult AutoSelectorBase::CheckMeshNumEqualToClosNum(const TopoInfoWithNetLayerDetails* topoInfo, bool& isEqual)
{
    (void)topoInfo;
    isEqual = false;
    return HCCL_SUCCESS;
}

// ---- AIV 指令集缓存重放路径（hccl_aiv_utils.cc 未编入本 UT）----
// 缓存统一返回一条「对称模式下录制的 AllGather 指令集」，门禁用例只验证重放前的对称能力探测。
u64 CalcAivCacheKeyHash(const AivOpCacheArgs& cacheKey)
{
    (void)cacheKey;
    return 0x5A5AU;
}

HcclResult BuildAivCacheCtxTag(u64 keyHash, std::string& ctxTag)
{
    ctxTag = std::to_string(keyHash);
    return HCCL_SUCCESS;
}

std::mutex& GetAivCacheMutex()
{
    static std::mutex mutex;
    return mutex;
}

HcclResult LookupAivCacheCtx(
    HcclComm comm, const std::string& ctxTag, u64 keyHash, bool& cacheHit, std::string& algName,
    AivInstruction*& instructions, u32& insCount, u32& symMode, AivSymSign& symSign)
{
    (void)comm;
    (void)ctxTag;
    (void)keyHash;
    cacheHit = true;
    algName = "AivAllGatherSoleMesh";
    instructions = nullptr;
    insCount = 0;
    symMode = 1;
    symSign.inWin = 0x1000U;
    symSign.outWin = 0x2000U;
    return HCCL_SUCCESS;
}

HcclResult ReplayAivInstructions(const AivInstruction* instructions, u32 insCount, OpParam& param)
{
    (void)instructions;
    (void)insCount;
    (void)param;
    ADD_FAILURE() << "inplace 调用不应命中缓存并回放对称指令集";
    return HCCL_E_NOT_SUPPORT;
}

HcclResult ReplayAivInstructionsV(const AivInstruction* instructions, u32 insCount, OpParam& param)
{
    (void)instructions;
    (void)insCount;
    (void)param;
    ADD_FAILURE() << "inplace 调用不应命中缓存并回放对称指令集";
    return HCCL_E_NOT_SUPPORT;
}

bool IsAivSymPeerTableFilled(const void* commInfo, const AivSymSign& sign)
{
    (void)commInfo;
    (void)sign;
    return true;
}

void RecordAivSymPeerTable(const void* commInfo, const AivSymSign& sign)
{
    (void)commInfo;
    (void)sign;
}
} // namespace ops_hccl
