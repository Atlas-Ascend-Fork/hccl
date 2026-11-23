/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INCONSISTENT_CHECK_H
#define INCONSISTENT_CHECK_H

#include <arpa/inet.h>

#include <string>

#include "alg_param.h"
#include "alg_type.h"

namespace ops_hccl {
bool NeedInconsistentCheck(HcclComm comm, const OpParam& param);

bool CheckCtxStatus(HcclComm comm, const OpParam& param);

HcclResult CompareOpExchangeInfos(
    HcclComm comm, const OpParam& param, const AlgResourceRequest& resRequest, const OpExchangeInfo& exchangeInfo);

// 一致性校验失败日志上下文：聚合本端/对端 rank 与 IP
struct CheckFailLogCtx {
    uint32_t localRank;
    uint32_t remoteRank;
    char remoteIp[INET6_ADDRSTRLEN];
    char localIp[INET6_ADDRSTRLEN];
};

HcclResult InconsistentCheckParams(
    HcclComm comm, const OpExchangeInfo& exchangeInfo, const std::vector<HcclChannelDesc>& channels);

HcclResult InconsistentCheckOpType(
    const CheckFailLogCtx& logCtx, const OpExchangeInfo& exchangeInfo, const HcclCMDType& rmtOpType);

HcclResult ReportOpExchangeInfoCheckFailed(
    const CheckFailLogCtx& logCtx, const OpExchangeInfo& exchangeInfo, const std::string& paraName, uint32_t expectVal,
    uint32_t remotePara);

HcclResult ReportOpExchangeInfoCheckFailed(
    const CheckFailLogCtx& logCtx, const OpExchangeInfo& exchangeInfo, const std::string& paraName,
    const std::string& expectVal, const std::string& remotePara);

HcclResult GetOpTypeName(const OpExchangeInfo& exchangeInfo, std::string& opInfo);

} // namespace ops_hccl
#endif // INCONSISTENT_CHECK_H
