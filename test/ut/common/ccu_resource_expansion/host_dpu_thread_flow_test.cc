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

#include <map>
#include <string>

#include "dlhcomm_function.h"
#include "hcomm_stub.h"
#include "op_common.h"
#include "securec.h"

namespace ops_hccl {
// 仅注入通信域配置查询，不加载真实HCOMM动态库。
DlHcommFunction& DlHcommFunction::GetInstance()
{
    static DlHcommFunction instance;
    return instance;
}

DlHcommFunction::~DlHcommFunction() = default;
} // namespace ops_hccl

using namespace ops_hccl;
using namespace ut_stub;

namespace {
constexpr const char* RHD_CONFIG = "barrier:sequence{level0=mesh,level1=rhd}";
constexpr const char* NHR_CONFIG = "barrier:sequence{level0=mesh,level1=nhr}";

class HostDpuThreadFlowTest : public testing::Test {
protected:
    void SetUp() override
    {
        g_resources = MockResources{};
        g_resources.hcommVersion = CANN_VERSION(9, 3, 0);
        DlHcommFunction::GetInstance().dlHcclConfigGetInfo
            = [this](HcclComm comm, HcclConfigType type, uint32_t length, void* buffer) {
                  ++queryCalls_;
                  EXPECT_EQ(type, static_cast<HcclConfigType>(HCCL_CONFIG_TYPE_HCCL_ALGO));
                  if (queryResult_ != HCCL_SUCCESS) {
                      return queryResult_;
                  }
                  const std::string& config = commConfigs_[comm];
                  return strcpy_s(static_cast<char*>(buffer), length, config.c_str()) == EOK ? HCCL_SUCCESS :
                                                                                               HCCL_E_INTERNAL;
              };
    }

    void TearDown() override
    {
        DlHcommFunction::GetInstance().dlHcclConfigGetInfo = nullptr;
        g_resources = MockResources{};
    }

    HcclComm CommA() { return static_cast<HcclComm>(&commA_); }
    HcclComm CommB() { return static_cast<HcclComm>(&commB_); }
    HcclComm CommC() { return static_cast<HcclComm>(&commC_); }

    int commA_ = 0;
    int commB_ = 0;
    int commC_ = 0;
    std::map<HcclComm, std::string> commConfigs_;
    HcclResult queryResult_ = HCCL_SUCCESS;
    uint32_t queryCalls_ = 0;
};

TEST_F(HostDpuThreadFlowTest, PreservesVersionBoundary)
{
    for (int version : {CANN_VERSION(9, 3, 0, 2), CANN_VERSION(9, 3, 0) - 1}) {
        g_resources.hcommVersion = version;
        EXPECT_FALSE(IsHostDpuThreadNewFlowEnabled(CommA()));
    }
    EXPECT_EQ(queryCalls_, 0U);
    g_resources.hcommVersion = CANN_VERSION(9, 3, 0);
    EXPECT_TRUE(IsHostDpuThreadNewFlowEnabled(CommA()));
    EXPECT_EQ(queryCalls_, 1U);
}

TEST_F(HostDpuThreadFlowTest, EffectiveConfigurationControlsNewFlow)
{
    const struct {
        const char* env;
        const char* config;
        bool newFlow;
    } cases[] = {
        {"EmptyString", "", true},
        {"", "", true},
        {NHR_CONFIG, NHR_CONFIG, true},
        {RHD_CONFIG, "", false},
        {"EmptyString", RHD_CONFIG, false},
        {RHD_CONFIG, NHR_CONFIG, true},
        {NHR_CONFIG, RHD_CONFIG, false},
        {"barrier:sequence{level0=mesh,level1=RhD}", "", false},
        {"barrier:sequence{level0=mesh,level1=RhD}", NHR_CONFIG, true},
        {NHR_CONFIG, "barrier:sequence{level0=mesh,level1=RHD}", false},
    };
    for (const auto& testCase : cases) {
        SCOPED_TRACE(std::string("env=") + testCase.env + ", config=" + testCase.config);
        g_resources.algoEnv = testCase.env;
        commConfigs_[CommA()] = testCase.config;
        EXPECT_EQ(IsHostDpuThreadNewFlowEnabled(CommA()), testCase.newFlow);
    }
}

TEST_F(HostDpuThreadFlowTest, SupportsMixedFlowsAcrossCommunicationDomains)
{
    commConfigs_[CommA()] = RHD_CONFIG;
    commConfigs_[CommB()] = NHR_CONFIG;
    EXPECT_FALSE(IsHostDpuThreadNewFlowEnabled(CommA()));
    EXPECT_TRUE(IsHostDpuThreadNewFlowEnabled(CommB()));
    EXPECT_TRUE(IsHostDpuThreadNewFlowEnabled(CommC()));
    EXPECT_FALSE(IsHostDpuThreadNewFlowEnabled(CommA()));
    g_resources.algoEnv = RHD_CONFIG;
    EXPECT_FALSE(IsHostDpuThreadNewFlowEnabled(CommA()));
    EXPECT_TRUE(IsHostDpuThreadNewFlowEnabled(CommB()));
    EXPECT_FALSE(IsHostDpuThreadNewFlowEnabled(CommC()));
    g_resources.algoEnv = "EmptyString";
    EXPECT_TRUE(IsHostDpuThreadNewFlowEnabled(CommB()));
    EXPECT_TRUE(IsHostDpuThreadNewFlowEnabled(CommC()));
}

TEST_F(HostDpuThreadFlowTest, MissingOrFailedCommQueryKeepsEnvironmentDecision)
{
    queryResult_ = HCCL_E_INTERNAL;
    EXPECT_TRUE(IsHostDpuThreadNewFlowEnabled(CommA()));
    g_resources.algoEnv = RHD_CONFIG;
    EXPECT_FALSE(IsHostDpuThreadNewFlowEnabled(CommA()));
    DlHcommFunction::GetInstance().dlHcclConfigGetInfo = nullptr;
    EXPECT_FALSE(IsHostDpuThreadNewFlowEnabled(CommA()));
    g_resources.algoEnv = "EmptyString";
    EXPECT_TRUE(IsHostDpuThreadNewFlowEnabled(CommA()));
}
} // namespace
