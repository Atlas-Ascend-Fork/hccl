/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <gtest/gtest.h>
#include "all_reduce_biar_selector.h"

using ops_hccl::TopoInfo;
using ops_hccl_experimental::BiarSelectResult;
using ops_hccl_experimental::BiarSelectResultToCode;
using ops_hccl_experimental::DecideAllReduceBiarAlg;

namespace {
TopoInfo MakeTopo(u32 userRankSize, u32 serverNum, HcclDevType deviceType)
{
    TopoInfo topo{};
    topo.userRankSize = userRankSize;
    topo.serverNum = serverNum;
    topo.deviceType = deviceType;
    return topo;
}

const std::string BIAR_ALG_NAME = "AllReduceBIARExecutor";
} // namespace

TEST(AllReduceBiarSelector, ServerNumZeroIsRejectedInsteadOfDivByZero)
{
    std::string algName;
    TopoInfo topo = MakeTopo(8, 0, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideAllReduceBiarAlg(topo, algName), BiarSelectResult::bRejectServerNumZero);
    EXPECT_TRUE(algName.empty());
}

TEST(AllReduceBiarSelector, DualCardSingleServerIsRejected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(2, 1, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideAllReduceBiarAlg(topo, algName), BiarSelectResult::bRejectRanksPerServerLT4);
    EXPECT_TRUE(algName.empty());
}

TEST(AllReduceBiarSelector, ThreeRanksPerServerIsRejected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(6, 2, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideAllReduceBiarAlg(topo, algName), BiarSelectResult::bRejectRanksPerServerLT4);
    EXPECT_TRUE(algName.empty());
}

TEST(AllReduceBiarSelector, RankSizeOneIsRejected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(1, 1, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideAllReduceBiarAlg(topo, algName), BiarSelectResult::bRejectRankSizeOne);
    EXPECT_TRUE(algName.empty());
}

TEST(AllReduceBiarSelector, FourRanksPerServerIsSelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(8, 2, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideAllReduceBiarAlg(topo, algName), BiarSelectResult::bSelected);
    EXPECT_EQ(algName, BIAR_ALG_NAME);
}

TEST(AllReduceBiarSelector, ExactlyFourRanksPerServerBoundarySelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(4, 1, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideAllReduceBiarAlg(topo, algName), BiarSelectResult::bSelected);
    EXPECT_EQ(algName, BIAR_ALG_NAME);
}

TEST(AllReduceBiarSelector, MoreThanFourRanksSingleServerIsSelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(16, 1, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideAllReduceBiarAlg(topo, algName), BiarSelectResult::bSelected);
    EXPECT_EQ(algName, BIAR_ALG_NAME);
}

TEST(AllReduceBiarSelector, MoreThanFourRanksPerServerIsSelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(16, 4, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideAllReduceBiarAlg(topo, algName), BiarSelectResult::bSelected);
    EXPECT_EQ(algName, BIAR_ALG_NAME);
}

TEST(AllReduceBiarSelector, NonA3DeviceIsNotSelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(8, 2, HcclDevType::DEV_TYPE_910B);
    EXPECT_EQ(DecideAllReduceBiarAlg(topo, algName), BiarSelectResult::bNotSelected);
    EXPECT_TRUE(algName.empty());
}

TEST(AllReduceBiarSelector, OddRankSizeIsNotSelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(9, 1, HcclDevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideAllReduceBiarAlg(topo, algName), BiarSelectResult::bNotSelected);
    EXPECT_TRUE(algName.empty());
}

TEST(AllReduceBiarSelector, SelectResultToCodeMapsRejectsToErrorCodes)
{
    EXPECT_EQ(BiarSelectResultToCode(BiarSelectResult::bSelected), HCCL_SUCCESS);
    EXPECT_EQ(BiarSelectResultToCode(BiarSelectResult::bNotSelected), HCCL_SUCCESS);
    EXPECT_EQ(BiarSelectResultToCode(BiarSelectResult::bRejectRankSizeOne), HCCL_E_INTERNAL);
    EXPECT_EQ(BiarSelectResultToCode(BiarSelectResult::bRejectServerNumZero), HCCL_E_PARA);
    EXPECT_EQ(BiarSelectResultToCode(BiarSelectResult::bRejectRanksPerServerLT4), HCCL_E_PARA);
}
