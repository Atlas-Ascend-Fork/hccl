/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_temp_all_to_all_v_mesh_1D.h"
#include "aiv_alg_template_base.h"
#include "hccl_aiv_utils.h"
#include "selector_test_common.h"
#include <numeric>

namespace ops_hccl {
namespace aiv_selector_test {
    namespace {

        // 覆盖alltoallv AIV模板的256/768双分支: 分支选择、ExtraArgs/ExtraArgsLarge四数组打包、
        // rankSize上界防御(HCCL_E_PARA)与URMA执行档的argsType标记
        class AllToAllVLargeRankTest : public AivTemplateSelectorTest {
        protected:
            void SetUp() override
            {
                AivTemplateSelectorTest::SetUp();
                param_.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
            }

            TemplateDataParams MakeParams(u32 rankSize)
            {
                TemplateDataParams tempAlgParams;
                tempAlgParams.sendCounts.resize(rankSize);
                tempAlgParams.sdispls.resize(rankSize);
                tempAlgParams.recvCounts.resize(rankSize);
                tempAlgParams.rdispls.resize(rankSize);
                for (u32 i = 0; i < rankSize; ++i) {
                    tempAlgParams.sendCounts[i] = 100u + i;
                    tempAlgParams.sdispls[i] = 1000u * rankSize + i;
                    tempAlgParams.recvCounts[i] = 200u + i;
                    tempAlgParams.rdispls[i] = 2000u * rankSize + i;
                }
                tempAlgParams.sliceSize = 4096;
                return tempAlgParams;
            }

            void RunKernel(u32 rankSize)
            {
                std::vector<std::vector<u32>> ranks(1, std::vector<u32>(rankSize));
                std::iota(ranks.front().begin(), ranks.front().end(), 0);
                topo_.userRankSize = rankSize;
                kernelLaunches = 0;
                launchedArgs = AivOpArgs{};
                AivTempAlltoAllVMesh1D algorithm(param_, 0, ranks);
                ASSERT_EQ(algorithm.KernelRun(param_, MakeParams(rankSize), TemplateResource{}), HCCL_SUCCESS);
                ASSERT_EQ(kernelLaunches, 1u);
                EXPECT_EQ(launchedArgs.rankSize, rankSize);
            }

            // 校验src数组前count项与期望特征值一致
            void ExpectArrayContent(const u64* src, u32 count, u64 base, u64 stride)
            {
                for (u32 i = 0; i < count; ++i) {
                    EXPECT_EQ(src[i], base + stride * i) << "idx=" << i;
                }
            }

            void ExpectArrayZero(const u64* src, u32 count)
            {
                for (u32 i = 0; i < count; ++i) {
                    EXPECT_EQ(src[i], 0u) << "idx=" << i;
                }
            }
        };

        TEST_F(AllToAllVLargeRankTest, Rank256FillsSmallExtraArgs)
        {
            RunKernel(256u);
            // 256恰在小分支上界: 走ExtraArgs且不触碰Large
            ExpectArrayContent(launchedArgs.extraArgs.sendCounts, 256u, 100u, 1u);
            ExpectArrayContent(launchedArgs.extraArgs.sendDispls, 256u, 256000u, 1u);
            ExpectArrayContent(launchedArgs.extraArgs.recvCounts, 256u, 200u, 1u);
            ExpectArrayContent(launchedArgs.extraArgs.recvDispls, 256u, 512000u, 1u);
            ExpectArrayZero(launchedArgs.extraArgsLarge.sendCounts, MAX_RANK_SIZE_V_LARGE);
            ExpectArrayZero(launchedArgs.extraArgsLarge.recvDispls, MAX_RANK_SIZE_V_LARGE);
            EXPECT_EQ(launchedArgs.argsType, KernelArgsType::ARGS_TYPE_SERVER);
        }

        TEST_F(AllToAllVLargeRankTest, Rank257FillsLargeExtraArgs)
        {
            RunKernel(257u);
            // 257越过小分支: 四数组进ExtraArgsLarge, ExtraArgs(256容量)不被触碰
            ExpectArrayContent(launchedArgs.extraArgsLarge.sendCounts, 257u, 100u, 1u);
            ExpectArrayContent(launchedArgs.extraArgsLarge.sendDispls, 257u, 257000u, 1u);
            ExpectArrayContent(launchedArgs.extraArgsLarge.recvCounts, 257u, 200u, 1u);
            ExpectArrayContent(launchedArgs.extraArgsLarge.recvDispls, 257u, 514000u, 1u);
            // 257项之外的尾部保持0, 证明写入被限制在count内
            EXPECT_EQ(launchedArgs.extraArgsLarge.sendCounts[257u], 0u);
            EXPECT_EQ(launchedArgs.extraArgsLarge.recvDispls[257u], 0u);
            ExpectArrayZero(launchedArgs.extraArgs.sendCounts, MAX_RANK_SIZE_V);
            ExpectArrayZero(launchedArgs.extraArgs.recvDispls, MAX_RANK_SIZE_V);
        }

        TEST_F(AllToAllVLargeRankTest, Rank768FillsLargeExtraArgs)
        {
            RunKernel(768u);
            // 768为大分支上限: 满数组打包
            ExpectArrayContent(launchedArgs.extraArgsLarge.sendCounts, 768u, 100u, 1u);
            ExpectArrayContent(launchedArgs.extraArgsLarge.sendDispls, 768u, 768000u, 1u);
            ExpectArrayContent(launchedArgs.extraArgsLarge.recvCounts, 768u, 200u, 1u);
            ExpectArrayContent(launchedArgs.extraArgsLarge.recvDispls, 768u, 1536000u, 1u);
            ExpectArrayZero(launchedArgs.extraArgs.sendCounts, MAX_RANK_SIZE_V);
        }

        TEST_F(AllToAllVLargeRankTest, Rank769RejectedBeforeLaunch)
        {
            const u32 rankSize = 769u;
            std::vector<std::vector<u32>> ranks(1, std::vector<u32>(rankSize));
            std::iota(ranks.front().begin(), ranks.front().end(), 0);
            topo_.userRankSize = rankSize;
            kernelLaunches = 0;
            launchedArgs = AivOpArgs{};
            AivTempAlltoAllVMesh1D algorithm(param_, 0, ranks);
            // 超过MAX_RANK_SIZE_V_LARGE直接拒绝, 不进入拷贝与launch
            EXPECT_EQ(algorithm.KernelRun(param_, MakeParams(rankSize), TemplateResource{}), HCCL_E_PARA);
            EXPECT_EQ(kernelLaunches, 0u);
        }

        TEST_F(AllToAllVLargeRankTest, UrmaConfigMarksArgsType)
        {
            param_.opExecuteConfig = OpExecuteConfig::AIV_URMA;
            RunKernel(64u);
            EXPECT_EQ(launchedArgs.argsType, KernelArgsType::ARGS_TYPE_URMA);
            // URMA档核数按对端并发组上限收敛
            EXPECT_EQ(launchedArgs.numBlocks, AIV_URMA_ALLTOALLV_GROUP_SIZE);
            ExpectArrayContent(launchedArgs.extraArgs.sendCounts, 64u, 100u, 1u);
        }

    } // namespace
} // namespace aiv_selector_test
} // namespace ops_hccl
