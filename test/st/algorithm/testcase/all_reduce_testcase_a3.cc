/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_param.h"
#include "gtest/gtest.h"
#include "sim_world.h"
#include "hccl.h"
#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"
#include "hccl_verifier.h"
#include "check_utils.h"
#include <thread>
#include "alg_env_config.h"

using namespace HcclSim;
using namespace ops_hccl;

class ST_ALL_REDUCE_TEST_A3 : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_BIRS_ENABLE");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void RunAllReduceBiarA3(
        const TopoMeta& topoMeta, const u64& count, const HcclDataType& dataType, const HcclReduceOp& reduceOp)
    {
        SimWorld::Global()->Init(topoMeta, HcclDevType::DEV_TYPE_910_93);

        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
        setenv("HCCL_BIRS_ENABLE", "TRUE", 1);

        // 算子执行参数设置
        auto rankSize = 0;
        for (u32 i = 0; i < topoMeta[0].size(); i++)
            rankSize += topoMeta[0][i].size(); // 参与集合通信的卡数(同topoMeta卡数一致)
        auto dataUnitSize = DATATYPE_SIZE_TABLE[dataType];
        // 多线程运行ReduceScatter算子
        std::vector<std::thread> threads;
        for (auto rankId = 0; rankId < rankSize; ++rankId) {
            threads.emplace_back([=]() {
                // 1.SetDevice
                aclrtSetDevice(rankId);

                // 2.创建流
                aclrtStream stream = nullptr;
                aclrtCreateStream(&stream);

                // 3.初始化通信域
                HcclComm comm = nullptr;
                CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

                void* sendBuf = nullptr;
                void* recvBuf = nullptr;
                u64 sendBufSize = count * dataUnitSize; // 数据量转化为字节数
                u64 recvBufSize = count * dataUnitSize;
                // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
                if (aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK))
                    != ACL_SUCCESS) {
                    return HcclResult::HCCL_E_MEMORY;
                }
                if (aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK))
                    != ACL_SUCCESS) {
                    return HcclResult::HCCL_E_MEMORY;
                }

                // 4.算子下发
                CHK_RET(HcclAllReduce(sendBuf, recvBuf, count, dataType, reduceOp, comm, stream));

                // 5.销毁通信域
                CHK_RET(HcclCommDestroy(comm));
                return HCCL_SUCCESS;
            });
        }

        // 等待多线程执行完成
        for (auto& thread : threads) {
            thread.join();
        }

        // 结果成图校验
        auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
        HcclResult res = CheckAllReduce(taskQueues, rankSize, dataType, count, reduceOp);
        EXPECT_TRUE(res == HCCL_SUCCESS);

        // 资源清理
        SimWorld::Global()->Deinit();
    }
};

TEST_F(ST_ALL_REDUCE_TEST_A3, st_all_reduce_114_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3}}};                  // 三维数组指定超节点-Server-Device信息
    auto count = 400;                                   // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceBiarA3(topoMeta, count, dataType, reduceOp);
}

TEST_F(ST_ALL_REDUCE_TEST_A3, st_all_reduce_114_rem_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3}}};                  // 三维数组指定超节点-Server-Device信息
    auto count = 402;                                   // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceBiarA3(topoMeta, count, dataType, reduceOp);
}

TEST_F(ST_ALL_REDUCE_TEST_A3, st_all_reduce_116_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5}}};
    auto count = 600;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceBiarA3(topoMeta, count, dataType, reduceOp);
}

TEST_F(ST_ALL_REDUCE_TEST_A3, st_all_reduce_116_rem_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5}}};
    auto count = 602;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceBiarA3(topoMeta, count, dataType, reduceOp);
}

TEST_F(ST_ALL_REDUCE_TEST_A3, st_all_reduce_118_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};
    auto count = 800;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceBiarA3(topoMeta, count, dataType, reduceOp);
}

TEST_F(ST_ALL_REDUCE_TEST_A3, st_all_reduce_118_rem_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};
    auto count = 803;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceBiarA3(topoMeta, count, dataType, reduceOp);
}

TEST_F(ST_ALL_REDUCE_TEST_A3, st_all_reduce_1110_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}}; // 三维数组指定超节点-Server-Device信息
    auto count = 1000;                                   // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceBiarA3(topoMeta, count, dataType, reduceOp);
}

TEST_F(ST_ALL_REDUCE_TEST_A3, st_all_reduce_1112_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}}}; // 三维数组指定超节点-Server-Device信息
    auto count = 1200;                                           // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;          // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceBiarA3(topoMeta, count, dataType, reduceOp);
}

TEST_F(ST_ALL_REDUCE_TEST_A3, st_all_reduce_1114_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}}}; // 三维数组指定超节点-Server-Device信息
    auto count = 1400;                                                   // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;                  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceBiarA3(topoMeta, count, dataType, reduceOp);
}

TEST_F(ST_ALL_REDUCE_TEST_A3, st_all_reduce_1116_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};
    auto count = 1600;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceBiarA3(topoMeta, count, dataType, reduceOp);
}

TEST_F(ST_ALL_REDUCE_TEST_A3, st_all_reduce_1116_rem_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};
    auto count = 1603;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceBiarA3(topoMeta, count, dataType, reduceOp);
}

TEST_F(ST_ALL_REDUCE_TEST_A3, st_all_reduce_18_min_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};
    auto count = 800;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunAllReduceBiarA3(topoMeta, count, dataType, reduceOp);
}

TEST_F(ST_ALL_REDUCE_TEST_A3, st_all_reduce_116_rem_max_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};
    auto count = 1603;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunAllReduceBiarA3(topoMeta, count, dataType, reduceOp);
}

TEST_F(ST_ALL_REDUCE_TEST_A3, st_all_reduce_1416_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    std::vector<u32> args{1, 4, 16};
    for (int i = 0; i < args[0]; i++) {
        SuperPodMeta superPodMeta;
        for (int j = 0; j < args[1]; j++) {
            ServerMeta serverMate;
            for (int k = 0; k < args[2]; k++) {
                serverMate.push_back((unsigned int)k);
            }
            superPodMeta.push_back(serverMate);
        }
        topoMeta.push_back(superPodMeta);
    }
    auto recvCount = 6400;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceBiarA3(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_ALL_REDUCE_TEST_A3, st_all_reduce_1416_max_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 16);
    auto count = 640;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunAllReduceBiarA3(topoMeta, count, dataType, reduceOp);
}
