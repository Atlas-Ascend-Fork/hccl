/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"
#include "gtest/gtest.h"
#include "sim_world.h"
#include "hccl.h"
#include "hccl_verifier.h"
#include "check_utils.h"
#include <thread>
#include "alg_env_config.h"

using namespace HcclSim;
using namespace ops_hccl;

constexpr uint32_t DATATYPE_SIZE_TABLE_ALL_GATHER_ST[HCCL_DATA_TYPE_RESERVED]
    = {sizeof(int8_t),
       sizeof(int16_t),
       sizeof(int32_t),
       2,
       sizeof(float),
       sizeof(int64_t),
       sizeof(uint64_t),
       sizeof(uint8_t),
       sizeof(uint16_t),
       sizeof(uint32_t),
       8,
       2,
       16,
       2,
       1,
       1,
       1,
       1};

class ST_ALL_GATHER_AICPU_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_OP_EXPANSION_MODE");
    }

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};
u32 AnalyseRankSize(const TopoMeta& topoInfo)
{
    u32 rankSize = 0;
    for (const auto& superPod : topoInfo) {
        for (const auto& podIdx : superPod) {
            rankSize += podIdx.size();
        }
    }
    return rankSize;
}

void RunAllGatherAicpuA5(const TopoMeta& topoInfo, const u64& sendCount, const HcclDataType& dataType)
{
    // 仿真模型初始化
    SimWorld::Global()->Init(topoInfo, HcclDevType::DEV_TYPE_950);

    // 设置展开模式为HOST_TS
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_ALL_GATHER_ST[dataType];
    auto rankSize = AnalyseRankSize(topoInfo);
    // 算子执行参数设置,多线程运行SCATTER算子
    std::vector<std::thread> threads;
    for (auto rankIdx = 0; rankIdx < rankSize; ++rankIdx) {
        threads.emplace_back([=]() {
            // 1.SetDevice
            aclrtSetDevice(rankIdx);
            // 2.创建流
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);
            // 3.初始化通信域
            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankIdx, &comm));

            void* sendBuf = nullptr;
            void* recvBuf = nullptr;
            u64 sendBufSize = sendCount * dataTypeSize; // 数据量转化为字节数
            u64 recvBufSize = sendCount * dataTypeSize * rankSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            HCCL_INFO("[ST_ALL_GATHER_AICPU_TEST]Run HcclAllGather");

            // 4.算子下发
            CHK_RET(HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream));

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
    HcclResult res = CheckAllGather(taskQueues, rankSize, dataType, sendCount);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit();
}

// 失败路径 runner：下发算子后断言数据 Checker 失败（算子执行失败，选路期不回退）。
// 仿真执行模型下 AICPU kernel 失败相对 API 提交是异步的：HcclAllGather 提交返回 SUCCESS，
// kernel 在 notify-wait 阶段失败（错误日志可查：幂守卫 HCCL_ERROR + call trace: hcclRet -> 1，
// 即 HCCL_E_PARA；API 返回码不作为失败判据，仅以 HCCL_INFO 记录供 CI 日志比对）。
// 算子执行失败的在用断言信号为数据 Checker：正常路径（含选路期回退成功的场景）Checker 必须
// PASS；本 runner 断言 Checker 失败——若 HDHD 被回退默认算法完成，Checker 将通过，用例失败。
void RunAllGatherAicpuA5ExpectOpFail(const TopoMeta& topoInfo, const u64& sendCount, const HcclDataType& dataType)
{
    // 仿真模型初始化
    SimWorld::Global()->Init(topoInfo, HcclDevType::DEV_TYPE_950);

    // 设置展开模式为HOST_TS
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_ALL_GATHER_ST[dataType];
    auto rankSize = AnalyseRankSize(topoInfo);
    // 各 rank 算子 API 返回码（每线程仅写自身下标、join 后仅记录日志，无竞争）
    std::vector<HcclResult> opRets(rankSize, HCCL_SUCCESS);
    // 算子执行参数设置,多线程运行SCATTER算子
    std::vector<std::thread> threads;
    for (auto rankIdx = 0; rankIdx < rankSize; ++rankIdx) {
        threads.emplace_back([=, &opRets]() {
            // 1.SetDevice
            aclrtSetDevice(rankIdx);
            // 2.创建流
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);
            // 3.初始化通信域
            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankIdx, &comm));

            void* sendBuf = nullptr;
            void* recvBuf = nullptr;
            u64 sendBufSize = sendCount * dataTypeSize; // 数据量转化为字节数
            u64 recvBufSize = sendCount * dataTypeSize * rankSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            HCCL_INFO("[ST_ALL_GATHER_AICPU_TEST]Run HcclAllGather expect op fail");

            // 4.算子下发（预期执行失败：捕获 API 返回码仅作日志记录）
            opRets[rankIdx] = HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream);

            // 5.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    // 等待多线程执行完成
    for (auto& thread : threads) {
        thread.join();
    }
    for (auto rankIdx = 0; rankIdx < rankSize; ++rankIdx) {
        HCCL_INFO(
            "[ST_ALL_GATHER_AICPU_TEST]rank[%u] HcclAllGather api ret[%d], expect op execute fail.", rankIdx,
            opRets[rankIdx]);
    }

    // 结果成图校验：算子执行失败（无数据搬运任务）时 Checker 必须失败
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckAllGather(taskQueues, rankSize, dataType, sendCount);
    EXPECT_TRUE(res != HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_2rank_int64_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT64; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_3rank_int64_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100; // 1GB单卡数据量，scratch单rank分区为100M（200M/2）, 有24M尾块
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT64; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_4rank_int32_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_6rank_int16_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 5}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_8rank_int8_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_2rank_fp64_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP64; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_3rank_fp32_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}, {0}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_4rank_fp16_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_8rank_bfp16_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_6rank_fp8e5m2_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}, {0}, {0}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                                 // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E5M2; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_6rank_fp8e8m0_big_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}, {0}, {0}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 1024 * 1024 * 1024; // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E8M0; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_2x2rank_small_data_test)
{
    // 1GB较大数据量（多loop），同时包含尾块，数据宽度8B
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100; // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT64; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_2x3rank_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100; // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT32; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_2x3rank_big_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 300 * 1024 * 1024; // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT16; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_3x3rank_big_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 300 * 1024 * 1024; // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_meshnhr_2x2x4rank_int8_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    auto sendCount = 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_meshnhr_2x1x8rank_int8_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 8);
    auto sendCount = 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

// asymmetric topology
TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_7server_asymmetric_fp8e8m0_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{
        {{0, 1, 2, 3, 4, 5, 6, 7},
         {0, 1, 2, 3, 4, 5, 6},
         {0, 1, 2, 3, 4, 5},
         {0, 1, 2, 3, 4},
         {0, 1, 2, 3},
         {0, 1, 2},
         {1, 2}}};

    // 算子执行参数设置
    auto sendCount = 100; // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E8M0; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

// ========== HD 算法强制选路（参照 all_gather_dpu_testcase.cc 的 setenv/unsetenv 模式）==========
// HD 仅经 HCCL_ALGO 显式启用（opCustomCheck 恒 false 不参与默认软检查竞争）；
// 非幂两级组合（如 2x3，level0=3）选路期不回退：executor CalcCostCoeff 逐段拼接系数，
// 幂段（level1）系数非空即进入 cost 表、HDHD 可被显式命中；执行期由模板 KernelRun 幂守卫
// （按算法作用域 subCommRanks_[0] 校验，守卫报 HCCL_E_PARA）拦截，不回退默认算法，
// 算子执行失败（数据 Checker 失败断言 + 守卫错误日志佐证）。
class ST_ALL_GATHER_AICPU_HD_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("HCCL_ALGO");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
    }

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// sole HD 强制选路：单 server 4 卡（2 的幂 rankSize），Checker 正常路径
TEST_F(ST_ALL_GATHER_AICPU_HD_TEST, st_all_gather_a5_aicpu_hd_sole_mesh_1d_4rank_int8_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3}}}; // 三维数组指定超节点-Server-Device信息
    setenv("HCCL_ALGO", "allgather:sole{hd}", 1);

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

// parallel HDHD 强制选路：2x2 两级规模均为 2 的幂，Checker 正常路径
TEST_F(ST_ALL_GATHER_AICPU_HD_TEST, st_all_gather_a5_aicpu_hd_parallel_hdhd_2x2rank_int8_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    setenv("HCCL_ALGO", "allgather:parallel{hd,hd}", 1);

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

// 非幂两级组合：2x3（level0=3 非 2 的幂）强制 HDHD——4 段齐整性校验已回退（用户评估公共模板
// executor 改动对其他算法风险不可接受），executor CalcCostCoeff 恢复逐段拼接：p0/p2 段（level0=3
// 非幂）系数为空、p1/p3 段（level1=2 幂）系数非空，部分系数进入 cost 表，HDHD 可被显式命中，
// 选路期不回退；两级非幂组合在执行期由模板 KernelRun 幂守卫（按算法作用域
// subCommRanks_[0].size()=3 校验）拦截，算子执行失败（Checker 失败断言）。
// 错误日志断言点（CI 运行日志佐证，ST 框架无进程内日志捕获）：守卫 HCCL_ERROR
// "[InsTempAllGatherHDOp] Rank [x], rankSize[3] is not a power of two or is less than 2,
// not supported." 与 "[KernelRun]call trace: hcclRet -> 1"（HCCL_E_PARA）、
// "orchestrate failed for alg:AicpuAllGatherParallelHDHD"（显式命中直证，无选路期回退）
TEST_F(ST_ALL_GATHER_AICPU_HD_TEST, st_all_gather_a5_aicpu_hd_parallel_hdhd_2x3rank_nonpower_expect_fail_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}}; // 三维数组指定超节点-Server-Device信息
    setenv("HCCL_ALGO", "allgather:parallel{hd,hd}", 1);

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5ExpectOpFail(topoMeta, sendCount, dataType);
}
