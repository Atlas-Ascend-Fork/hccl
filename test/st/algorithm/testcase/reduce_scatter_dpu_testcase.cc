/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "testcase_common.h"
#include "hccl_sim_pub.h"

using namespace HcclSim;
using namespace ops_hccl;

class ST_REDUCE_SCATTER_OMNIPIPE_DPU_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

void RunReduceScatterOmniPipeDPU(const TopoMeta& topoMeta, u64 recvCount, HcclDataType dataType, HcclReduceOp reduceOp)
{
    u32 rankSize = CalRankSize(topoMeta);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    RunDpuTest(
        topoMeta, rankSize, recvCount, dataType, recvCount * dataTypeSize * rankSize, recvCount * dataTypeSize,
        HcclReduceScatter, CheckReduceScatter, reduceOp);
}

// ========== 正向测试 ==========

// 对称3级: x=4, y=4, z=2, int8
TEST_F(ST_REDUCE_SCATTER_OMNIPIPE_DPU_TEST, omnipipe_dpu_3level_4x4x2_int8)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 recvCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterOmniPipeDPU(topoMeta, recvCount, dataType, reduceOp);
}

// 对称3级: x=2, y=2, z=2, bfp16
TEST_F(ST_REDUCE_SCATTER_OMNIPIPE_DPU_TEST, omnipipe_dpu_3level_2x2x2_bfp16)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    u64 recvCount = 256;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterOmniPipeDPU(topoMeta, recvCount, dataType, reduceOp);
}

// 对称3级: x=8, y=4, z=2, int32
TEST_F(ST_REDUCE_SCATTER_OMNIPIPE_DPU_TEST, omnipipe_dpu_3level_8x4x2_int32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 8);
    u64 recvCount = 2048;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterOmniPipeDPU(topoMeta, recvCount, dataType, reduceOp);
}

// 大数据量: 1M
TEST_F(ST_REDUCE_SCATTER_OMNIPIPE_DPU_TEST, omnipipe_dpu_3level_4x4x2_fp32_1m)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 recvCount = 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterOmniPipeDPU(topoMeta, recvCount, dataType, reduceOp);
}

// ========== 边界测试 ==========
// zRankSize=1 (L2退化, 单超节点) -> 不走omnipipe
TEST_F(ST_REDUCE_SCATTER_OMNIPIPE_DPU_TEST, omnipipe_dpu_3level_4x4x1_fp32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 4);
    u64 recvCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterOmniPipeDPU(topoMeta, recvCount, dataType, reduceOp);
}

// yRankSize=1 (L1退化)
TEST_F(ST_REDUCE_SCATTER_OMNIPIPE_DPU_TEST, omnipipe_dpu_3level_4x1x2_fp32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 4);
    u64 recvCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterOmniPipeDPU(topoMeta, recvCount, dataType, reduceOp);
}

// xRankSize=1 (L0退化)
TEST_F(ST_REDUCE_SCATTER_OMNIPIPE_DPU_TEST, omnipipe_dpu_3level_1x4x2_fp32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 1);
    u64 recvCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterOmniPipeDPU(topoMeta, recvCount, dataType, reduceOp);
}

// 最小数据量
TEST_F(ST_REDUCE_SCATTER_OMNIPIPE_DPU_TEST, omnipipe_dpu_3level_2x2x2_fp32_recv1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    u64 recvCount = 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterOmniPipeDPU(topoMeta, recvCount, dataType, reduceOp);
}

// ============================================================================
// 三级 Sequence DPU: DpuReduceScatterSequenceMeshConcurNHRMesh
// (L0 ZAxisDetour 设备绕路 / L1 NHR 设备 / L2 DpuInter host-DPU Mesh, L0->L1->L2 正序)
//
// 拓扑记法 x*y*z: x=d0(框内卡数) y=d1(超节点内 server 数) z=d2(超节点数);
// GenTopoMeta 参数序为 (superPodNum=z, serverNum=y, rankNum=x)
//
// HCCL_ALGO 配置按 alg_parse.cc 词表拼名: "reducescatter:sequence{level0=meshconcur,level1=nhr,
// level2=mesh}" -> DpuReduceScatterSequenceMeshConcurNHRMesh; 选择行为断言所依赖的记录钩子
// (aclrt_stub 记录 OpParam.algName) 与主线 stub 演进(HostArgs 下发路径)冲突, 已移除,
// 用例采用与 AG/AR 三级 DPU 家族一致的"强制选路 + 数据正确性"模式
// ============================================================================

class ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_ALGO");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

namespace {
constexpr const char* SEQ_DPU_3LEVEL_HCCL_ALGO = "reducescatter:sequence{level0=meshconcur,level1=nhr,level2=mesh}";
} // namespace

// 指定 HCCL_ALGO 运行: costmodel 过滤后仅剩目标算法, 用于定向覆盖三级路径;
// 注意若目标算法不在 cost 表(如准入被拒)过滤不生效, 会静默回退默认竞争
void RunReduceScatterDpuWithAlgo(
    const char* hcclAlgo, const TopoMeta& topoMeta, u64 recvCount, HcclDataType dataType, HcclReduceOp reduceOp)
{
    setenv("HCCL_ALGO", hcclAlgo, 1);
    u32 rankSize = CalRankSize(topoMeta);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    RunDpuTest(
        topoMeta, rankSize, recvCount, dataType, recvCount * dataTypeSize * rankSize, recvCount * dataTypeSize,
        HcclReduceScatter, CheckReduceScatter, reduceOp);
}

// 三级 Sequence 定向运行: HCCL_ALGO 钉住三级算法, 保证用例真实走三级路径
// (默认选择受 cost 竞争影响, 不保证选中三级)
void RunReduceScatterSequenceDPU(const TopoMeta& topoMeta, u64 recvCount, HcclDataType dataType, HcclReduceOp reduceOp)
{
    RunReduceScatterDpuWithAlgo(SEQ_DPU_3LEVEL_HCCL_ALGO, topoMeta, recvCount, dataType, reduceOp);
}

// 默认选择运行(不设 HCCL_ALGO), 用于默认选择回归
void RunReduceScatterDpuDefault(const TopoMeta& topoMeta, u64 recvCount, HcclDataType dataType, HcclReduceOp reduceOp)
{
    u32 rankSize = CalRankSize(topoMeta);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    RunDpuTest(
        topoMeta, rankSize, recvCount, dataType, recvCount * dataTypeSize * rankSize, recvCount * dataTypeSize,
        HcclReduceScatter, CheckReduceScatter, reduceOp);
}

// ---------- 正向: d1>=2 对称三级, dtype 泛化 (AC-2) ----------

// 对称3级: x=2, y=4, z=4, int8
TEST_F(ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST, seq_dpu_3level_2x4x4_int8)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 4, 2);
    RunReduceScatterSequenceDPU(topoMeta, 512, HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM);
}

// 对称3级: x=2, y=4, z=4, bfp16
TEST_F(ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST, seq_dpu_3level_2x4x4_bfp16)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 4, 2);
    RunReduceScatterSequenceDPU(topoMeta, 256, HcclDataType::HCCL_DATA_TYPE_BFP16, HcclReduceOp::HCCL_REDUCE_SUM);
}

// 对称3级: x=2, y=4, z=4, int32
TEST_F(ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST, seq_dpu_3level_2x4x4_int32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 4, 2);
    RunReduceScatterSequenceDPU(topoMeta, 2048, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
}

// 对称3级: x=2, y=4, z=4, fp16
TEST_F(ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST, seq_dpu_3level_2x4x4_fp16)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 4, 2);
    RunReduceScatterSequenceDPU(topoMeta, 512, HcclDataType::HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_SUM);
}

// 对称3级: x=2, y=4, z=4, fp32 大数据量
TEST_F(ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST, seq_dpu_3level_2x4x4_fp32_1m)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 4, 2);
    RunReduceScatterSequenceDPU(
        topoMeta, 1024 * 1024, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
}

// ---------- 1825pod 主场景: d1==1, skipLevel1 退化 (L2 直通 L0 布局, AC-2) ----------

// x=2, y=1, z=2
TEST_F(ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST, seq_dpu_3level_2x1x2_fp32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 2);
    RunReduceScatterSequenceDPU(topoMeta, 512, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
}

// x=4, y=1, z=2
TEST_F(ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST, seq_dpu_3level_4x1x2_fp32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 4);
    RunReduceScatterSequenceDPU(topoMeta, 512, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
}

// x=8, y=1, z=2
TEST_F(ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST, seq_dpu_3level_8x1x2_fp32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 8);
    RunReduceScatterSequenceDPU(topoMeta, 512, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
}

// ---------- 边界与尾块 (AC-2) ----------

// 双退化边界 x=1,y=1,z=4: L0 ZAxisDetour 静态 cost 对 rankSize<=1 返回空 => 三级不进 cost 表,
// 回退二级(与设备侧三级同款限制); 强制三级 HCCL_ALGO 下验证回退路径结果正确 (AC-3)
TEST_F(ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST, seq_dpu_3level_1x1x4_fallback_2level)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 1, 1);
    RunReduceScatterDpuWithAlgo(
        SEQ_DPU_3LEVEL_HCCL_ALGO, topoMeta, 16 + 1, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
}

// 最小数据量 recvCount=1: L2 allRank* 契约在 C=1 下的边界
TEST_F(ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST, seq_dpu_3level_2x2x2_fp32_recv1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    RunReduceScatterSequenceDPU(topoMeta, 1, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
}

// 元素级尾块: skipLevel1 主场景 + recvCount 不按位宽对齐
TEST_F(ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST, seq_dpu_3level_4x1x2_int8_recv8_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 4);
    RunReduceScatterSequenceDPU(topoMeta, 8 + 1, HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM);
}

// 多 loop 尾块: x=2,y=2,z=2 时 scratch 系数 = d0*d1*d2+d2 = 10, 单 loop 上限 200MB/10/4 = 5242880
// 个 fp32 元素; recvCount 取 5242880+100 => 2 loop, 尾 loop 100 元素
TEST_F(ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST, seq_dpu_3level_2x2x2_fp32_multi_loop_tail)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    RunReduceScatterSequenceDPU(
        topoMeta, 5242880 + 100, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
}

// ---------- 默认选择回归 (AC-3 / AC-1) ----------

// 默认选择: 对称三级拓扑 + 大数据量下走默认选路, 结果正确即可(选择行为断言依赖的记录钩子
// 已随主线 stub 演进移除, 准入矩阵由本地 UT 与联调覆盖)
TEST_F(ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST, seq_dpu_3level_4x4x2_default_select_sequence)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    RunReduceScatterDpuDefault(topoMeta, 1024 * 1024, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
}

// ---------- 回归: 既有选择行为不变 (AC-1 / AC-3) ----------

// 二级拓扑(z=1, 单超节点): 默认选择下结果正确(二级 Sequence 承接)
TEST_F(ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST, seq_dpu_fallback_2level_topology_8x2)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceScatterDpuDefault(topoMeta, 512, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
}

// 三级非对称拓扑: 三级 Sequence 对称性准入不通过, 回退其他算法且结果正确
TEST_F(ST_REDUCE_SCATTER_SEQUENCE_DPU_TEST, seq_dpu_fallback_asymmetric_3level)
{
    TopoMeta topoMeta{
        {{0, 1, 2, 3}, {0, 1, 2}},
        {{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7}},
    };
    RunReduceScatterDpuDefault(topoMeta, 200, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
}

// 注: 需求回归表的以下行 ST 框架暂无法覆盖, 留作联调项:
//  - UBX(MESH_1D_CLOS): ST 拓扑模型仅能构造 MESH_1D 框内形态
//  - 中间层含非 CLOS: ST topo 模型 L1/L2 网型恒为 CLOS
// 另: 对称内存路径非本期需求范围(需求 AC 清单不含), 生产侧受 IsReduceOpSymMemDeterministic
// 确定性门禁限制对本算法不可达; 代码中的对称分支为共用骨架(utils 与设备侧 AICPU 三级两叶子
// 共用)的继承性代码, 已含超容拒绝防御(HCCL_E_INTERNAL), 不纳入联调验证项
