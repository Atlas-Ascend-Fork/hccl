/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTO_SELECTOR_BASE
#define AUTO_SELECTOR_BASE

#include <string>
#include <unordered_map>
#include "alg_param.h"
#include "log.h"
#include "alg_env_config.h"
#include "alg_attrs.h"

namespace ops_hccl {

constexpr uint64_t SMALL_COUNT_512KB = 512 * 1024;          // Byte, UB协议一次传输的最大size
constexpr uint64_t LARGE_COUNT_1024KB = 1024 * 1024;        // Byte, 可掩盖多mission尾块开销
constexpr u64 AIV_MAX_PER_RANK_DATA_SIZE = 8 * 1024 * 1024; // Byte, AIV单卡数据量上限(除alltoallv外)
constexpr uint64_t SMALL_COUNT_16M = 16 * 1024 * 1024;      // 960 大小数据量边界

constexpr u32 CCU_MS_MODE = 2;
constexpr double DEFAULT_RANK_SIZE = 8.0;
constexpr u64 RS_2D_SMALL_DATA_SIZE = 1024 * 1024;
constexpr u64 RS_M2M_1D_MAX_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 CCU_PARALLEL_MAX_DATA_SIZE = 64 * 1024 * 1024;

constexpr u32 MAX_FRAME_NUM_FOR_CCU_ALGO = 16; // 仅AR和RS

enum class SelectorStatus { MATCH, NOT_MATCH };

const std::map<HcclCMDType, std::string> OP_TYPE_TO_AICPU_SOLE_ALG_MAP = {
    {HcclCMDType::HCCL_CMD_ALLGATHER, "InsAllGatherMesh"},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "AicpuReduceScatterSoleNHR"},
    {HcclCMDType::HCCL_CMD_ALLREDUCE, "AicpuAllReduceSoleNHR"},
    {HcclCMDType::HCCL_CMD_ALLTOALL, "InsAlltoAllMesh"},
    {HcclCMDType::HCCL_CMD_ALLTOALLV, "InsAlltoAllvMesh"},
    {HcclCMDType::HCCL_CMD_ALLTOALLVC, "InsAlltoAllvcMesh"},
};

const std::map<HcclCMDType, std::string> OP_TYPE_TO_CCU_1D_ALG_MAP = {
    {HcclCMDType::HCCL_CMD_ALLGATHER, "CcuMSAllGatherSoleMesh"},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "CcuMSReduceScatterSoleMesh"},
    {HcclCMDType::HCCL_CMD_ALLREDUCE, "CcuMSAllReduceSoleMesh"},
    {HcclCMDType::HCCL_CMD_REDUCE, "CcuMSReduceSoleMesh"},
    {HcclCMDType::HCCL_CMD_ALLTOALL, "CcuSchedAllToAllSoleMesh"},
    {HcclCMDType::HCCL_CMD_ALLTOALLV, "CcuSchedAllToAllVSoleMesh"},
    {HcclCMDType::HCCL_CMD_HALF_ALLTOALLV, "CcuHalfAll2AllVMesh1D"},
};

const std::map<HcclCMDType, std::string> OP_TYPE_TO_CCU_2D_ALG_MAP = {
    {HcclCMDType::HCCL_CMD_ALLGATHER, "CcuAllGatherMesh2D"},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "CcuReduceScatterMesh2D"},
    {HcclCMDType::HCCL_CMD_ALLREDUCE, "CcuAllReduceMesh2DOneShot"},
    {HcclCMDType::HCCL_CMD_REDUCE, "CcuReduceMesh2D"},
    {HcclCMDType::HCCL_CMD_ALLTOALL, "CcuAlltoAllMesh2D"},
};

const std::map<HcclCMDType, std::string> OP_TYPE_TO_DPU_ALG_MAP = {

};

const std::unordered_map<std::string, std::string> RES_REUSE_ALG
    = {{"AicpuReduceScatterSoleMesh", "InsReduceScatterMeshClass"},
       {"AicpuReduceScatterSoleMeshChunk", "InsReduceScatterMeshClass"},
       {"AicpuAllReduceSoleMeshOneShot", "InsAllReduceMeshClass"},
       {"AicpuAllReduceSoleMeshTwoShot", "InsAllReduceMeshClass"},
       {"AicpuSendSoleMesh", "InsSendRecv"},
       {"AicpuRecvSoleMesh", "InsSendRecv"}};

class AutoSelectorBase {
public:
    SelectorStatus Select(OpParam& opParam, TopoInfoWithNetLayerDetails* topoInfo, std::string& selectAlgName) const;
    bool IsDefaultAlg(const HcclAlgoType algoType) const;
    bool IsSmallData(const u64 dataSize) const;
    bool IsLargeData(const u64 dataSize) const;
    virtual SelectorStatus SelectCcuMsAlgo(
        const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
        const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const;
    virtual SelectorStatus SelectCcuScheduleAlgo(
        const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
        const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const;
    virtual SelectorStatus SelectAicpuAlgo(
        const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
        const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const;
    virtual SelectorStatus SelectAivAlgo(
        const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
        const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const;
    virtual SelectorStatus SelectDPUAlgo(
        const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
        const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const;
    virtual SelectorStatus SelectNdaAlgo(
        const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
        const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const;
    bool IsStarsState(const OpExecuteConfig& opExecuteConfig) const;
    static bool IsRollBackAiv(OpParam& opParam, TopoInfoWithNetLayerDetails* topoInfo);
    static bool IsLayerAllConnetedWithTopo(
        const TopoInfoWithNetLayerDetails* topoInfo, const u32 netLayer, const CommTopo topoType);
    // CLOS 链路跑 CCU Mesh 要求拓扑中只有一个 die 的 channel：level0 出现两个及以上 CLOS 拓扑实例时，
    // 两个实例分属不同 die，Mesh 建链会出现 die0 连 die1 的异常，此时不允许选 Mesh。
    // 头文件内联实现：部分 UT 目标单独编译 selector 且不链接 auto_selector_base.cc，避免链接缺符号
    static bool IsLevel0SingleClosInst(const TopoInfoWithNetLayerDetails* topoInfo)
    {
        constexpr u32 level0 = 0;
        if (topoInfo == nullptr || topoInfo->topoInstDetailsOfLayer.size() <= level0) {
            return false;
        }
        const auto& rankNumMap = topoInfo->topoInstDetailsOfLayer[level0].rankNumForTopoType;
        auto closItr = rankNumMap.find(COMM_TOPO_CLOS);
        // CLOS 实例列表缺失或为空时均不可判定单 die，保守回退 NHR
        return closItr != rankNumMap.end() && closItr->second.size() == 1;
    }

    static HcclResult CheckMeshNumEqualToClosNum(const TopoInfoWithNetLayerDetails* topoInfo, bool& isEqual);
    static HcclResult CheckClosNumMultipleOfMeshNum(const TopoInfoWithNetLayerDetails* topoInfo, bool& isMultiple);
    static bool IsTwoLevelNetLayer(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam);
    static bool IsDevType960();
    bool IsInputOutputOverlap(const OpParam& opParam) const;
    bool IsSmallDataCCU(const u64 dataSize, const u64 rankSize) const;
    // 计算非对称拓扑展开后的 layer0 子组数（框数）。
    // 使用全局 instSizeListOfLayer[0] 的 GCD，避免 localNetInsSizeOfLayer[0] 在非对称场景下各 rank 不同。
    // frameNum = userRankSize / gcd(instSizeListOfLayer[0])，例如 6 卡 / GCD(4,2)=2 / 2 = 3 框。
    static u32 CalcFrameNum(const TopoInfoWithNetLayerDetails* topoInfo);

protected:
    // 未配置或仅有一个配置项时沿用原默认vector；不补齐或截断配置。
    static std::vector<HcclAlgoType>
    GetConfiguredAlgorithms(HcclCMDType opType, const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap)
    {
        std::vector<HcclAlgoType> algos
            = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
        auto it = configAlgMap.find(opType);
        if ((it != configAlgMap.end()) && (it->second.size() > 1)) {
            algos = it->second;
        }
        return algos;
    }

    static bool IsPairwiseCapable(const TopoInfoWithNetLayerDetails* topoInfo)
    {
        constexpr u32 PAIRWISE_RANK_NUM_PER_BOARD = 8;                           // 模板硬约束：每板 8 卡
        constexpr u32 PAIRWISE_UNIT_RANK_SIZE = 2 * PAIRWISE_RANK_NUM_PER_BOARD; // 2 套流 × 8 卡/板
        constexpr u32 PAIRWISE_MIN_RANK_SIZE = 64; // 单pod不走 Pairwise，仅跨 pod 场景启用

        if (topoInfo->topoLevelNums <= 1 || topoInfo->level0Topo != Level0Shape::MESH_1D) {
            return false;
        }

        if (topoInfo->deviceNumPerModule != PAIRWISE_RANK_NUM_PER_BOARD) {
            return false;
        }

        if (topoInfo->physicalLevels.empty()) {
            return false;
        }
        const auto& topLayer = topoInfo->physicalLevels.back();

        return topoInfo->userRankSize > PAIRWISE_MIN_RANK_SIZE
               && (topoInfo->userRankSize % PAIRWISE_UNIT_RANK_SIZE == 0) && topoInfo->level0Symmetric
               && (topLayer.instSizeListByLayer.size() == 1)
               && (topoInfo->userRankSize == topLayer.localRanks.size()); // 最高层全连接
    }

private:
    bool ProcessAivConfig(
        OpParam& opParam, TopoInfoWithNetLayerDetails* topoInfo,
        const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName,
        SelectorStatus& ret) const;
};

inline bool Is64BitDataType(const HcclDataType dataType)
{
    return dataType == HcclDataType::HCCL_DATA_TYPE_INT64 || dataType == HcclDataType::HCCL_DATA_TYPE_UINT64
           || dataType == HcclDataType::HCCL_DATA_TYPE_FP64;
}

inline bool IsDataTypeOrReduceTypeSpecial(const OpParam& opParam)
{
    return UNSUPPORTED_64BIT.count(opParam.DataDes.dataType) > 0
           || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD;
}

inline bool Is8BitDataType(const HcclDataType dataType) { return dataType == HcclDataType::HCCL_DATA_TYPE_INT8; }

// AIV_ONLY 选路失败的对外报错前缀(契约文本, 日志采集/用例依赖逐字匹配, 勿改动)
inline constexpr const char* AIV_ONLY_NOT_MATCH_PREFIX
    = "Failed to select AIV algorithm while configured as AIV_ONLY. ";

} // namespace ops_hccl

// AIV_ONLY 额外打 ERROR（前缀见 ops_hccl::AIV_ONLY_NOT_MATCH_PREFIX，直接报错不回退，原因同 BASE_LOG）
// 宏体内命名空间符号一律加 ops_hccl:: 限定, 保证宏在任意展开点自包含
#define HCCL_AIV_NOT_MATCH_LOG(opParam, BASE_LOG, fmt, ...)                           \
    do {                                                                              \
        BASE_LOG(fmt, ##__VA_ARGS__);                                                 \
        if ((opParam).opExecuteConfig == ops_hccl::OpExecuteConfig::AIV_ONLY) {       \
            HCCL_ERROR("%s" fmt, ops_hccl::AIV_ONLY_NOT_MATCH_PREFIX, ##__VA_ARGS__); \
        }                                                                             \
    } while (0)

#endif
