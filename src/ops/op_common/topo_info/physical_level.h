/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_TOPO_PHYSICAL_LEVEL
#define OPS_HCCL_SRC_OPS_TOPO_PHYSICAL_LEVEL

#include <hccl/hccl_types.h>
#include <vector>
#include "alg_param.h"

namespace ops_hccl {

// Endpoint数量的合理性阈值, 不是防御性截断: 该值是纯本地量(接口数 x 协议数), 不随rankSize增长,
// 超过该量级只可能是HCOMM侧异常, 此时局部降级并告警
constexpr u32 ENDPOINT_NUM_SANITY_LIMIT = 64;

// 单个端口组的端口数合理性阈值。HCOMM侧MAX_PORT_NUM是32, 驱动侧UB口上限36,
// 超过该量级说明ENDPOINT_ATTR_BW_COEFF返回的不是端口数, 整个Level的portNums不可信
constexpr u32 PORT_NUM_SANITY_LIMIT = 64;

// ---- 纯函数: 不依赖HcclComm与RankGraph, 可离线UT (physical_level_normalize.cc) ----

/**
 * EndpointDesc的稳定排序键。GetEndpointDesc的输出是哈希序, 必须归一化后再保存。
 * 按字段比较而不是memcmp整个结构体: 尾部raws在HCOMM侧从未赋值。
 */
bool EndpointDescLess(const EndpointDesc& lhs, const EndpointDesc& rhs);

/**
 * 两个EndpointDesc是否指向同一个iface。判据是commAddr —— HCOMM的endpointToIfaceMap以
 * (commAddr, protocol)为键, 同addr不同protocol必然映射到同一个iface。用于按链路统计端口数。
 */
bool CommAddrEqual(const CommAddr& lhs, const CommAddr& rhs);

/**
 * 候选范围的标准化: 归一 -> 排序 -> 范围校验, 不做合并。candidates按值语义被移动消耗。
 * 普通拓扑保持包含链; 分轨按netLayer保存物理来源, 仅要求层内及Server/超节点范围成链。
 * 返回HCCL_E_NOT_SUPPORT表示不满足对应范围约束或排序键取不到值, 由调用方降级。
 */
HcclResult NormalizePhysicalLevels(
    std::vector<PhysicalLevelInfo>& candidates, u32 userRank, u32 userRankSize, std::vector<PhysicalLevelInfo>& levels);

/**
 * 标准化结果的一致性校验, 逐条对应标准化后应当成立的不变量。
 */
HcclResult ValidatePhysicalLevels(const std::vector<PhysicalLevelInfo>& levels, u32 userRank, u32 userRankSize);

// ---- 依赖HcclComm (physical_level_build.cc) ----

/**
 * 构建topoInfo->physicalLevels。原始网络成员、分区等数据错误直接返回。
 * 分轨连通性仅依据全局分区规模判定, 不以本地交集或比例模型决定准入。
 * 在超节点内可达且网络层信息完整的前提下, 跨超节点连接数量上限不足时返回HCCL_E_NOT_SUPPORT。
 * 已证明连通或信息不足均继续处理; 成功不保证实际连通或算法支持, 视图表达限制仍降级为空视图。
 * 相同网络层和全局分区信息得到相同连通性结论, 不替代跨rank数据一致性校验。
 */
HcclResult BuildPhysicalLevels(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo);

} // namespace ops_hccl

#endif
