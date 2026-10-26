/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "physical_level.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>

#include "log.h"
#include "hccl_rank_graph_dl.h"

namespace ops_hccl {
namespace {

    // 日志里一个vector最多展开的元素数, 超出部分省略
    constexpr size_t LOG_VEC_MAX_ITEM = 16;

    struct NetLayerSnapshot {
        u32 netLayer = 0;
        std::vector<u32> localRanks;
        std::vector<u32> instSizeList;
    };

    std::string VecToStr(const std::vector<u32>& vec)
    {
        std::string str;
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i >= LOG_VEC_MAX_ITEM) {
                str += ",...";
                break;
            }
            if (i > 0) {
                str += ",";
            }
            str += std::to_string(vec[i]);
        }
        return str;
    }

    // 把一个Level拼成一行日志。localRanks只打首尾与个数, 不整条展开: 顶层那一级等于整个通信域,
    // 万卡场景整条打出来没人看得完, 而升序与含myRank由校验侧保证
    std::string DescribeLevel(const PhysicalLevelInfo& level, size_t idx, size_t total)
    {
        return "level[" + std::to_string(idx) + "/" + std::to_string(total) + "] rankNum["
               + std::to_string(level.localRanks.size()) + "] ranks["
               + (level.localRanks.empty() ?
                      std::string("-") :
                      std::to_string(level.localRanks.front()) + ".." + std::to_string(level.localRanks.back()))
               + "] view[" + std::to_string(static_cast<u32>(level.view)) + "] instSizeListByLayer["
               + VecToStr(level.instSizeListByLayer) + "] ref[layer " + std::to_string(level.ref.netLayer) + " inst "
               + std::to_string(level.ref.topoInstId) + "] hasTopoInst[" + std::to_string(level.hasTopoInst ? 1 : 0)
               + "] topoType[" + std::to_string(static_cast<s32>(level.topoType)) + "] locType["
               + std::to_string(static_cast<s32>(level.locType)) + "] protocolNum["
               + std::to_string(level.protocols.size()) + "] portNums[" + VecToStr(level.portNums) + "]";
    }

    /**
     * 提取当前rank在指定TopoInstance上的Endpoint快照。
     * 返回void: endpoints是payload叶子, 不参与排序键与范围链结构, 全部失败路径都局部降级为空。
     */
    void FetchEndpoints(HcclComm comm, u32 layer, u32 instId, std::vector<EndpointDesc>& out)
    {
        out.clear();
        u32 num = 0;
        // 与下面的num == 0分开判: 取数失败是真异常, 需要留日志
        if (HcclRankGraphGetEndpointNum(comm, layer, instId, &num) != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[PhysicalLevel][Build] get endpoint num failed at layer[%u] inst[%u], skip endpoints", layer, instId);
            return;
        }
        // 0是合法结果, 不是错误: 当前rank在该topoInst上没有接口/协议时就是0
        if (num == 0) {
            HCCL_DEBUG("[PhysicalLevel][Build] no endpoint at layer[%u] inst[%u]", layer, instId);
            return;
        }
        // 合理性阈值, 不截断: 截断只会把异常掩盖成"正常但数据少", 直接局部降级并告警
        if (num > ENDPOINT_NUM_SANITY_LIMIT) {
            HCCL_WARNING(
                "[PhysicalLevel][Build] implausible endpoint num[%u] at layer[%u] inst[%u], skip endpoints", num, layer,
                instId);
            return;
        }

        // num是实际写入条数的上界(GetEndpointNum求和时不去重), 必须以回写的descNum为准resize
        std::vector<EndpointDesc> buf(num);
        u32 actualNum = num;
        if (HcclRankGraphGetEndpointDesc(comm, layer, instId, &actualNum, buf.data()) != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[PhysicalLevel][Build] get endpoint desc failed at layer[%u] inst[%u], skip endpoints", layer, instId);
            return;
        }
        if (actualNum > num) {
            HCCL_WARNING(
                "[PhysicalLevel][Build] endpoint descNum[%u] exceeds requested[%u] at layer[%u] inst[%u], skip "
                "endpoints",
                actualNum, num, layer, instId);
            return;
        }
        buf.resize(actualNum);
        // GetEndpointDesc的输出顺序是unordered_map哈希序, 必须归一化后再保存
        std::sort(buf.begin(), buf.end(), EndpointDescLess);
        out = std::move(buf);
    }

    /**
     * 采集本rank在该Level上各条物理链路的端口数, 降序写入out。按iface(commAddr)去重, 一条链路一项:
     * 一个iface有N种协议就有N个EndpointDesc, 逐endpoint查会把同一条链路的端口数重复计入。
     * 全有或全无: 任一条取不到就整个清空 —— 残缺数组会让消费侧算出"看着合理但偏小"的总端口数。
     */
    void FetchPortNums(HcclComm comm, u32 myRank, const std::vector<EndpointDesc>& endpoints, std::vector<u32>& out)
    {
        out.clear();
        // 空有两种来源: 本就没有接口, 或FetchEndpoints已降级(含HCOMM低版本弱符号未命中),
        // 因此这里不需要再做一次能力探测
        if (endpoints.empty()) {
            return;
        }

        std::vector<CommAddr> seenAddrs; // 已计入的iface。规模是个位数, 线性查找即可
        std::vector<u32> portNums;
        for (const auto& desc : endpoints) {
            bool seen = false;
            for (const auto& addr : seenAddrs) {
                if (CommAddrEqual(addr, desc.commAddr)) {
                    seen = true;
                    break;
                }
            }
            if (seen) {
                continue; // 同一个iface的另一种协议, 端口数已经计过
            }

            EndpointAttrBwCoeff portNum{};
            // ENDPOINT_ATTR_BW_COEFF名为"带宽系数", HCOMM侧实现即iface->GetPorts().size()
            if (HcclRankGraphGetEndpointInfo(
                    comm, myRank, &desc, ENDPOINT_ATTR_BW_COEFF, sizeof(EndpointAttrBwCoeff), &portNum)
                != HCCL_SUCCESS) {
                HCCL_WARNING(
                    "[PhysicalLevel][Build] get port num failed for rank[%u] protocol[%d], drop port nums of this "
                    "level",
                    myRank, static_cast<s32>(desc.protocol));
                return;
            }
            // 0与超限都判为不可信, 口径对齐op_common.cc的BuildChannelInfo
            if (portNum == 0 || portNum > PORT_NUM_SANITY_LIMIT) {
                HCCL_WARNING(
                    "[PhysicalLevel][Build] implausible port num[%u] for rank[%u] protocol[%d], drop port nums of "
                    "this level",
                    portNum, myRank, static_cast<s32>(desc.protocol));
                return;
            }
            seenAddrs.push_back(desc.commAddr);
            portNums.push_back(static_cast<u32>(portNum));
        }
        // 降序。与endpoints的(protocol, locType, addr)序无关, 两者是同一批iface的两种独立排列
        std::sort(portNums.begin(), portNums.end(), std::greater<u32>());
        out = std::move(portNums);
    }

    /**
     * 从endpoints提炼该Level的位置与协议集合。locType各endpoint不一致时置RESERVED并告警:
     * 一个Level对应一种网络平面, 位置本应唯一, 给出任一个都会误导"是否需要host网卡"的判断。
     */
    void FetchLocAndProtocols(
        const std::vector<EndpointDesc>& endpoints, EndpointLocType& locType, std::vector<CommProtocol>& protocols)
    {
        locType = EndpointLocType::ENDPOINT_LOC_TYPE_RESERVED;
        protocols.clear();
        if (endpoints.empty()) {
            return;
        }

        locType = endpoints.front().loc.locType;
        for (const auto& desc : endpoints) {
            if (desc.loc.locType != locType) {
                HCCL_WARNING(
                    "[PhysicalLevel][Build] mixed endpoint locType[%d] vs [%d] on one level, mark location unknown",
                    static_cast<s32>(desc.loc.locType), static_cast<s32>(locType));
                locType = EndpointLocType::ENDPOINT_LOC_TYPE_RESERVED;
                break;
            }
        }

        protocols.reserve(endpoints.size());
        for (const auto& desc : endpoints) {
            protocols.push_back(desc.protocol);
        }
        // 去重升序: endpoints已按protocol为首键排过, 但同一协议可能出现在多个iface上
        std::sort(protocols.begin(), protocols.end());
        protocols.erase(std::unique(protocols.begin(), protocols.end()), protocols.end());
    }

    // 取该layer本地NetInstance的rank集合, 并与netLayerDetails做跨调用一致性校验
    HcclResult
    FetchLocalNetRanks(HcclComm comm, const TopoInfoWithNetLayerDetails& topoInfo, u32 layer, std::vector<u32>& ranks)
    {
        u32* rawRanks = nullptr;
        u32 rankNum = 0;
        HcclResult ret = HcclRankGraphGetRanksByLayer(comm, layer, &rawRanks, &rankNum);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[PhysicalLevel][Connectivity] get ranks by layer[%u] failed, ret[%d]", layer, ret);
            return ret;
        }
        if (rawRanks == nullptr || rankNum == 0 || rankNum > topoInfo.userRankSize) {
            HCCL_ERROR("[PhysicalLevel][Connectivity] invalid rank list at layer[%u], rankNum[%u]", layer, rankNum);
            return HCCL_E_PARA;
        }
        // 跨调用一致性校验: localNetInsSizeOfLayer来自ExtractNetLayerDetails中的另一次调用,
        // 与此处不同源, 不一致说明RankGraph在两次调用之间发生了变化
        if (rankNum != topoInfo.netLayerDetails.localNetInsSizeOfLayer[layer]) {
            HCCL_ERROR(
                "[PhysicalLevel][Connectivity] layer[%u] rankNum[%u] mismatches localNetInsSize[%u]", layer, rankNum,
                topoInfo.netLayerDetails.localNetInsSizeOfLayer[layer]);
            return HCCL_E_PARA;
        }
        // HCOMM复用返回数组, 下一次查询前立即复制。先核对数量, 再读取数组。
        ranks.assign(rawRanks, rawRanks + rankNum);
        // 排序但不去重, 避免把错误成员列表归一成合法范围。
        std::sort(ranks.begin(), ranks.end());
        if (ranks.back() >= topoInfo.userRankSize || std::adjacent_find(ranks.begin(), ranks.end()) != ranks.end()
            || !std::binary_search(ranks.begin(), ranks.end(), topoInfo.userRank)) {
            HCCL_ERROR(
                "[PhysicalLevel][Connectivity] invalid members at layer[%u], rank[%u], userRankSize[%u]", layer,
                topoInfo.userRank, topoInfo.userRankSize);
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    /**
     * 校验分区: 非空、总和等于通信域规模、无空分区, 且本地实例的规模是其中某个分区的大小。
     */
    HcclResult
    ValidateInstSizeLayout(const std::vector<u32>& instSizeList, u32 layer, size_t localRankNum, u32 userRankSize)
    {
        if (instSizeList.empty() || instSizeList.size() > userRankSize) {
            HCCL_ERROR(
                "[PhysicalLevel][Connectivity] invalid inst count[%zu] at layer[%u]", instSizeList.size(), layer);
            return HCCL_E_PARA;
        }
        // 正常路径永不触发(ExtractNetLayerDetails已用同一等式先行校验过)。
        // 保留它只为在HCOMM改变分层语义时第一时间暴露
        const u64 totalRankNum = std::accumulate(instSizeList.begin(), instSizeList.end(), u64{0});
        if (totalRankNum != userRankSize) {
            HCCL_ERROR(
                "[PhysicalLevel][Connectivity] layer[%u] inst size sum[%llu] mismatches userRankSize[%u]", layer,
                static_cast<unsigned long long>(totalRankNum), userRankSize);
            return HCCL_E_PARA;
        }
        // 0能完整穿过其余检查, 于是幽灵空分区会被当成真实Instance计入
        if (std::find(instSizeList.begin(), instSizeList.end(), 0U) != instSizeList.end()) {
            HCCL_ERROR("[PhysicalLevel][Connectivity] layer[%u] inst size list contains a zero entry", layer);
            return HCCL_E_PARA;
        }
        // 顺序无关的自检: 本地实例的规模必须是这一层真实存在的某个分区大小。两个量来源独立
        // (前者来自GetRanksByLayer, 后者来自GetInstSizeListByLayer), 对不上说明两次调用之间
        // RankGraph发生了变化
        if (std::find(instSizeList.begin(), instSizeList.end(), static_cast<u32>(localRankNum)) == instSizeList.end()) {
            HCCL_ERROR(
                "[PhysicalLevel][Connectivity] layer[%u] local instance has [%zu] ranks, which is not one of the "
                "inst sizes",
                layer, localRankNum);
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    // 取该netLayer本地NetInstance的rank集合与全层分区。这是"合一"里ranktable那一半:
    // 只有NetInstance看得到兄弟实例, 因此只有它能给出全局分区
    HcclResult FetchNetInstance(
        HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, u32 layer, std::vector<u32>& ranks,
        std::vector<u32>& instSizeListByLayer)
    {
        const NetLayerDetails& details = topoInfo->netLayerDetails;
        if (layer >= details.localNetInsSizeOfLayer.size() || layer >= details.instSizeListOfLayer.size()
            || layer >= details.netInstNumOfLayer.size()) {
            HCCL_ERROR(
                "[PhysicalLevel][Connectivity][MissingMetadata] missing partition information at layer[%u]", layer);
            return HCCL_E_NOT_SUPPORT;
        }
        HcclResult ret = FetchLocalNetRanks(comm, *topoInfo, layer, ranks);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        // 原样透传HCOMM的返回序, 不重排。该序不保证任何含义(HCOMM侧以unordered_map实现),
        // physicalLevels的消费方只做与顺序无关的判断(IsInstListSymmetric/CalcGcd)。
        // 不据此前缀和恢复远端成员; 旧topoInfo初始化中的布局推导不属于该字段的契约。
        instSizeListByLayer = details.instSizeListOfLayer[layer];
        if (instSizeListByLayer.size() != details.netInstNumOfLayer[layer]) {
            HCCL_ERROR("[PhysicalLevel][Connectivity] inconsistent partition count at layer[%u]", layer);
            return HCCL_E_PARA;
        }
        return ValidateInstSizeLayout(instSizeListByLayer, layer, ranks.size(), topoInfo->userRankSize);
    }

    HcclResult FetchNetLayerSnapshots(
        HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, std::vector<NetLayerSnapshot>& snapshots)
    {
        auto layers = topoInfo->netLayerDetails.netLayers;
        std::sort(layers.begin(), layers.end());
        if (layers.size() != topoInfo->netLayerDetails.netLayerNum
            || std::adjacent_find(layers.begin(), layers.end()) != layers.end()) {
            HCCL_ERROR(
                "[PhysicalLevel][Connectivity] inconsistent or duplicate netLayers, rank[%u]", topoInfo->userRank);
            return HCCL_E_PARA;
        }
        if (layers.empty() && topoInfo->userRankSize > 1) {
            HCCL_ERROR(
                "[PhysicalLevel][Connectivity][MissingMetadata] no netLayer information, rank[%u]", topoInfo->userRank);
            return HCCL_E_NOT_SUPPORT;
        }
        for (u32 layer : layers) {
            NetLayerSnapshot snapshot;
            snapshot.netLayer = layer;
            CHK_RET(FetchNetInstance(comm, topoInfo, layer, snapshot.localRanks, snapshot.instSizeList));
            snapshots.push_back(std::move(snapshot));
        }
        return HCCL_SUCCESS;
    }

    HcclResult ValidateRailConnectivity(
        const std::vector<NetLayerSnapshot>& snapshots, const NetLayerSnapshot& pod, const NetLayerSnapshot& rail,
        u32 myRank, u32 rankSize)
    {
        // 只使用全局规模, 不以本rank所在超节点、轨或交集决定准入。各rank取得相同层列表和
        // 完整分区时结论相同; 假定NetInstance内理论可达, 不据此保证具体算法或实际建链成功。
        const u32 minPodSize = *std::min_element(pod.instSizeList.begin(), pod.instSizeList.end());
        const u32 maxRailSize = *std::max_element(rail.instSizeList.begin(), rail.instSizeList.end());
        if (maxRailSize > rankSize - minPodSize) {
            // 最大轨无法放入任一超节点的补集, 因而必然覆盖全部超节点。
            HCCL_INFO(
                "[PhysicalLevel][Connectivity] global rail sizes prove connectivity, rank[%u], userRankSize[%u], "
                "minPodSize[%u], maxRailSize[%u]",
                myRank, rankSize, minPodSize, maxRailSize);
            return HCCL_SUCCESS;
        }

        // 将超节点收缩为P个点。一个s卡实例最多减少s-1个连通分量, 所以每层的上限为N-实例数。
        // 必须计入layer3等额外跨超节点网络; 多层累计使用u64。上限足够不代表实际连接充分。
        // 在完整分区和实例内理论可达的模型下, 上限不足是连通性的否证, 不是启发式拒绝。
        u64 maxConnections = 0;
        for (const auto& layer : snapshots) {
            if (layer.netLayer >= RAIL_NET_LAYER) {
                maxConnections += static_cast<u64>(rankSize) - layer.instSizeList.size();
            }
        }
        const size_t podNum = pod.instSizeList.size();
        if (maxConnections < podNum - 1) {
            HCCL_ERROR(
                "[PhysicalLevel][Connectivity][Disconnected] disconnected: global connection capacity is insufficient, "
                "rank[%u], userRankSize[%u], podNum[%zu], maxConnections[%llu]",
                myRank, rankSize, podNum, static_cast<unsigned long long>(maxConnections));
            return HCCL_E_NOT_SUPPORT;
        }
        HCCL_INFO(
            "[PhysicalLevel][Connectivity] global sizes cannot determine rail connectivity, continue with "
            "topology matching and channel acquisition, rank[%u], userRankSize[%u], podNum[%zu], "
            "maxConnections[%llu]",
            myRank, rankSize, podNum, static_cast<unsigned long long>(maxConnections));
        return HCCL_SUCCESS;
    }

    HcclResult ValidatePhysicalConnectivity(const std::vector<NetLayerSnapshot>& snapshots, u32 myRank, u32 rankSize)
    {
        // 分区已校验总和为rankSize, 只有一个实例即覆盖全域。具体算法所需链路仍由建链检查。
        if (rankSize == 1 || std::any_of(snapshots.begin(), snapshots.end(), [rankSize](const NetLayerSnapshot& layer) {
                return layer.instSizeList.size() == 1 && layer.instSizeList.front() == rankSize;
            })) {
            return HCCL_SUCCESS;
        }
        const NetLayerSnapshot* pod = nullptr;
        const NetLayerSnapshot* rail = nullptr;
        for (const auto& layer : snapshots) {
            if (layer.netLayer == POD_NET_LAYER) {
                pod = &layer;
            } else if (layer.netLayer == RAIL_NET_LAYER) {
                rail = &layer;
            }
        }
        if (pod == nullptr || rail == nullptr) {
            // 缺少本地层信息不能排除代表卡或其他网络层提供连接, 不将信息不足判为断网。
            HCCL_INFO(
                "[PhysicalLevel][Connectivity] pod or rail layer unavailable, connectivity undetermined, "
                "continue with topology matching and channel acquisition, rank[%u], userRankSize[%u]",
                myRank, rankSize);
            return HCCL_SUCCESS;
        }
        for (const auto& layer : snapshots) {
            if (layer.netLayer < POD_NET_LAYER
                && !std::includes(
                    pod->localRanks.begin(), pod->localRanks.end(), layer.localRanks.begin(), layer.localRanks.end())) {
                HCCL_ERROR(
                    "[PhysicalLevel][Connectivity] layer[%u] exceeds local pod, rank[%u]", layer.netLayer, myRank);
                return HCCL_E_PARA;
            }
        }
        return ValidateRailConnectivity(snapshots, *pod, *rail, myRank, rankSize);
    }

    // 取该layer上全部TopoInstance的id。空map是合法结果, 返回空列表且不算失败
    HcclResult FetchTopoInstIds(HcclComm comm, u32 layer, std::vector<u32>& instIds)
    {
        instIds.clear();
        u32* rawInstIds = nullptr;
        u32 instNum = 0;
        HcclResult ret = HcclRankGraphGetTopoInstsByLayer(comm, layer, &rawInstIds, &instNum);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[PhysicalLevel][Build] get topo insts of layer[%u] failed, ret[%d]", layer, ret);
            return HCCL_E_INTERNAL;
        }
        if (instNum == 0) {
            // 空map即返回0且不报错。意味着该层没有endpoints与topoType可供建链,
            // 调用方据此把hasTopoInst置false
            HCCL_DEBUG("[PhysicalLevel][Build] layer[%u] has no topo instance", layer);
            return HCCL_SUCCESS;
        }
        if (rawInstIds == nullptr) {
            HCCL_WARNING("[PhysicalLevel][Build] topo insts of layer[%u] is null while num[%u]", layer, instNum);
            return HCCL_E_INTERNAL;
        }
        // 立即复制: 该接口的下一次调用会clear()并重填同一个成员vector
        instIds.assign(rawInstIds, rawInstIds + instNum);
        return HCCL_SUCCESS;
    }

    // 采集单个TopoInstance并追加到out。当前rank不在其中时跳过且不写out, 不算失败
    HcclResult
    AppendTopoInstLevel(HcclComm comm, u32 myRank, u32 layer, u32 instId, std::vector<PhysicalLevelInfo>& out)
    {
        u32* rawRanks = nullptr;
        u32 rankNum = 0;
        if (HcclRankGraphGetRanksByTopoInst(comm, layer, instId, &rawRanks, &rankNum) != HCCL_SUCCESS
            || rawRanks == nullptr) {
            HCCL_WARNING("[PhysicalLevel][Build] get ranks by topo inst[%u] of layer[%u] failed", instId, layer);
            return HCCL_E_INTERNAL;
        }
        std::vector<u32> ranks(rawRanks, rawRanks + rankNum);

        // GetTopoInstsByLayer返回的应当只含当前rank所在的topoInstance, 这里再过滤一次兜底
        if (std::find(ranks.begin(), ranks.end(), myRank) == ranks.end()) {
            HCCL_DEBUG(
                "[PhysicalLevel][Build] skip sibling topo inst[%u] of layer[%u], myRank[%u] not in it", instId, layer,
                myRank);
            return HCCL_SUCCESS;
        }

        // 必须用按topoInst的GetTopoType。按netLayer的GetTopoTypeByLayer查的是NetType,
        // A5上Mesh层是TOPO_FILE_DESC描述的, 会返回COMM_TOPO_CUSTOM, TopoTypeOrder定不了序
        CommTopo topoType = CommTopo::COMM_TOPO_RESERVED;
        if (HcclRankGraphGetTopoType(comm, layer, instId, &topoType) != HCCL_SUCCESS) {
            HCCL_WARNING("[PhysicalLevel][Build] get topo type of inst[%u] layer[%u] failed", instId, layer);
            return HCCL_E_INTERNAL;
        }

        PhysicalLevelInfo level;
        level.localRanks = std::move(ranks);
        level.ref.netLayer = layer;
        level.ref.topoInstId = instId;
        level.hasTopoInst = true;
        level.topoType = topoType;
        // 以下三项无返回值: 内部失败一律局部降级, 理由见各自声明处
        FetchEndpoints(comm, layer, instId, level.endpoints);
        FetchLocAndProtocols(level.endpoints, level.locType, level.protocols);
        FetchPortNums(comm, myRank, level.endpoints, level.portNums);
        out.push_back(std::move(level));
        return HCCL_SUCCESS;
    }

    // 取该netLayer上、含当前rank的每个TopoInstance的rank集合, 并把链路属性填进level。
    // 这是"合一"里topo那一半: 只有TopoInstance带得出形态/位置/协议/端口数
    HcclResult FetchTopoInstances(HcclComm comm, u32 myRank, u32 layer, std::vector<PhysicalLevelInfo>& out)
    {
        out.clear();
        std::vector<u32> instIds;
        HcclResult ret = FetchTopoInstIds(comm, layer, instIds);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        for (u32 instId : instIds) {
            ret = AppendTopoInstLevel(comm, myRank, layer, instId, out);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
        }
        return HCCL_SUCCESS;
    }

    /**
     * 按netLayer把ranktable层级与topo层级合成候选Level。合并规则:
     *   同范围的TopoInstance -> 与NetInstance合并成一级(view=GLOBAL); 更小的 -> 独立成级(view=LOCAL);
     *   该层没有TopoInstance -> NetInstance独立成级(hasTopoInst=false)。
     */
    HcclResult BuildLayerCandidates(
        HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, const NetLayerSnapshot& snapshot,
        std::vector<PhysicalLevelInfo>& candidates)
    {
        const u32 myRank = topoInfo->userRank;
        const u32 layer = snapshot.netLayer;
        const auto& netRanks = snapshot.localRanks;
        const auto& instSizeListByLayer = snapshot.instSizeList;

        std::vector<PhysicalLevelInfo> topoLevels;
        CHK_RET(FetchTopoInstances(comm, myRank, layer, topoLevels));

        // NetInstance快照已检查并排序; TopoInstance另行排序后按成员比较。

        bool merged = false;
        for (auto& level : topoLevels) {
            std::vector<u32> sortedTopoRanks = level.localRanks;
            std::sort(sortedTopoRanks.begin(), sortedTopoRanks.end());
            if (sortedTopoRanks == netRanks) {
                // 同范围: 把NetInstance的全局分区并进来。分区是该层的全局事实, 同层多个
                // 同范围TopoInstance(如netLayer 0的Mesh与CLOS)各自都持有它
                level.view = PhysicalLevelView::GLOBAL;
                level.instSizeListByLayer = instSizeListByLayer;
                merged = true;
            } else {
                // 比NetInstance更细: 看不到兄弟NetInstance, 没有全局分区可言
                level.view = PhysicalLevelView::LOCAL;
                level.instSizeListByLayer.clear();
            }
            HCCL_DEBUG(
                "[PhysicalLevel][Build] layer[%u] inst[%u] rankNum[%zu] view[%u] topoType[%d] locType[%d] "
                "protocolNum[%zu] portNumCnt[%zu]",
                layer, level.ref.topoInstId, level.localRanks.size(), static_cast<u32>(level.view),
                static_cast<s32>(level.topoType), static_cast<s32>(level.locType), level.protocols.size(),
                level.portNums.size());
            candidates.push_back(std::move(level));
        }

        if (!merged) {
            // 没有同范围的TopoInstance: 分区信息仍有效但拿不到链路属性, 用hasTopoInst=false显式注明
            PhysicalLevelInfo level;
            level.localRanks = netRanks;
            level.view = PhysicalLevelView::GLOBAL;
            level.instSizeListByLayer = instSizeListByLayer;
            level.ref.netLayer = layer;
            level.ref.topoInstId = INVALID_UINT;
            level.hasTopoInst = false;
            HCCL_DEBUG(
                "[PhysicalLevel][Build] layer[%u] has no same-range topo instance, level carries partition only, "
                "rankNum[%zu]",
                layer, level.localRanks.size());
            candidates.push_back(std::move(level));
        }
        return HCCL_SUCCESS;
    }

    HcclResult BuildPhysicalLevelCandidates(
        HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, const std::vector<NetLayerSnapshot>& snapshots,
        std::vector<PhysicalLevelInfo>& candidates)
    {
        candidates.clear();
        if (topoInfo->netLayerDetails.netLayers.empty()) {
            HCCL_WARNING("[PhysicalLevel][Build] netLayers is empty, rank[%u]", topoInfo->userRank);
            return HCCL_E_INTERNAL;
        }
        // 只遍历netLayers里的layer: 规避HCOMM对非法layer的抛异常分支, 并保证ref.netLayer都来自
        // GetLayers的实际结果。收集先后无所谓, LevelLess是全序, 不依赖输入顺序
        for (const auto& snapshot : snapshots) {
            CHK_RET(BuildLayerCandidates(comm, topoInfo, snapshot, candidates));
        }
        return HCCL_SUCCESS;
    }

} // namespace

HcclResult BuildPhysicalLevels(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo)
{
    CHK_PTR_NULL(topoInfo);
    topoInfo->physicalLevels.clear();
    CHK_PTR_NULL(comm);
    CHK_PRT_RET(
        topoInfo->userRankSize == 0 || topoInfo->userRank >= topoInfo->userRankSize,
        HCCL_ERROR(
            "[PhysicalLevel][Connectivity] invalid rank[%u], userRankSize[%u]", topoInfo->userRank,
            topoInfo->userRankSize),
        HCCL_E_PARA);

    // 两类错误均直接失败: 必需元数据缺失/非法; 完整分区证明连接数量上限不足。
    // 后者已否证物理连通性, legacy算法不能补出不存在的跨分区连接, 因此不保留兜底。
    // 仅视图表达限制可在后续降级; 无法判定连通性的拓扑继续匹配与建链。
    std::vector<NetLayerSnapshot> snapshots;
    CHK_RET(FetchNetLayerSnapshots(comm, topoInfo, snapshots));
    CHK_RET(ValidatePhysicalConnectivity(snapshots, topoInfo->userRank, topoInfo->userRankSize));

    // 在临时对象中构建, 全部校验通过后再赋值; 降级或失败时physicalLevels保持为空
    std::vector<PhysicalLevelInfo> candidates;
    std::vector<PhysicalLevelInfo> levels;

    HcclResult ret = BuildPhysicalLevelCandidates(comm, topoInfo, snapshots, candidates);
    if (ret == HCCL_SUCCESS) {
        ret = NormalizePhysicalLevels(candidates, topoInfo->userRank, topoInfo->userRankSize, levels);
    }
    if (ret == HCCL_SUCCESS) {
        ret = ValidatePhysicalLevels(levels, topoInfo->userRank, topoInfo->userRankSize);
    }
    if (ret != HCCL_SUCCESS) {
        // 仅保留物理视图构建/表达限制的降级; 前面的结构校验错误已直接返回。
        HCCL_WARNING(
            "[PhysicalLevel][Build] normalize degraded, ret[%d], rank[%u]. physicalLevels stays empty, legacy path "
            "unaffected.",
            ret, topoInfo->userRank);
        return HCCL_SUCCESS;
    }

    topoInfo->physicalLevels = std::move(levels);
    const size_t levelNum = topoInfo->physicalLevels.size();
    HCCL_RUN_INFO("[PhysicalLevel][Build] rank[%u] built [%zu] physical levels", topoInfo->userRank, levelNum);
    // 最终产物逐级各打一行, 打的是真正落进topoInfo的内容(BuildLayerCandidates那条DEBUG打的是候选)。
    // RUN_INFO让默认日志级别下就搜得到, INFO让这几行与同批INFO落在同一条时间线上
    for (size_t idx = 0; idx < levelNum; ++idx) {
        const std::string desc = DescribeLevel(topoInfo->physicalLevels[idx], idx, levelNum);
        HCCL_RUN_INFO("[PhysicalLevel][Build] rank[%u] %s", topoInfo->userRank, desc.c_str());
        HCCL_INFO("[PhysicalLevel][Build] rank[%u] %s", topoInfo->userRank, desc.c_str());
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
