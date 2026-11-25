/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef P2P_CHANNEL_THREADS_H
#define P2P_CHANNEL_THREADS_H

#include <cstring>
#include <vector>
#include "alg_param.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

// P2P Send/Recv 通用通道/线程装配（send/recv 两侧共用）

// 算子入口在 group 模式下将 tag 追加 "_Group" 后缀，executor 侧据此识别 group 算子（不进多通道路径）。
inline bool P2pIsGroupMode(const OpParam& param)
{
    const size_t tagLen = strnlen(param.tag, sizeof(param.tag));
    constexpr const char* GROUP_TAG_SUFFIX = "_Group";
    const size_t suffixLen = strlen(GROUP_TAG_SUFFIX);
    return (tagLen >= suffixLen) && (strncmp(param.tag + tagLen - suffixLen, GROUP_TAG_SUFFIX, suffixLen) == 0);
}

inline void
P2pSelectGroupChannel(const OpParam& param, u32 myRank, u32 remoteRank, bool isSend, std::vector<ChannelInfo>& channels)
{
    if (!P2pIsGroupMode(param) || channels.empty()) {
        return;
    }
    const bool lowerSide = (myRank <= remoteRank);
    const size_t selectedIdx = (channels.size() > 1) ? (isSend ? (lowerSide ? 0 : 1) : (lowerSide ? 1 : 0)) : 0;
    ChannelInfo selected = channels[selectedIdx];
    channels.clear();
    channels.push_back(selected);
}

inline HcclResult
P2pCollectChannelsByRank(const AlgResourceCtxSerializable& resCtx, u32 remoteRank, std::vector<ChannelInfo>& channels)
{
    channels.clear();
    for (const ChannelInfo& channel : resCtx.channels.at(0)) {
        if (channel.remoteRank == remoteRank) {
            channels.push_back(channel);
        }
    }
    CHK_PRT_RET(
        channels.empty(),
        HCCL_ERROR("[P2pCollectChannelsByRank] Channel[%d]-[%d] not found.", resCtx.topoInfo.userRank, remoteRank),
        HcclResult::HCCL_E_NOT_FOUND);
    return HCCL_SUCCESS;
}

inline HcclResult P2pBuildDataThreads(
    const AlgResourceCtxSerializable& resCtx, const std::vector<ChannelInfo>& channels, const ThreadHandle& mainThread,
    std::vector<ThreadHandle>& dataThreads)
{
    dataThreads.clear();
    u32 threadNum = static_cast<u32>(channels.size());
    CHK_PRT_RET(
        threadNum == 0 || resCtx.threads.size() < threadNum,
        HCCL_ERROR(
            "[P2pBuildDataThreads] threadNum[%u] is invalid or exceeds resCtx threads size[%zu].", threadNum,
            resCtx.threads.size()),
        HcclResult::HCCL_E_INTERNAL);
    dataThreads.push_back(mainThread);
    for (u32 i = 1; i < threadNum; i++) {
        dataThreads.push_back(resCtx.threads[i]);
    }
    if (threadNum > 1) {
        std::vector<ThreadHandle> subThreads(dataThreads.begin() + 1, dataThreads.end());
        std::vector<u32> notifyIdxMainToSub(threadNum - 1, 0);
        CHK_RET(PreSyncInterThreads(dataThreads[0], subThreads, notifyIdxMainToSub));
    }
    return HCCL_SUCCESS;
}

// 统一装配执行通道与线程：group 模式按方向规则收缩为单通道（与原有行为一致），非 group 全通道每通道独立线程。
// mainThread 为框架指定的主执行线程（OrchestrateWithThread 传 sendRecvThread），从线程取资源上下文。
inline HcclResult P2pAssembleChannelsAndThreads(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, u32 myRank, u32 remoteRank, bool isSend,
    const ThreadHandle& mainThread, std::vector<ChannelInfo>& channels, std::vector<ThreadHandle>& dataThreads)
{
    CHK_RET(P2pCollectChannelsByRank(resCtx, remoteRank, channels));
    P2pSelectGroupChannel(param, myRank, remoteRank, isSend, channels);
    CHK_RET(P2pBuildDataThreads(resCtx, channels, mainThread, dataThreads));
    for (u32 i = 0; i < channels.size(); i++) {
        HCCL_DEBUG(
            "[P2pAssembleChannelsAndThreads] remoteRank[%u] channelIdx[%u] portGroupSize[%u].", remoteRank, i,
            channels[i].portGroupSize);
    }
    return HCCL_SUCCESS;
}

inline HcclResult P2pFinishDataThreads(const std::vector<ThreadHandle>& dataThreads)
{
    u32 threadNum = static_cast<u32>(dataThreads.size());
    if (threadNum > 1) {
        std::vector<ThreadHandle> subThreads(dataThreads.begin() + 1, dataThreads.end());
        std::vector<u32> notifyIdxSubToMain(threadNum - 1);
        for (u32 i = 0; i < threadNum - 1; i++) {
            notifyIdxSubToMain[i] = i;
        }
        CHK_RET(PostSyncInterThreads(dataThreads[0], subThreads, notifyIdxSubToMain));
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl

#endif // P2P_CHANNEL_THREADS_H
