/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <numeric>

#include "ins_temp_dpu_alltoall_mesh.h"
#include "nda_test_access.h"

namespace nda_ut {
namespace {

    class DpuAlltoAllGroupedMeshTest : public ::testing::Test {
    protected:
        void SetUp() override { ResetStubState(); }
        void TearDown() override { ResetStubState(); }
    };

    TEST_F(DpuAlltoAllGroupedMeshTest, ReusesSixteenThreadsAcrossTwoSynchronizedRounds)
    {
        constexpr u32 rankCount = 33;
        constexpr u32 threadCount = 16;
        constexpr u64 sliceBytes = 2;
        constexpr auto device = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
        const auto ranks = MakeRankIds(rankCount);
        const u32 myRank = ranks.front();

        OpParam op{};
        op.all2AllVDataDes.sendType = HCCL_DATA_TYPE_FP16;
        TopoInfoWithNetLayerDetails topo{};
        topo.userRank = myRank;
        topo.userRankSize = rankCount;
        topo.level0Topo = Level0Shape::MESH_1D;
        PhysicalLevelInfo pod{};
        pod.localRanks = ranks;
        pod.ref.netLayer = POD_NET_LAYER;
        pod.locType = device;
        pod.hasTopoInst = true;
        topo.physicalLevels.push_back(pod);

        InsTempDpuAlltoAllMesh mesh(op, myRank, {ranks});
        ASSERT_EQ(mesh.SetLocalPodRanks(topo), HCCL_SUCCESS);
        for (u32 peer = 1; peer < rankCount; ++peer) {
            HcclChannelDesc channel{};
            channel.remoteRank = peer;
            channel.remoteEndpoint.loc.locType = device;
            MeshChannelDescs().push_back(channel);
        }

        AlgResourceRequest request{};
        ASSERT_EQ(mesh.CalcRes(nullptr, op, &topo, request), HCCL_SUCCESS);
        ASSERT_EQ(request.slaveThreadNum, threadCount - 1);
        EXPECT_EQ(request.notifyNumOnMainThread, threadCount - 1);
        EXPECT_EQ(request.notifyNumPerThread, std::vector<u32>(threadCount - 1, 1));
        ASSERT_EQ(request.channels.size(), 1U);
        ASSERT_EQ(request.channels.front().size(), rankCount - 1);

        std::vector<uint8_t> input(rankCount * sliceBytes);
        std::vector<uint8_t> output(rankCount * sliceBytes);
        std::vector<uint8_t> cclBuffer(2 * rankCount * sliceBytes);
        std::vector<u64> counts(rankCount, 1);
        std::vector<u64> displacements(rankCount);
        std::iota(displacements.begin(), displacements.end(), 0);
        auto data = MakeParams(
            rankCount, counts, counts, displacements, displacements, input.data(), output.data(), cclBuffer.data(), 0);
        data.buffInfo.hcclBuff.size = cclBuffer.size();
        data.inputSliceStride = sliceBytes;

        TemplateResource resource{};
        resource.isDpuThreadEnabled = true;
        resource.threads.resize(request.slaveThreadNum + 1);
        std::iota(resource.threads.begin(), resource.threads.end(), 1);
        resource.dpuThread = resource.threads.back() + 1;
        resource.dpuThreadNotifyIdxOnMain = request.notifyNumOnMainThread;
        for (const auto& channel : request.channels.front()) {
            u32 peer = channel.remoteRank;
            resource.channels[peer] = {MakeChannel(
                peer, CommProtocol::COMM_PROTOCOL_RESERVED, device, peer, cclBuffer.data(), cclBuffer.size())};
        }

        ASSERT_EQ(mesh.KernelRun(op, data, resource), HCCL_SUCCESS);
        const auto& sync = GroupSyncCalls();
        const auto& writes = WriteCalls();
        ASSERT_EQ(sync.size(), 4U);
        ASSERT_EQ(writes.size(), rankCount - 1);

        const std::vector<ThreadHandle> subThreads(resource.threads.begin() + 1, resource.threads.end());
        const std::vector<u32> startIndices(subThreads.size(), 0);
        std::vector<u32> finishIndices(subThreads.size());
        std::iota(finishIndices.begin(), finishIndices.end(), 0);
        std::vector<u64> sentOffsets;
        for (u32 round = 0; round < 2; ++round) {
            SCOPED_TRACE(round);
            const auto& start = sync[2 * round];
            const auto& finish = sync[2 * round + 1];
            ASSERT_TRUE(start.beforeTransfer);
            ASSERT_FALSE(finish.beforeTransfer);
            ASSERT_EQ(start.writeCount, round * threadCount);
            ASSERT_EQ(finish.writeCount, (round + 1) * threadCount);
            EXPECT_EQ(start.mainThread, resource.threads.front());
            EXPECT_EQ(finish.mainThread, resource.threads.front());
            EXPECT_EQ(start.subThreads, subThreads);
            EXPECT_EQ(finish.subThreads, subThreads);
            EXPECT_EQ(start.notifyIndices, startIndices);
            EXPECT_EQ(finish.notifyIndices, finishIndices);

            for (u32 slot = 0; slot < threadCount; ++slot) {
                const auto& write = writes[start.writeCount + slot];
                EXPECT_EQ(write.kind, WriteKind::kSendRecvWrite);
                EXPECT_EQ(write.thread, resource.threads[slot]);
                ASSERT_EQ(write.txSrc.size(), 1U);
                sentOffsets.push_back(write.txSrc.front().offset_);
            }
        }
        std::sort(sentOffsets.begin(), sentOffsets.end());
        for (u32 peer = 1; peer < rankCount; ++peer) {
            EXPECT_EQ(sentOffsets[peer - 1], peer * sliceBytes);
        }
    }

} // namespace
} // namespace nda_ut
