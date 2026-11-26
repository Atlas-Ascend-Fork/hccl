/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl_rt.h"
#include "alg_env_config.h"
#include "alg_param.h"
#include "gtest/gtest.h"
#include "hccl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_sym_win.h"
#include "hccl_verifier.h"
#include "op_common.h"
#include "sim_communicator.h"
#include "sim_world.h"

namespace {
using namespace HcclSim;
using namespace ops_hccl;

constexpr u32 MAX_TEST_RANKS = 8;
constexpr char MULTILINK_ALGORITHM[] = "AicpuAllReduceSoleNHRMultiLink";
constexpr char SPECIAL_ALGORITHM[] = "AicpuAllReduceSoleNHRAicpuReduce";
constexpr int64_t DAYU_BOARD_ID = 0x48;
constexpr int64_t OTHER_BOARD_ID = 0x44;

struct Window {
    bool registered = false;
    u32 rank = 0;
    ::BufferType bufferType = ::BufferType::INPUT;
    uintptr_t base = 0;
    uint64_t size = 0;
    u32 peerQueries = 0;
};

struct LaunchObservation {
    std::string algorithm;
    CommEngine engine = CommEngine::COMM_ENGINE_RESERVED;
    bool symmetric = false;
    bool reused = false;
    unsigned int result = std::numeric_limits<unsigned int>::max();
    u32 launches = 0;
    u32 channels = 0;
    u32 threads = 0;
    u32 topologyLevels = 0;
    Level0Shape topologyShape = Level0Shape::MESH_1D;
    bool uniformChannels = false;
    bool deviceUbChannels = false;
    bool windowsChecked = false;
};

std::array<std::array<Window, 2>, MAX_TEST_RANKS> windows;
std::array<LaunchObservation, MAX_TEST_RANKS> observations;
u32 linksPerPeer = 1;
int64_t boardId = DAYU_BOARD_ID;
bool boardQueryFails = false;
std::array<u32, MAX_TEST_RANKS> windowQueries{};

bool simulateCcuFallback = false;
struct FallbackObservation {
    u32 resourceFailures = 0;
    u32 cacheCreates = 0;
    u32 cacheHits = 0;
};
std::array<FallbackObservation, MAX_TEST_RANKS> fallbackObservations;
std::array<std::map<std::string, std::vector<char>>, MAX_TEST_RANKS> fallbackContexts;

HcclResult ModelConfigGetInfo(HcclComm comm, HcclConfigType type, uint32_t length, void* info)
{
    if (comm == nullptr || info == nullptr) {
        return HCCL_E_PARA;
    }
    if (static_cast<int>(type) == HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM && length == sizeof(u32)) {
        *static_cast<u32*>(info) = linksPerPeer;
        return HCCL_SUCCESS;
    }
    if (type == HCCL_CONFIG_TYPE_OP_EXPANSION_MODE && length == sizeof(HcclOpExpansionMode)) {
        *static_cast<HcclOpExpansionMode*>(info) = simulateCcuFallback ?
                                                       HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_CCU_SCHED :
                                                       HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AI_CPU;
        return HCCL_SUCCESS;
    }
    return HCCL_E_NOT_SUPPORT;
}

struct TestCase {
    u32 ranks = 4;
    u32 servers = 1;
    u32 channels = 2;
    u64 count = 1025;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HCCL_REDUCE_SUM;
    bool registerInput = true;
    bool registerOutput = true;
    bool expectSymmetric = true;
    const char* algorithm = MULTILINK_ALGORITHM;
    u64 cclBytes = 200 * 1024 * 1024;
    u32 repetitions = 1;
    int64_t board = DAYU_BOARD_ID;
    bool failBoardQuery = false;
    bool startWithCcu = false;
    bool checkAllChannelsCarryData = false;
    bool changeOutputRegistration = false;
    // Negative selects a distinct output allocation.
    int outputInInputArena = -1;
};

class ScopedEnvironment {
public:
    ScopedEnvironment()
    {
        for (const char* name :
             {"ST_HCOMM_VERSION", "HCCL_OP_EXPANSION_MODE", "HCCL_INDEPENDENT_OP", "HCCL_ALGO", "HCCL_DETERMINISTIC",
              "ENABLE_HOSTDPU_FOR_LLT", "HCCL_ENABLE_OPEN_AICPU"}) {
            const char* value = std::getenv(name);
            saved_.push_back({name, value == nullptr ? "" : value, value != nullptr});
            unsetenv(name);
        }
        setenv("HCCL_OP_EXPANSION_MODE", simulateCcuFallback ? "CCU_SCHED" : "AI_CPU", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
        setenv("ST_HCOMM_VERSION", std::to_string(CANN_VERSION_NUM).c_str(), 1);
        ResetAlgEnvConfigInitState();
    }
    ~ScopedEnvironment()
    {
        for (const auto& value : saved_) {
            if (value.present) {
                setenv(value.name, value.value.c_str(), 1);
            } else {
                unsetenv(value.name);
            }
        }
        ResetAlgEnvConfigInitState();
    }

private:
    struct SavedValue {
        const char* name;
        std::string value;
        bool present;
    };
    std::vector<SavedValue> saved_;
};

struct RankResources {
    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
    void* input = nullptr;
    void* output = nullptr;
    void* receiveBuffer = nullptr;
    std::array<HcclCommSymWindow, 2> registeredWindows{};

    ~RankResources()
    {
        for (auto window : registeredWindows) {
            if (window != nullptr) {
                EXPECT_EQ(HcclCommSymWinDeregister(window), HCCL_SUCCESS);
            }
        }
        if (comm != nullptr) {
            EXPECT_EQ(HcclCommDestroy(comm), HCCL_SUCCESS);
        }
        if (input != nullptr) {
            EXPECT_EQ(aclrtFree(input), ACL_SUCCESS);
        }
        if (output != nullptr) {
            EXPECT_EQ(aclrtFree(output), ACL_SUCCESS);
        }
        if (stream != nullptr) {
            EXPECT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
        }
    }
};

struct ScopedWorld {
    ~ScopedWorld() { SimWorld::Global()->Deinit(); }
};

HcclResult ResolvePeer(Window* window, size_t offset, uint32_t peerRank, void** ptr)
{
    if (window == nullptr || ptr == nullptr || !window->registered || offset >= window->size
        || peerRank >= MAX_TEST_RANKS) {
        return HCCL_E_PARA;
    }
    const Window& peerWindow = windows[peerRank][window->bufferType == ::BufferType::INPUT ? 0 : 1];
    if (!peerWindow.registered || peerWindow.size != window->size) {
        return HCCL_E_PARA;
    }
    ++window->peerQueries;
    *ptr = reinterpret_cast<void*>(peerWindow.base + offset);
    return HCCL_SUCCESS;
}

void CheckDataPath(bool symmetric, u32 expectedDataQueues, const SingleTaskQueue& queues)
{
    u32 cclCopies = 0;
    u32 dataQueues = 0;
    for (const auto& queue : queues) {
        bool carriesData = false;
        for (const auto& task : queue) {
            if (task->GetType() == TaskTypeStub::WRITE_REDUCE) {
                carriesData |= static_cast<TaskStubWriteReduce*>(task.get())->GetLocalSlice().GetSize() > 0;
            } else if (task->GetType() == TaskTypeStub::READ) {
                carriesData |= static_cast<TaskStubRead*>(task.get())->GetLocalSlice().GetSize() > 0;
            }
            if (task->GetType() != TaskTypeStub::LOCAL_COPY) {
                continue;
            }
            auto* copy = static_cast<TaskStubLocalCopy*>(task.get());
            if (!copy->IsGenFromSync()
                && (copy->GetSrcSlice().GetType() == ::BufferType::CCL
                    || copy->GetDstSlice().GetType() == ::BufferType::CCL)) {
                ++cclCopies;
            }
        }
        dataQueues += carriesData;
    }
    EXPECT_GE(dataQueues, expectedDataQueues);
    if (symmetric) {
        EXPECT_EQ(cclCopies, 0U);
    } else {
        EXPECT_GT(cclCopies, 0U);
    }
}

void RunCase(const TestCase& test)
{
    ASSERT_LE(test.ranks, MAX_TEST_RANKS);
    simulateCcuFallback = test.startWithCcu;
    fallbackContexts = {};
    boardId = test.board;
    boardQueryFails = test.failBoardQuery;
    ScopedEnvironment environment;
    std::unique_ptr<void, decltype(&dlclose)> symbols(dlopen(nullptr, RTLD_NOW), dlclose);
    ASSERT_NE(symbols.get(), nullptr);
    HcclSymWinDlInit(symbols.get());
    ASSERT_TRUE(HcommIsSupportHcclSymWinGetRemoteAddr());
    ASSERT_FALSE(HcommIsSupportHcclSymWinGetPeerPointer());
    linksPerPeer = test.channels;
    windows = {};
    observations = {};
    ASSERT_GT(test.servers, 0U);
    ASSERT_EQ(test.ranks % test.servers, 0U);
    const u32 ranksPerServer = test.ranks / test.servers;
    TopoMeta topology(1, SuperPodMeta(test.servers));
    for (u32 rank = 0; rank < test.ranks; ++rank) {
        topology[0][rank / ranksPerServer].push_back(rank % ranksPerServer);
    }
    SimWorld::Global()->Init(topology, HcclDevType::DEV_TYPE_950);
    ScopedWorld world;
    std::vector<std::unique_ptr<RankResources>> resources;
    const u64 bytes = test.count * ops_hccl::DATATYPE_SIZE_TABLE[test.dataType];
    const bool sharedWindow = test.outputInInputArena >= 0;
    const u64 inputBytes = sharedWindow ? bytes + test.outputInInputArena : bytes;
    for (u32 rank = 0; rank < test.ranks; ++rank) {
        SCOPED_TRACE(testing::Message() << "prepare rank=" << rank);
        ASSERT_EQ(aclrtSetDevice(rank), ACL_SUCCESS);
        SimWorld::Global()->GetSimNpuByRankId(rank).AllocMemory(::BufferType::CCL, test.cclBytes);
        resources.emplace_back(std::make_unique<RankResources>());
        auto& resource = *resources.back();
        ASSERT_EQ(aclrtCreateStream(&resource.stream), ACL_SUCCESS);
        ASSERT_EQ(HcclCommInitClusterInfo("./ranktable.json", rank, &resource.comm), HCCL_SUCCESS);
        ASSERT_EQ(
            aclrtMalloc(&resource.input, inputBytes, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK)),
            ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMalloc(&resource.output, bytes, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK)), ACL_SUCCESS);
        resource.receiveBuffer
            = sharedWindow ? static_cast<char*>(resource.input) + test.outputInInputArena : resource.output;
        if (test.registerInput) {
            ASSERT_EQ(
                HcclCommSymWinRegister(resource.comm, resource.input, inputBytes, &resource.registeredWindows[0], 1),
                HCCL_SUCCESS);
        }
        if (test.registerOutput && !sharedWindow) {
            ASSERT_EQ(
                HcclCommSymWinRegister(resource.comm, resource.output, bytes, &resource.registeredWindows[1], 1),
                HCCL_SUCCESS);
        }
    }
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    ASSERT_EQ(InitEnvConfig(), HCCL_SUCCESS);

    for (u32 repetition = 0; repetition < test.repetitions; ++repetition) {
        SCOPED_TRACE(testing::Message() << "repetition=" << repetition);
        SimTaskQueue::Global()->Clear();
        observations = {};
        windowQueries = {};
        fallbackObservations = {};
        bool outputRegistered = test.registerOutput;
        if (repetition > 0 && test.changeOutputRegistration) {
            outputRegistered = !outputRegistered;
            for (auto& resource : resources) {
                auto& handle = resource->registeredWindows[1];
                if (outputRegistered) {
                    ASSERT_EQ(
                        HcclCommSymWinRegister(resource->comm, resource->output, bytes, &handle, 1), HCCL_SUCCESS);
                } else {
                    ASSERT_EQ(HcclCommSymWinDeregister(handle), HCCL_SUCCESS);
                    handle = nullptr;
                }
            }
        }
        const bool symmetric = test.changeOutputRegistration ? outputRegistered : test.expectSymmetric;
        for (auto& rankWindows : windows) {
            for (auto& window : rankWindows) {
                window.peerQueries = 0;
            }
        }
        std::vector<HcclResult> results(test.ranks, HCCL_E_INTERNAL);
        std::vector<std::string> errors(test.ranks);
        std::vector<std::thread> workers;
        for (u32 rank = 0; rank < test.ranks; ++rank) {
            workers.emplace_back([&, rank]() {
                try {
                    if (aclrtSetDevice(rank) != ACL_SUCCESS) {
                        return;
                    }
                    auto& resource = *resources[rank];
                    results[rank] = HcclAllReduce(
                        resource.input, resource.receiveBuffer, test.count, test.dataType, test.reduceOp, resource.comm,
                        resource.stream);
                } catch (const std::exception& error) {
                    errors[rank] = error.what();
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        auto& taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
        for (u32 rank = 0; rank < test.ranks; ++rank) {
            SCOPED_TRACE(testing::Message() << "rank=" << rank << " " << errors[rank]);
            ASSERT_EQ(results[rank], HCCL_SUCCESS);
            const auto& observed = observations[rank];
            ASSERT_EQ(observed.launches, 1U);
            EXPECT_EQ(observed.algorithm, test.algorithm);
            EXPECT_EQ(observed.engine, CommEngine::COMM_ENGINE_AICPU_TS);
            EXPECT_EQ(observed.symmetric, symmetric);
            EXPECT_EQ(windowQueries[rank], test.registerInput ? 2U : 1U);
            EXPECT_TRUE(observed.windowsChecked);
            ASSERT_EQ(observed.result, 0U);
            EXPECT_EQ(observed.topologyLevels, test.servers == 1 ? 1U : 2U);
            EXPECT_EQ(observed.topologyShape, Level0Shape::CLOS);
            if (test.algorithm == std::string(MULTILINK_ALGORITHM)) {
                EXPECT_TRUE(observed.uniformChannels);
                EXPECT_TRUE(observed.deviceUbChannels);
                EXPECT_EQ(observed.channels, test.channels);
                EXPECT_GE(observed.threads, test.channels);
            }
            if (repetition != 0) {
                EXPECT_TRUE(observed.reused);
            }
            if (test.startWithCcu) {
                EXPECT_EQ(fallbackObservations[rank].resourceFailures, repetition == 0 ? 1U : 0U);
                EXPECT_EQ(fallbackObservations[rank].cacheCreates, repetition == 0 ? 1U : 0U);
                EXPECT_EQ(fallbackObservations[rank].cacheHits, repetition == 0 ? 0U : 1U);
            }
            EXPECT_EQ(windows[rank][0].peerQueries > 0, symmetric);
            EXPECT_EQ(windows[rank][1].peerQueries > 0, symmetric && !sharedWindow);
            const auto pos = SimWorld::Global()->GetNpuPosByRankId(rank);
            ASSERT_EQ(taskQueues.count(pos.superpodId), 1U);
            const auto& podQueues = taskQueues.at(pos.superpodId);
            ASSERT_EQ(podQueues.count(pos.serverId), 1U);
            const auto& serverQueues = podQueues.at(pos.serverId);
            ASSERT_EQ(serverQueues.count(pos.phyId), 1U);
            CheckDataPath(symmetric, test.checkAllChannelsCarryData ? test.channels : 0U, serverQueues.at(pos.phyId));
            if (sharedWindow) {
                // Read back the actual receive range on the caller stream into the verifier's OUTPUT buffer.
                // The AllReduce tasks, their addresses and dependencies stay intact.
                const auto& resource = *resources[rank];
                auto& npu = SimWorld::Global()->GetSimNpuByRankId(rank);
                HcclSim::DataSlice src, dst;
                ASSERT_EQ(npu.GetSlice(reinterpret_cast<uintptr_t>(resource.receiveBuffer), bytes, src), HCCL_SUCCESS);
                ASSERT_EQ(npu.GetSlice(reinterpret_cast<uintptr_t>(resource.output), bytes, dst), HCCL_SUCCESS);
                SimTaskQueue::Global()->AppendTask(
                    pos, static_cast<SimStream*>(resource.stream), std::make_shared<TaskStubLocalCopy>(src, dst));
            }
        }
        ASSERT_EQ(CheckAllReduce(taskQueues, test.ranks, test.dataType, test.count, test.reduceOp), HCCL_SUCCESS);
    }
}
} // namespace

// These model overrides are exported only by st_all_reduce_symmetric_memory_test.
// Each server has a CLOS at layer 0. The model supplies the ranks, links and upper layer.
extern "C" HcclResult
HcclRankGraphGetTopoType(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType)
{
    if (comm == nullptr || topoType == nullptr || topoInstId != 0 || netLayer > 1) {
        return HCCL_E_PARA;
    }
    if (netLayer == 1) {
        auto* model = static_cast<HcclSim::SimCommunicator*>(comm);
        model->topoModel_->GetTopoType(model->GetRankId(), netLayer, topoInstId, topoType);
        return HCCL_SUCCESS;
    }
    *topoType = CommTopo::COMM_TOPO_CLOS;
    return HCCL_SUCCESS;
}

extern "C" aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t* value)
{
    if (attr == ACL_DEV_ATTR_MAINBOARD_ID) {
        if (boardQueryFails || value == nullptr) {
            return ACL_ERROR_INVALID_PARAM;
        }
        *value = boardId;
        return ACL_SUCCESS;
    }
    static auto next = reinterpret_cast<decltype(&aclrtGetDeviceInfo)>(dlsym(RTLD_NEXT, "aclrtGetDeviceInfo"));
    return next == nullptr ? ACL_ERROR_INVALID_PARAM : next(deviceId, attr, value);
}

extern "C" void* HcclDlsym(void* handle, const char* name)
{
    if (name == nullptr) {
        return nullptr;
    }
    if (std::strcmp(name, "HcclConfigGetInfo") == 0) {
        return reinterpret_cast<void*>(&ModelConfigGetInfo);
    }
    if (std::strcmp(name, "HcclSymWinGetRemoteAddr") == 0) {
        return reinterpret_cast<void*>(&HcclSymWinGetRemoteAddr);
    }
    if (std::strcmp(name, "HcclSymWinGetPeerPointer") == 0) {
        return nullptr;
    }
    return dlsym(handle, name);
}

extern "C" HcclResult
HcclCommSymWinRegister(HcclComm comm, void* addr, uint64_t size, HcclCommSymWindow* handle, uint32_t flag)
{
    if (comm == nullptr || addr == nullptr || size == 0 || handle == nullptr || flag != 1) {
        return HCCL_E_PARA;
    }
    const u32 rank = static_cast<HcclSim::SimCommunicator*>(comm)->GetRankId();
    if (rank >= MAX_TEST_RANKS) {
        return HCCL_E_PARA;
    }
    for (u32 side = 0; side < 2; ++side) {
        const auto type = side == 0 ? ::BufferType::INPUT : ::BufferType::OUTPUT;
        auto block = HcclSim::SimWorld::Global()->GetSimNpuByRankId(rank).GetMemBlock(type);
        if (reinterpret_cast<uintptr_t>(addr) == block.startAddr && size <= block.size) {
            auto& window = windows[rank][side];
            window = {true, rank, type, reinterpret_cast<uintptr_t>(addr), size, 0};
            *handle = &window;
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_PARA;
}

extern "C" HcclResult HcclCommSymWinDeregister(HcclCommSymWindow handle)
{
    if (handle == nullptr) {
        return HCCL_E_PARA;
    }
    static_cast<Window*>(handle)->registered = false;
    return HCCL_SUCCESS;
}

extern "C" HcclResult
HcclCommSymWinGet(HcclComm comm, void* ptr, size_t size, HcclCommSymWindow* handle, size_t* offset)
{
    if (comm == nullptr || ptr == nullptr || handle == nullptr || offset == nullptr) {
        return HCCL_E_PARA;
    }
    *handle = nullptr;
    *offset = 0;
    const u32 rank = static_cast<HcclSim::SimCommunicator*>(comm)->GetRankId();
    if (rank >= MAX_TEST_RANKS) {
        return HCCL_E_PARA;
    }
    ++windowQueries[rank];
    const auto address = reinterpret_cast<uintptr_t>(ptr);
    for (auto& window : windows[rank]) {
        if (window.registered && address >= window.base && address - window.base < window.size
            && size <= window.size - (address - window.base)) {
            *handle = &window;
            *offset = address - window.base;
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_NOT_FOUND;
}

// Fail only CCU resource allocation; production FallbackOp and its context cache remain in control.
extern "C" HcclResult HcclThreadAcquireWithStream(
    HcclComm comm, CommEngine engine, aclrtStream stream, uint32_t notifyNum, ThreadHandle* thread)
{
    auto* model = static_cast<HcclSim::SimCommunicator*>(comm);
    if (model == nullptr || thread == nullptr) {
        return HCCL_E_PARA;
    }
    if (simulateCcuFallback && engine == CommEngine::COMM_ENGINE_CCU) {
        auto& observed = fallbackObservations.at(model->GetRankId());
        ++observed.resourceFailures;
        return HCCL_E_UNAVAIL;
    }
    return model->independentOpThreadMgr_->HcclThreadAcquireWithStream(engine, stream, notifyNum, thread);
}

extern "C" bool HcommIsSupportHcclChannelDestroy() { return !simulateCcuFallback; }

extern "C" HcclResult
HcclEngineCtxCreate(HcclComm comm, const char* tag, CommEngine engine, uint64_t size, void** context)
{
    auto* model = static_cast<HcclSim::SimCommunicator*>(comm);
    if (model == nullptr || tag == nullptr || context == nullptr) {
        return HCCL_E_PARA;
    }
    if (simulateCcuFallback && engine == CommEngine::COMM_ENGINE_CCU) {
        auto& contexts = fallbackContexts.at(model->GetRankId());
        auto result = contexts.emplace(tag, std::vector<char>(size));
        if (!result.second) {
            return HCCL_E_PARA;
        }
        ++fallbackObservations.at(model->GetRankId()).cacheCreates;
        *context = result.first->second.data();
        return HCCL_SUCCESS;
    }
    return model->contextManager_->CreateCommEngineCtx(tag, engine, size, context);
}

extern "C" HcclResult
HcclEngineCtxGet(HcclComm comm, const char* tag, CommEngine engine, void** context, uint64_t* size)
{
    auto* model = static_cast<HcclSim::SimCommunicator*>(comm);
    if (model == nullptr || tag == nullptr || context == nullptr || size == nullptr) {
        return HCCL_E_PARA;
    }
    if (simulateCcuFallback && engine == CommEngine::COMM_ENGINE_CCU) {
        auto& contexts = fallbackContexts.at(model->GetRankId());
        const auto found = contexts.find(tag);
        if (found == contexts.end()) {
            return HCCL_E_NOT_FOUND;
        }
        ++fallbackObservations.at(model->GetRankId()).cacheHits;
        *context = found->second.data();
        *size = found->second.size();
        return HCCL_SUCCESS;
    }
    return model->contextManager_->GetCommEngineCtx(tag, engine, context, size);
}

extern "C" HcclResult HcclSymWinGetRemoteAddr(HcclCommSymWindow handle, size_t offset, uint32_t peerRank, void** ptr)
{
    return ResolvePeer(static_cast<Window*>(handle), offset, peerRank, ptr);
}

extern "C" unsigned int HcclLaunchAicpuKernel(ops_hccl::OpParam* param)
{
    using Launch = unsigned int (*)(ops_hccl::OpParam*);
    static auto launch = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, "HcclLaunchAicpuKernel"));
    if (launch == nullptr || param == nullptr || param->hcclComm == nullptr) {
        return 1;
    }
    const u32 rank = static_cast<HcclSim::SimCommunicator*>(param->hcclComm)->GetRankId();
    if (rank >= MAX_TEST_RANKS) {
        return 1;
    }
    auto& observed = observations[rank];
    ++observed.launches;
    observed.algorithm = param->algName;
    observed.engine = param->engine;
    observed.symmetric = param->supportSymmetricMemory;
    observed.reused = param->cacheValid;
    observed.windowsChecked = param->symMemChecked;
    if (param->resCtx == nullptr || param->ctxSize == 0) {
        return 1;
    }
    const char* context = static_cast<const char*>(param->resCtx);
    std::vector<char> serialized(context, context + param->ctxSize);
    AlgResourceCtxSerializable resources;
    resources.DeSerialize(serialized);
    if (!resources.deserializeValid) {
        return 1;
    }
    observed.topologyLevels = resources.topoInfo.topoLevelNums;
    observed.topologyShape = resources.topoInfo.level0Topo;
    observed.threads = resources.threads.size();
    if (resources.channels.size() == 1 && !resources.channels[0].empty()) {
        std::map<u32, u32> channels;
        observed.deviceUbChannels = true;
        for (const auto& channel : resources.channels[0]) {
            ++channels[channel.remoteRank];
            observed.deviceUbChannels = observed.deviceUbChannels && channel.isValid && channel.handle != 0
                                        && channel.protocol == CommProtocol::COMM_PROTOCOL_UB_CTP
                                        && channel.locationType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
        }
        observed.channels = channels.begin()->second;
        observed.uniformChannels = true;
        for (const auto& peer : channels) {
            observed.uniformChannels = observed.uniformChannels && peer.second == observed.channels;
        }
    }
    observed.result = launch(param);
    return observed.result;
}

class ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST : public testing::Test {};

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, AutomaticallySelectsMultiLinkAndReducesAllRanks)
{
    TestCase test;
    test.ranks = 8;
    test.count = 1024;
    RunCase(test);
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, NonPowerOfTwoRanksPreserveTheTail)
{
    TestCase test;
    test.ranks = 3;
    test.count = 1031;
    RunCase(test);
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, TwoServersInOnePodSelectOrdinaryMultiLinkWithRegisteredWindows)
{
    TestCase test;
    test.servers = 2;
    test.expectSymmetric = false;
    RunCase(test);
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, SingleElementWithMoreChannelsThanElements)
{
    TestCase test;
    test.channels = 4;
    test.count = 1;
    RunCase(test);
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, ThreeChannelsReduceIntegerMinWithUnevenSlices)
{
    TestCase test;
    test.channels = 3;
    test.checkAllChannelsCarryData = true;
    test.count = 32771;
    test.dataType = HCCL_DATA_TYPE_INT32;
    test.reduceOp = HCCL_REDUCE_MIN;
    RunCase(test);
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, HalfPrecisionMaxPreservesTheTail)
{
    TestCase test;
    test.count = 1027;
    test.dataType = HCCL_DATA_TYPE_FP16;
    test.reduceOp = HCCL_REDUCE_MAX;
    RunCase(test);
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, UnregisteredBuffersUseTheOrdinaryMultiLinkPath)
{
    TestCase test;
    test.registerInput = false;
    test.registerOutput = false;
    test.expectSymmetric = false;
    RunCase(test);
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, InputWindowAloneDoesNotEnableSymmetricMemory)
{
    TestCase test;
    test.registerOutput = false;
    test.expectSymmetric = false;
    RunCase(test);
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, SpecialDataTypesUseAicpuReduceWithoutSymmetricMemory)
{
    for (auto dataType : {HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64}) {
        SCOPED_TRACE(testing::Message() << "dataType=" << dataType);
        TestCase test;
        test.dataType = dataType;
        test.expectSymmetric = false;
        test.algorithm = SPECIAL_ALGORITHM;
        ASSERT_NO_FATAL_FAILURE(RunCase(test));
    }
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, ProdUsesAicpuReduceWithoutSymmetricMemory)
{
    TestCase test;
    test.reduceOp = HCCL_REDUCE_PROD;
    test.expectSymmetric = false;
    test.algorithm = SPECIAL_ALGORITHM;
    RunCase(test);
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, SymmetricInputLargerThanCclBufferCompletes)
{
    TestCase test;
    test.count = 65537;
    test.cclBytes = 4096;
    RunCase(test);
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, OtherMainboardKeepsRegisteredClosOnOrdinaryMultiLink)
{
    TestCase test;
    test.board = OTHER_BOARD_ID;
    test.expectSymmetric = false;
    RunCase(test);
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, MainboardQueryFailureKeepsRegisteredClosOnOrdinaryMultiLink)
{
    TestCase test;
    test.failBoardQuery = true;
    test.expectSymmetric = false;
    RunCase(test);
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, ReusedResourcesFollowCurrentOutputWindow)
{
    for (bool initiallyRegistered : {false, true}) {
        SCOPED_TRACE(initiallyRegistered);
        TestCase test;
        test.registerOutput = initiallyRegistered;
        test.changeOutputRegistration = true;
        test.repetitions = 2;
        ASSERT_NO_FATAL_FAILURE(RunCase(test));
    }
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, RegisteredInplaceStaysSymmetricAndPreservesAllRankContributions)
{
    TestCase test;
    test.outputInInputArena = 0;
    test.expectSymmetric = true;
    RunCase(test);
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, OutputWindowAloneDoesNotEnableSymmetricMemory)
{
    TestCase test;
    test.registerInput = false;
    test.expectSymmetric = false;
    RunCase(test);
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, AdjacentOperandsInOneWindowRemainSymmetric)
{
    TestCase test;
    test.outputInInputArena = static_cast<int>(test.count * sizeof(float));
    RunCase(test);
}

TEST_F(ST_ALL_REDUCE_SYMMETRIC_MEMORY_TEST, CcuResourceFailureAndCachedFallbackReevaluateDayuEligibility)
{
    for (bool dayu : {true, false}) {
        SCOPED_TRACE(dayu);
        TestCase test;
        test.startWithCcu = true;
        test.board = dayu ? DAYU_BOARD_ID : OTHER_BOARD_ID;
        test.expectSymmetric = dayu;
        test.repetitions = 2;
        ASSERT_NO_FATAL_FAILURE(RunCase(test));
    }
}
