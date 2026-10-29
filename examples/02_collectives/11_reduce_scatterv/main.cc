/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if (ret != ACL_SUCCESS) {                                                              \
            printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                        \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(ret)                                                                          \
    do {                                                                                        \
        if (ret != HCCL_SUCCESS) {                                                              \
            printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                         \
        }                                                                                       \
    } while (0)

struct ThreadContext {
    HcclRootInfo* rootInfo;
    uint32_t device;
    uint32_t devCount;
};

// 每个 rank 从其他 rank 接收的数据量。CountOfRank(i) 表示编号为 i 的那段数据（即发送给 rank i 的数据）的元素个数。
// 与 ReduceScatter 不同，ReduceScatterV 允许同一 rank 内不同编号的数据大小不一致，此处刻意构造为 i + 1，
// 用于体现 V（Variable）语义。注意：所有 rank 的 sendCounts / sendDispls 必须一致。
static uint64_t CountOfRank(uint32_t rank) { return static_cast<uint64_t>(rank) + 1U; }

int Sample(void* arg)
{
    ThreadContext* ctx = (ThreadContext*)arg;
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint32_t device = ctx->device;
    uint32_t rankSize = ctx->devCount;
    uint64_t recvCount = CountOfRank(device); // 本 rank 接收的数据量（即编号为 device 的那段数据）

    // 计算本 rank 的 sendBuf 大小：向所有 rank 发送的数据量之和
    uint64_t totalSendCount = 0U;
    for (uint32_t i = 0; i < rankSize; ++i) {
        totalSendCount += CountOfRank(i);
    }
    size_t sendSize = totalSendCount * sizeof(float);
    size_t recvSize = recvCount * sizeof(float);

    // 设置当前线程操作的设备
    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));

    // 申请集合通信操作的 Device 内存
    ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY));

    // 申请 Host 内存用于存放输入数据，并将内容初始化为 Device ID
    void* hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, sendSize));
    float* tmpHostBuff = static_cast<float*>(hostBuf);
    for (uint64_t i = 0; i < totalSendCount; ++i) {
        tmpHostBuff[i] = static_cast<float>(device);
    }
    // 将 Host 侧输入数据拷贝到 Device 侧
    ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostBuf, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // 释放 Host 侧内存
    ACLCHECK(aclrtFreeHost(hostBuf));

    // 初始化集合通信域
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(rankSize, ctx->rootInfo, device, &hcclComm));

    // 创建任务流
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    // 设置 sendCounts 与 sendDispls，向各 rank 发送的数据量与其 recvCount 一致，偏移量按编号顺序累加
    std::vector<uint64_t> sendCounts(rankSize);
    std::vector<uint64_t> sendDispls(rankSize);
    uint64_t displ = 0U;
    for (uint32_t i = 0; i < rankSize; ++i) {
        sendCounts[i] = CountOfRank(i);
        sendDispls[i] = displ;
        displ += sendCounts[i];
    }

    // 执行 ReduceScatterV，将所有 rank 对应编号的数据相加后，再把结果按照编号分散到各个 rank 的 recvBuf
    HCCLCHECK(HcclReduceScatterV(
        sendBuf, sendCounts.data(), sendDispls.data(), recvBuf, recvCount, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
        hcclComm, stream));
    // 阻塞等待任务流中的集合通信任务执行完成
    ACLCHECK(aclrtSynchronizeStream(stream));

    // 将 Device 侧集合通信任务结果拷贝到 Host，并打印结果
    std::this_thread::sleep_for(std::chrono::seconds(device));
    void* resultBuff;
    ACLCHECK(aclrtMallocHost(&resultBuff, recvSize));
    ACLCHECK(aclrtMemcpy(resultBuff, recvSize, recvBuf, recvSize, ACL_MEMCPY_DEVICE_TO_HOST));
    float* tmpResBuff = static_cast<float*>(resultBuff);
    std::cout << "rankId: " << device << ", output: [";
    for (uint64_t i = 0; i < recvCount; ++i) {
        std::cout << " " << tmpResBuff[i];
    }
    std::cout << " ]" << std::endl;
    ACLCHECK(aclrtFreeHost(resultBuff));

    // 释放资源
    HCCLCHECK(HcclCommDestroy(hcclComm)); // 销毁通信域
    ACLCHECK(aclrtFree(sendBuf));         // 释放 Device 侧内存
    ACLCHECK(aclrtFree(recvBuf));         // 释放 Device 侧内存
    ACLCHECK(aclrtDestroyStream(stream)); // 销毁任务流
    return 0;
}

int main()
{
    // 设备资源初始化
    ACLCHECK(aclInit(NULL));
    // 查询设备数量
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    std::cout << "Found " << devCount << " NPU device(s) available" << std::endl;

    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    // 生成 Root 节点信息，各线程使用同一份 RootInfo
    void* rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo* rootInfo = (HcclRootInfo*)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    // 启动线程执行集合通信操作
    std::vector<std::thread> threads(devCount);
    std::vector<ThreadContext> args(devCount);
    for (uint32_t i = 0; i < devCount; i++) {
        args[i].rootInfo = rootInfo;
        args[i].device = i;
        args[i].devCount = devCount;
        threads[i] = std::thread(Sample, (void*)&args[i]);
    }
    for (uint32_t i = 0; i < devCount; i++) {
        threads[i].join();
    }

    // 释放资源
    ACLCHECK(aclrtFreeHost(rootInfoBuf)); // 释放 Host 内存
    ACLCHECK(aclFinalize());              // 设备去初始化
    return 0;
}
