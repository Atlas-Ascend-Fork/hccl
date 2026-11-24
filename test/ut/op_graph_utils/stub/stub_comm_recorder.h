/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 通信入口记录桩的共享定义: op_layer_stub.cc 定义, 测试 TU include 本头 extern 引用 —
// 结构体单点维护, 字段增删两处同步编译期可见
#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct StubCommCall {
    const void* buf1;
    const void* buf2;
    uint64_t count;
    std::string tag;
};

extern std::vector<StubCommCall> g_stubArCalls;
extern std::vector<StubCommCall> g_stubRsCalls;
extern std::vector<StubCommCall> g_stubSendCalls;
extern std::vector<StubCommCall> g_stubBcastCalls;
extern std::vector<StubCommCall> g_stubReduceCalls;
extern std::vector<StubCommCall> g_stubRecvCalls;
