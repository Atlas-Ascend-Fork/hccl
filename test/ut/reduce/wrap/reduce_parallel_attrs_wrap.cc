/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// UT 专用包装编译单元（reduce_sequence_mesh_nhr_aicpu_reduce_test 目标独享）：
// 以直接 #include 生产源文件的方式，使 reduce_parallel_executor.cc 中的
// REGISTER_ALG_ATTRS 注册体（AicpuReduceParallelMeshNHR 等）在本测试进程中执行，
// UT 断言直接作用于生产注册项（非测试内复刻条件）。
//
// 同时将 REGISTER_EXECUTOR_BY_FOUR_TEMPS 展开置空：executor 注册会实例化
// ReduceParallelExecutor 及 AICPU/CCU 算法模板（vtable 隐式实例化拉入 CCU 设备侧
// 符号），与"仅验证 attrs 注册"的 UT 目标无关。置空后本 TU 不再实例化任何
// executor/template 代码，仅保留 attrs 注册闭包。
//
// 本包装仅存在于 test/ 目录，不构成对生产源文件的任何修改。

#include "coll_alg_v2_exec_registry.h"

#undef REGISTER_EXECUTOR_BY_FOUR_TEMPS
#define REGISTER_EXECUTOR_BY_FOUR_TEMPS( \
    type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3)

// 目标注册位于生产源 CANN_VERSION_NUM >= 9.0.0 版本门内，8.x 下契约用例由测试文件同名门跳过
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "reduce_parallel_executor.cc"
#endif
