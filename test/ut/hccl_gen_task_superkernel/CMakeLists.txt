# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 该 UT 依赖 GE 对外接口 TaskDefBuilder / kHcclSuperKernel，接口不存在时跳过。
# 注意：以下编译探测代码与 src/hccl.cmake 保持逐字一致，修改时请同步另一处。
include(CheckCXXSourceCompiles)
set(_hccl_sk_probe_saved_includes "${CMAKE_REQUIRED_INCLUDES}")
set(_hccl_sk_probe_saved_flags "${CMAKE_REQUIRED_FLAGS}")
set(_hccl_sk_probe_saved_try_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_REQUIRED_INCLUDES
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
)
set(CMAKE_REQUIRED_FLAGS "-std=c++17")
# 仅做编译探测，不参与链接，避免被 GE 库的传递依赖（protobuf/platform 等）干扰
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
unset(HCCL_HAS_GE_SUPERKERNEL_GENTASK CACHE)
check_cxx_source_compiles([=[
#include <vector>
#include "exe_graph/runtime/task_def_builder.h"
#include "graph/arg_desc_info.h"
int main()
{
    gert::TaskDefBuilder task;
    task.SetStreamId(1U).SetType(0U);
    task.GetKernelBuilder().SetBlockDim(1U).GetContextBuilder().SetArgsFormat(ge::AscendString("{i0*}"));
    std::vector<ge::ArgDescInfo> args;
    args.emplace_back(ge::ArgDescInfo(ge::ArgDescType::kIrInput, 0));
    args.insert(args.begin(), ge::ArgDescInfo::CreateHiddenInput(ge::HiddenInputSubType::kHcclSuperKernel));
    (void)ge::ArgsFormatSerializer::Serialize(args);
    return 0;
}
]=] HCCL_HAS_GE_SUPERKERNEL_GENTASK)
set(CMAKE_REQUIRED_INCLUDES "${_hccl_sk_probe_saved_includes}")
set(CMAKE_REQUIRED_FLAGS "${_hccl_sk_probe_saved_flags}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_hccl_sk_probe_saved_try_type}")

if(NOT HCCL_HAS_GE_SUPERKERNEL_GENTASK)
    message(STATUS "[UT] hccl_gen_task_superkernel skipped: GE TaskDefBuilder/kHcclSuperKernel not available.")
    return()
endif()

# 与 src/hccl.cmake 保持一致：cann-cmake 未提供 add_cann_third_party 或未创建 json 目标时跳过本 UT
if(COMMAND add_cann_third_party)
    add_cann_third_party(json)
endif()
if(NOT TARGET json)
    message(STATUS "[UT] hccl_gen_task_superkernel skipped: nlohmann/json third-party target not available.")
    return()
endif()

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

add_executable(hccl_gen_task_superkernel_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_hccl_gen_task_superkernel.cc
    ${_repo_root}/src/common/op_graph/hccl_gen_task_superkernel.cc
    ${_repo_root}/src/common/log.cc
)

target_include_directories(hccl_gen_task_superkernel_test PRIVATE
    ${_repo_root}/src/common
    ${_repo_root}/src/common/op_graph
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
    ${ASCEND_CANN_PACKAGE_PATH}/include/external
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
)

target_compile_options(hccl_gen_task_superkernel_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_link_directories(hccl_gen_task_superkernel_test PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(hccl_gen_task_superkernel_test PRIVATE
    intf_llt_pub
    GTest::gtest_main
    -lpthread
    ${CMAKE_DL_LIBS}
    json
    graph
    graph_base
    register
    opp_registry
    exe_graph
    platform
    ge_common
    unified_dlog
)

target_link_options(hccl_gen_task_superkernel_test PRIVATE
    "-Wl,-rpath-link,${ASCEND_CANN_PACKAGE_PATH}/lib64"
    "-Wl,-rpath,${ASCEND_CANN_PACKAGE_PATH}/lib64"
)

run_llt_test(TARGET hccl_gen_task_superkernel_test)
