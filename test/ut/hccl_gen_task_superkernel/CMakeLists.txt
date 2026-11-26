# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 仅验证 HCCL 自身逻辑，GE 外部符号由 ge_dependencies_stub.cc 提供，不链接真实 GE 库。
# 配置阶段直接编译实际桩源，并检查 UT/生产头使用的 GE 声明，避免探针与真实编译面漂移。
set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(_hccl_sk_probe_includes
    ${_repo_root}/src/common
    ${_repo_root}/src/common/op_graph
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
    ${ASCEND_CANN_PACKAGE_PATH}/include/external
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
)

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)
cmake_push_check_state(RESET)
set(_hccl_sk_probe_saved_try_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_REQUIRED_INCLUDES ${_hccl_sk_probe_includes})
set(CMAKE_REQUIRED_FLAGS "-std=c++17")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

unset(HCCL_SUPERKERNEL_UT_DECLARATIONS_OK CACHE)
check_cxx_source_compiles([=[
#include "hccl_gen_task_superkernel.h"
#include "op_util.h"
#include "exe_graph/runtime/task_def_builder.h"
#include "graph/arg_desc_info.h"
#include "register/op_impl_registry.h"
int main()
{
    (void)&ge::ArgsFormatSerializer::Serialize;
    (void)&gert::ExeResGenerationContext::GetStreamId;
    (void)&gert::ExeResGenerationContext::GetStrAttrVal;
    (void)&gert::ExeResGenerationContext::GetIntAttrVal;
    (void)&gert::ExeResGenerationContext::SetStrAttrVal;
    return 0;
}
]=] HCCL_SUPERKERNEL_UT_DECLARATIONS_OK)

unset(HCCL_SUPERKERNEL_UT_STUB_COMPILES CACHE)
try_compile(HCCL_SUPERKERNEL_UT_STUB_COMPILES
    ${CMAKE_BINARY_DIR}/hccl_superkernel_ut_probe
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ge_dependencies_stub.cc
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${_hccl_sk_probe_includes}"
    COMPILE_DEFINITIONS "-std=c++17"
    OUTPUT_VARIABLE _hccl_sk_probe_output
)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_hccl_sk_probe_saved_try_type}")
cmake_pop_check_state()

if(NOT HCCL_SUPERKERNEL_UT_DECLARATIONS_OK OR NOT HCCL_SUPERKERNEL_UT_STUB_COMPILES)
    message(STATUS "[UT] hccl_gen_task_superkernel skipped: GE declarations incompatible with actual UT sources.")
    message(STATUS "${_hccl_sk_probe_output}")
    return()
endif()

if(COMMAND add_cann_third_party)
    add_cann_third_party(json)
endif()
if(NOT TARGET json)
    message(STATUS "[UT] hccl_gen_task_superkernel skipped: nlohmann/json third-party target not available.")
    return()
endif()

add_executable(hccl_gen_task_superkernel_test
    test_hccl_gen_task_superkernel.cc
    ge_dependencies_stub.cc
    ../op_graph_utils/stub/log_stub.cc
    ${_repo_root}/src/common/op_graph/hccl_gen_task_superkernel.cc
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

target_compile_options(hccl_gen_task_superkernel_test PRIVATE -U_FORTIFY_SOURCE)
target_link_libraries(hccl_gen_task_superkernel_test PRIVATE
    intf_llt_pub
    GTest::gtest_main
    -lpthread
    json
)

run_llt_test(TARGET hccl_gen_task_superkernel_test)
