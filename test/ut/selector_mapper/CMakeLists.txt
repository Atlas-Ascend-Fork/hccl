# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# AlgoNameMapper 真实现 UT（非桩）——覆盖 Lookup 全映射/miss/Init 幂等/晚注册检测。
# 编入生产代码 algo_name_mapper.cc，桩仅覆盖 log.h 的日志函数。

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

set(_mapper_ut_include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${_repo_root}/src/ops/op_common/selector
    ${_repo_root}/src/ops/op_common/selector/pub_inc
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/ops/op_common
    ${_repo_root}/src/ops/op_common/topo_info
    ${_repo_root}/src/ops/op_common/algorithm/template
    ${_repo_root}/src/ops/op_common/algorithm/template/registry
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper
    ${_repo_root}/src/ops/op_common/algorithm/executor
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry
    ${_repo_root}/src/ops/op_common/algorithm/executor/channel
    ${_repo_root}/src/ops/op_common/algorithm/topo_match
    ${_repo_root}/src/common
    ${_repo_root}/src/common/tuner
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
)

add_executable(algo_name_mapper_test
    algo_name_mapper_test.cc
    ${_repo_root}/src/ops/op_common/selector/algo_name_mapper.cc
    stub/log_stub.cc
)

# CANN 头路径按架构自适应：hccl_res_expt.h 等仅存在于 <arch>/pkg_inc/hccl/，
# 写死 x86_64-linux 在 aarch64 环境必然缺失（对齐 reduce_scatter_selector UT 模式）。
if(NOT ASCEND_CANN_PACKAGE_PATH OR NOT IS_DIRECTORY "${ASCEND_CANN_PACKAGE_PATH}")
    message(FATAL_ERROR
        "[UT] ASCEND_CANN_PACKAGE_PATH is not set or does not exist: '${ASCEND_CANN_PACKAGE_PATH}'")
endif()

file(GLOB _cann_arch_dirs LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
if(NOT _cann_arch_dirs)
    message(FATAL_ERROR
        "[UT] No *-linux directory found under ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")
endif()

set(_cann_arch_dir "")
foreach(_dir ${_cann_arch_dirs})
    if(_dir MATCHES "/${CMAKE_SYSTEM_PROCESSOR}-linux$")
        set(_cann_arch_dir "${_dir}")
        break()
    endif()
endforeach()
if(NOT _cann_arch_dir)
    message(FATAL_ERROR
        "[UT] Cannot select CANN arch directory for CMAKE_SYSTEM_PROCESSOR='${CMAKE_SYSTEM_PROCESSOR}'. "
        "Candidates: ${_cann_arch_dirs}")
endif()

target_include_directories(algo_name_mapper_test PRIVATE
    ${_mapper_ut_include_dirs}
    ${_cann_arch_dir}/include
    ${_cann_arch_dir}/include/hccl
    ${_cann_arch_dir}/pkg_inc
    ${_cann_arch_dir}/pkg_inc/hccl
    ${_cann_arch_dir}/pkg_inc/base
    ${_cann_arch_dir}/pkg_inc/hcomm/ccu
)

# dlsym_common.h 的 HcclCommStatus/ThreadHandle 桩以 CANN_VERSION_NUM 守卫，UT 目标须与
# CANN 环境头一致传入该定义，否则 9.1+ 环境下桩与公开头冲突（同 selector_engine UT 模式）
set(_ut_cann_ver_header "${_cann_arch_dir}/include/version/cann_version.h")
if(EXISTS "${_ut_cann_ver_header}")
    file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line
         REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
    if(_ut_cann_ver_line)
        string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
            _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
        math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
        target_compile_definitions(algo_name_mapper_test PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
    else()
        message(FATAL_ERROR "[UT] Failed to parse CANN_VERSION_NUM from ${_ut_cann_ver_header}")
    endif()
endif()

target_link_libraries(algo_name_mapper_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET algo_name_mapper_test TASK_NUM 1)
