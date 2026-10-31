# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_executable(reduce_scatter_selector_test reduce_scatter_selector_test.cc)

target_compile_options(reduce_scatter_selector_test PRIVATE
    -U_FORTIFY_SOURCE
)

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
target_include_directories(reduce_scatter_selector_test PRIVATE
    ${_repo_root}/src/ops/reduce_scatter/selector
    ${_repo_root}/src/ops/reduce_scatter/algorithm/template/ccu/kernel
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
)

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

target_include_directories(reduce_scatter_selector_test PRIVATE
    ${_cann_arch_dir}/include
    ${_cann_arch_dir}/include/hccl
    ${_cann_arch_dir}/include/hcomm/ccu
    ${_cann_arch_dir}/pkg_inc
    ${_cann_arch_dir}/pkg_inc/hccl
    ${_cann_arch_dir}/pkg_inc/base
    ${_cann_arch_dir}/pkg_inc/hcomm/ccu
)

set(_ut_cann_ver_header "${_cann_arch_dir}/include/version/cann_version.h")
if(EXISTS "${_ut_cann_ver_header}")
    file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line
         REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
    if(_ut_cann_ver_line)
        string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
            _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
        math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
        target_compile_definitions(reduce_scatter_selector_test
            PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
    else()
        message(FATAL_ERROR "[UT] Failed to parse CANN_VERSION_NUM from ${_ut_cann_ver_header}")
    endif()
endif()

target_link_libraries(reduce_scatter_selector_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET reduce_scatter_selector_test TASK_NUM 1)
