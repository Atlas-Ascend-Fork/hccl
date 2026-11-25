# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

# CLOS 单层拓扑下 6 个算子的 CcuSched Mesh/NHR 选路 UT: 真实编译 auto selector 与 registry,
# dlsym/deterministic 等外部依赖由本目录桩提供, 不依赖硬件
set(clos_topo_selector_test_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/clos_topo_selector_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/log_stub.cc
    ${_repo_root}/src/ops/op_common/selector/auto_selector_base.cc
    ${_repo_root}/src/ops/op_common/selector/selector_registry.cc
    ${_repo_root}/src/ops/all_reduce/selector/all_reduce_auto_selector.cc
    ${_repo_root}/src/ops/all_gather/selector/all_gather_auto_selector.cc
    ${_repo_root}/src/ops/broadcast/selector/broadcast_auto_selector.cc
    ${_repo_root}/src/ops/reduce/selector/reduce_auto_selector.cc
    ${_repo_root}/src/ops/reduce_scatter/selector/reduce_scatter_auto_selector.cc
    ${_repo_root}/src/ops/scatter/selector/scatter_auto_selector.cc
)

add_executable(clos_topo_selector_test ${clos_topo_selector_test_src_list})

target_compile_options(clos_topo_selector_test PRIVATE
    -U_FORTIFY_SOURCE
)

# 六个算子的 auto selector 依赖本算子 algorithm/executor 与公共模板下的头文件
target_include_directories(clos_topo_selector_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${_repo_root}/src/ops/op_common
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/ops/op_common/selector
    ${_repo_root}/src/ops/op_common/topo_info
    ${_repo_root}/src/ops/op_common/algorithm/executor
    ${_repo_root}/src/ops/op_common/algorithm/executor/channel
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry
    ${_repo_root}/src/ops/op_common/algorithm/template
    ${_repo_root}/src/ops/op_common/algorithm/template/registry
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper
    ${_repo_root}/src/ops/op_common/algorithm/template/aiv
    ${_repo_root}/src/ops/op_common/algorithm/topo_match
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
    ${_repo_root}/src/ops/all_reduce/selector
    ${_repo_root}/src/ops/all_reduce/algorithm/executor
    ${_repo_root}/src/ops/all_gather/selector
    ${_repo_root}/src/ops/all_gather/algorithm/executor
    ${_repo_root}/src/ops/broadcast/selector
    ${_repo_root}/src/ops/broadcast/algorithm/executor
    ${_repo_root}/src/ops/reduce/selector
    ${_repo_root}/src/ops/reduce/algorithm/executor
    ${_repo_root}/src/ops/reduce_scatter/selector
    ${_repo_root}/src/ops/reduce_scatter/algorithm/executor
    ${_repo_root}/src/ops/scatter/selector
    ${_repo_root}/src/ops/scatter/algorithm/executor
)

file(GLOB _cann_arch_dirs LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
if(_cann_arch_dirs)
    set(_cann_arch_dir "")
    foreach(_dir ${_cann_arch_dirs})
        if(_dir MATCHES "/${CMAKE_SYSTEM_PROCESSOR}-linux$")
            set(_cann_arch_dir ${_dir})
            break()
        endif()
    endforeach()
    if(NOT _cann_arch_dir)
        list(GET _cann_arch_dirs 0 _cann_arch_dir)
    endif()
    target_include_directories(clos_topo_selector_test PRIVATE
        ${_cann_arch_dir}/include
        ${_cann_arch_dir}/include/hccl
        ${_cann_arch_dir}/include/acl
        ${_cann_arch_dir}/include/hcomm/ccu
        ${_cann_arch_dir}/pkg_inc
        ${_cann_arch_dir}/pkg_inc/hccl
        ${_cann_arch_dir}/pkg_inc/base
        ${_cann_arch_dir}/pkg_inc/hcomm/ccu
    )

    # 前向兼容：探测 CANN 版本号，与主构建保持一致。缺少该宏时 dlsym_common.h
    # 中的桩类型块会启用，与 CANN 公共头重复定义导致 conflicting declaration。
    set(_ut_cann_ver_header "${_cann_arch_dir}/include/version/cann_version.h")
    if(EXISTS "${_ut_cann_ver_header}")
        file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line
             REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
        if(_ut_cann_ver_line)
            string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
                _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
            math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
            target_compile_definitions(clos_topo_selector_test PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
            message(STATUS "[UT] clos_topo_selector Detected CANN_VERSION_NUM = ${_UT_CANN_VERSION_NUM}")
        else()
            message(FATAL_ERROR "[UT] Failed to parse CANN_VERSION_NUM from ${_ut_cann_ver_header}: "
                "no '#define CANN_VERSION_NUM' line found.")
        endif()
    else()
        message(WARNING "[UT] cann_version.h not found at ${_ut_cann_ver_header}, CANN_VERSION_NUM left undefined.")
    endif()
else()
    message(FATAL_ERROR "[UT] No CANN arch directory (*-linux) found under: ${ASCEND_CANN_PACKAGE_PATH}")
endif()

find_library(UT_CSEC_LIB NAMES c_sec
    PATHS ${ASCEND_CANN_PACKAGE_PATH}/lib64 ${_cann_arch_dir}/lib64
    NO_DEFAULT_PATH
)
if(NOT UT_CSEC_LIB)
    message(FATAL_ERROR "[UT] libc_sec not found under CANN package: ${ASCEND_CANN_PACKAGE_PATH}")
endif()
get_filename_component(UT_CSEC_LIB_DIR "${UT_CSEC_LIB}" DIRECTORY)

target_link_libraries(clos_topo_selector_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    ${UT_CSEC_LIB}
    -ldl
    -lpthread
)

# 运行时优先从 CANN 包目录加载 libc_sec.so，避免依赖外部 LD_LIBRARY_PATH
set_target_properties(clos_topo_selector_test PROPERTIES BUILD_RPATH "${UT_CSEC_LIB_DIR}")

run_llt_test(TARGET clos_topo_selector_test TASK_NUM 1)
