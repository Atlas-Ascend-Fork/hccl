# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# hccl_op_utils (GE 自定义算子公共工具) 纯函数 UT: 真实编译被测源, hcomm/rt/log 依赖由本目录桩提供
set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

set(ut_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_op_utils_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_op_utils_read_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_custom_op_parse_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_op_v_parse_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_rt_utils_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hcom_op_layer_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hcom_execute_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hcom_guard_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_reg_gate_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/log_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/op_deps_stub.cc
    ${_repo_root}/src/common/op_graph/hccl_op_utils.cc
    ${_repo_root}/src/common/op_graph/hccl_op_v_utils.cc
    ${_repo_root}/src/common/op_graph/hccl_rt_utils.cc
    # 算子层 UT (M2/M10): 链接真实算子 TU, 外部依赖经 op_layer_stub.cc 桩化
    ${_repo_root}/src/common/op_graph/hccl_custom_op.cc
    ${_repo_root}/src/ops/all_reduce/op_graph/hcom_all_reduce_op.cc
    ${_repo_root}/src/ops/reduce_scatter/op_graph/hcom_reduce_scatter_op.cc
    ${_repo_root}/src/ops/send/op_graph/hcom_send_op.cc
    ${_repo_root}/src/ops/broadcast/op_graph/hcom_broadcast_op.cc
    ${_repo_root}/src/ops/all_to_all_v/op_graph/hcom_alltoallv_op.cc
    ${_repo_root}/src/ops/all_gather_v/op_graph/hcom_all_gather_v_op.cc
    ${_repo_root}/src/ops/recv/op_graph/hcom_receive_op.cc
    ${_repo_root}/src/ops/reduce/op_graph/hcom_reduce_op.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/op_layer_stub.cc
)

add_executable(op_graph_utils_test ${ut_src_list})

target_compile_options(op_graph_utils_test PRIVATE
    -U_FORTIFY_SOURCE
)

# 头路径: 本目录桩优先 (log.h), 其余为被测源与 toolkit 公共头 (hccl.h/hcom.h/graph/gert 均为 header-only 使用)
target_include_directories(op_graph_utils_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${_repo_root}/src/common/op_graph
    ${_repo_root}/src/ops/all_reduce/op_graph
    ${_repo_root}/src/ops/reduce_scatter/op_graph
    ${_repo_root}/src/ops/send/op_graph
    ${_repo_root}/src/ops/broadcast/op_graph
    ${_repo_root}/src/ops/all_to_all_v/op_graph
    ${_repo_root}/src/ops/all_gather_v/op_graph
    ${_repo_root}/src/ops/recv/op_graph
    ${_repo_root}/src/ops/reduce/op_graph
    ${_repo_root}/src/common
    ${_repo_root}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
)

# CANN 版本号探测: 与主构建口径一致 (宏逻辑照搬 test/ut/selector_engine)
macro(_detect_cann_version _target)
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
        target_include_directories(${_target} PRIVATE
            ${_cann_arch_dir}/include
            ${_cann_arch_dir}/include/hccl
            ${_cann_arch_dir}/include/hcomm
            ${_cann_arch_dir}/pkg_inc
        )
        foreach(_arch_dir ${_cann_arch_dirs})
            set(_ut_cann_ver_header "${_arch_dir}/include/version/cann_version.h")
            if(EXISTS "${_ut_cann_ver_header}")
                file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line
                     REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
                if(_ut_cann_ver_line)
                    string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
                        _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
                    math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
                    target_compile_definitions(${_target}
                        PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
                    message(STATUS "[UT] Detected CANN_VERSION_NUM = ${_UT_CANN_VERSION_NUM}")
                endif()
                break()
            endif()
        endforeach()
    endif()
endmacro()

_detect_cann_version(op_graph_utils_test)

target_link_libraries(op_graph_utils_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET op_graph_utils_test TASK_NUM 1)

# 注册门控缺符号降级 UT: 同套桩 + HCCL_UT_HIDE_HCOMM_RES_APIS 隐藏两接口桩定义,
# hccl_custom_op.cc 弱引用解析为 null → 「缺符号不注册」降级分支可被断言
add_executable(op_graph_reg_gate_test
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_reg_gate_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/log_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/op_deps_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/op_layer_stub.cc
    ${_repo_root}/src/common/op_graph/hccl_custom_op.cc
    ${_repo_root}/src/common/op_graph/hccl_op_v_utils.cc
    ${_repo_root}/src/common/op_graph/hccl_rt_utils.cc
)

target_compile_options(op_graph_reg_gate_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_compile_definitions(op_graph_reg_gate_test PRIVATE HCCL_UT_HIDE_HCOMM_RES_APIS)

target_include_directories(op_graph_reg_gate_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${_repo_root}/src/common/op_graph
    ${_repo_root}/src/common
    ${_repo_root}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
)

_detect_cann_version(op_graph_reg_gate_test)

target_link_libraries(op_graph_reg_gate_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

add_executable(op_graph_reg_gate_hide_acq
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_reg_gate_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/log_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/op_deps_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/op_layer_stub.cc
    ${_repo_root}/src/common/op_graph/hccl_custom_op.cc
    ${_repo_root}/src/common/op_graph/hccl_op_v_utils.cc
    ${_repo_root}/src/common/op_graph/hccl_rt_utils.cc
)

target_compile_options(op_graph_reg_gate_hide_acq PRIVATE
    -U_FORTIFY_SOURCE
)

target_compile_definitions(op_graph_reg_gate_hide_acq PRIVATE HCCL_UT_HIDE_DEDICATED_ACQUIRE)

target_include_directories(op_graph_reg_gate_hide_acq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${_repo_root}/src/common/op_graph
    ${_repo_root}/src/common
    ${_repo_root}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
)

_detect_cann_version(op_graph_reg_gate_hide_acq)

target_link_libraries(op_graph_reg_gate_hide_acq PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

add_executable(op_graph_reg_gate_hide_res
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_reg_gate_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/log_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/op_deps_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/op_layer_stub.cc
    ${_repo_root}/src/common/op_graph/hccl_custom_op.cc
    ${_repo_root}/src/common/op_graph/hccl_op_v_utils.cc
    ${_repo_root}/src/common/op_graph/hccl_rt_utils.cc
)

target_compile_options(op_graph_reg_gate_hide_res PRIVATE
    -U_FORTIFY_SOURCE
)

target_compile_definitions(op_graph_reg_gate_hide_res PRIVATE HCCL_UT_HIDE_THREAD_RES_INFO)

target_include_directories(op_graph_reg_gate_hide_res PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${_repo_root}/src/common/op_graph
    ${_repo_root}/src/common
    ${_repo_root}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
)

_detect_cann_version(op_graph_reg_gate_hide_res)

target_link_libraries(op_graph_reg_gate_hide_res PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)


run_llt_test(TARGET op_graph_reg_gate_test TASK_NUM 1)
run_llt_test(TARGET op_graph_reg_gate_hide_acq TASK_NUM 1)
run_llt_test(TARGET op_graph_reg_gate_hide_res TASK_NUM 1)
