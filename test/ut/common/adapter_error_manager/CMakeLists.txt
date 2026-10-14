# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

file(GLOB UT_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

# 被测源文件：ErrMgr上下文适配层（Get/Set字段映射）+ CommWorker（errCtx透传链路的采集/恢复端）+ 桩
# （UT零SDK库依赖：securec/log/ACL/ErrorMgr均以桩实现，memcpy_s桩支持用例注入失败触达失败分支）
set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/common/adapter_error_manager_pub.cc
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/comm_worker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/err_mgr_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/securec_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/log_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/aclrt_stub.cc
)

add_executable(adapter_error_manager_test ${UT_FILES} ${SOURCE_FILES})

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/x86_64-linux")
else()
    set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/aarch64-linux")
endif()

target_include_directories(adapter_error_manager_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/ops/op_common
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/base
    ${ASCEND_HOME_PATH}/pkg_inc/hccl
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
)

target_compile_options(adapter_error_manager_test PRIVATE
    -U_FORTIFY_SOURCE
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
)

target_link_libraries(adapter_error_manager_test PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    GTest::gtest
    GTest::gtest_main
    -lpthread
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -lgcov
    -lrt
    -lpthread
)

run_llt_test(TARGET adapter_error_manager_test)
