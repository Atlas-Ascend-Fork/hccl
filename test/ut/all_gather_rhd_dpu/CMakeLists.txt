# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

add_executable(all_gather_rhd_dpu_test
    ag_rhd_test_stubs.cc
    log_stub.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/template_utils.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/alg_v2_template_base.cc
)

target_compile_options(all_gather_rhd_dpu_test PRIVATE
    -U_FORTIFY_SOURCE
    -ffunction-sections
    -fdata-sections
    -fno-access-control
)

# 与 reduce_scatter_nhr_dpu 同款口径：AICPU_COMPILE 裁掉 DPU 侧下发段
# （RunAllGatherRHDDpu/RunStepDpu/BuildStepSlicesDpu），host 侧 KernelRun 完整保留，
# 本 UT 即以真实 KernelRun 驱动 batch 守卫窗口的全部错误出口。
target_compile_definitions(all_gather_rhd_dpu_test PRIVATE AICPU_COMPILE)

target_include_directories(all_gather_rhd_dpu_test PRIVATE
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/ops/op_common
    ${_repo_root}/src/ops/op_common/selector
    ${_repo_root}/src/ops/op_common/topo_info
    ${_repo_root}/src/ops/op_common/algorithm/template
    ${_repo_root}/src/ops/op_common/algorithm/template/registry
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper
    ${_repo_root}/src/ops/op_common/algorithm/executor
    ${_repo_root}/src/ops/op_common/algorithm/executor/channel
    ${_repo_root}/src/ops/all_gather/algorithm/template/aicpu
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
)

file(GLOB _cann_arch_dirs LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
foreach(_dir ${_cann_arch_dirs})
    if(_dir MATCHES "/${CMAKE_SYSTEM_PROCESSOR}-linux$")
        target_include_directories(all_gather_rhd_dpu_test PRIVATE
            ${_dir}/include
            ${_dir}/include/hccl
            ${_dir}/include/hcomm/ccu
            ${_dir}/pkg_inc
            ${_dir}/pkg_inc/hccl
            ${_dir}/pkg_inc/base
            ${_dir}/pkg_inc/hcomm/ccu
        )
    endif()
    set(_ut_cann_ver_header "${_dir}/include/version/cann_version.h")
    if(EXISTS "${_ut_cann_ver_header}")
        file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
        if(_ut_cann_ver_line)
            string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
                _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
            math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
            target_compile_definitions(all_gather_rhd_dpu_test PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
            break()
        endif()
    endif()
endforeach()

# RHD DPU 生产源与 guard 用例仅在 >= 9.0 段编译（对齐 reduce_scatter_nhr_dpu 的
# _UT_CANN_VERSION_NUM 门控）：该源依赖新版 CANN 才提供的 DPU wrapper/HCOMM batch
# 接口（hcomm_primitives.h 的 HcommBatchModeStart/End 等），低版本包上无条件编译会整体红。
# 差异说明：本目标与 nhr 不同，REGISTER_TEMPLATE_V2 在生产源内未包 #ifndef AICPU_COMPILE，
# 故 alg_v2_template_register.cc 须一并编入；InsTempAllGatherRHD 基类（vtable/析构/
# SetchannelsPerRank/CalcScratchMultiple 等非覆写虚函数落盘于该 .cc）亦须编入，
# 其对外部原语的引用由 ag_rhd_test_stubs.cc 桩满足。
if(_UT_CANN_VERSION_NUM GREATER_EQUAL 90000000)
    target_sources(all_gather_rhd_dpu_test PRIVATE
        test_batch_mode_guard.cc
        ${_repo_root}/src/ops/op_common/algorithm/template/registry/alg_v2_template_register.cc
        ${_repo_root}/src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_rhd.cc
        ${_repo_root}/src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_rhd_dpu.cc
    )
endif()

target_link_libraries(all_gather_rhd_dpu_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
    -Wl,--gc-sections
)

run_llt_test(TARGET all_gather_rhd_dpu_test TASK_NUM 1)
