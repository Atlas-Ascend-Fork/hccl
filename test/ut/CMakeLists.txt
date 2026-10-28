# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(common/prepare_ut_env)
add_subdirectory(common/alg_parse)
add_subdirectory(common/topo_match)
add_subdirectory(common/adapter_error_manager)
add_subdirectory(common/executor)
add_subdirectory(common/channel_process_link)
add_subdirectory(selector_engine)
add_subdirectory(alltoall_hier)
add_subdirectory(aicpu_task_cache)
# recursive_executor 测试仅在 --experimental 构建（ENABLE_EXPERIMENTAL=ON）时编译
if(ENABLE_EXPERIMENTAL)
    add_subdirectory(reduce_scatter_birs)
    add_subdirectory(recursive_executor)
    add_subdirectory(all_reduce_biar)
    add_subdirectory(reduce_scatter_v_birs)
endif()
