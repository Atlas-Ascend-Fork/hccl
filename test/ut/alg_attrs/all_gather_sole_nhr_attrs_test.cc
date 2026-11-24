/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 断言 AicpuAllGatherSoleNHR opCustomCheck 过滤条件收窄后的生产注册行为：
// 16P MESH_1D 总数据量 32~64MB 区段的规避排除仅在 2 server（serverNum==2）组网时生效。
// attrs 经 wrap/ins_v2_all_gather_sole_attrs_wrap.cc 以生产源文件直接编入本测试进程，
// 断言对象为 AlgAttrsRegistry 中的生产注册项，非测试内复刻条件。

#include "gtest/gtest.h"
#include <string>
#include "alg_param.h"
#include "alg_attrs_registry.h"

using namespace ops_hccl;

namespace {
const std::string AG_SOLE_NHR_ALG = "AicpuAllGatherSoleNHR";

// 与生产 lambda 同源计算 totalSize：count * DATATYPE_SIZE_TABLE[fp32] * userRankSize
OpParam MakeParamForTotalSize(u64 totalSize, u32 userRankSize)
{
    OpParam param{};
    param.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    param.DataDes.count = totalSize / (DATATYPE_SIZE_TABLE[HcclDataType::HCCL_DATA_TYPE_FP32] * userRankSize);
    return param;
}

bool Admit(u32 serverNum, u64 totalSize, u32 userRankSize, Level0Shape level0)
{
    const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(AG_SOLE_NHR_ALG);
    if (attrs == nullptr || attrs->op.opCustomCheck == nullptr) {
        ADD_FAILURE() << "attrs/opCustomCheck not registered: " << AG_SOLE_NHR_ALG;
        return false;
    }
    OpParam param = MakeParamForTotalSize(totalSize, userRankSize);
    TopoInfoWithNetLayerDetails topo;
    topo.serverNum = serverNum;
    topo.userRankSize = userRankSize;
    topo.level0Topo = level0;
    return attrs->op.opCustomCheck(param, &topo);
}
} // namespace

// 修改点：规避区段（16P + MESH_1D + totalSize∈[32MB,64MB]）内的排除仅 serverNum==2 生效
TEST(AllGatherSoleNhrOpCustomCheckTest, SegmentExclusionOnlyWhenServerNumTwo)
{
    constexpr u64 totalSize = 48ULL * 1024 * 1024; // 区段中值，16P fp32 下 count 无截断
    constexpr u32 rankSize = 16;
    EXPECT_FALSE(Admit(2, totalSize, rankSize, Level0Shape::MESH_1D)); // serverNum==2：区段内仍排除
    EXPECT_TRUE(Admit(1, totalSize, rankSize, Level0Shape::MESH_1D));  // serverNum!=2：不再排除
    EXPECT_TRUE(Admit(4, totalSize, rankSize, Level0Shape::MESH_1D));  // 4 server：不再排除
}
