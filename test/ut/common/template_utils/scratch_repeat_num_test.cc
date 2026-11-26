/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "template_utils.h"

using namespace ops_hccl;

TEST(CheckScratchRepeatNumTest, AllowsDefaultZeroRepeatNumAsNoOp)
{
    TemplateDataParams params;
    EXPECT_EQ(params.repeatNum, 0U);
    EXPECT_EQ(CheckScratchRepeatNum(params, "CheckScratchRepeatNumTest"), HCCL_SUCCESS);
}

TEST(CheckScratchRepeatNumTest, AcceptsSingleRepeatNum)
{
    TemplateDataParams params;
    params.repeatNum = 1;
    EXPECT_EQ(CheckScratchRepeatNum(params, "CheckScratchRepeatNumTest"), HCCL_SUCCESS);
}

TEST(CheckScratchRepeatNumTest, RejectsMultiRepeatNum)
{
    TemplateDataParams params;
    params.repeatNum = 2;
    EXPECT_EQ(CheckScratchRepeatNum(params, "CheckScratchRepeatNumTest"), HCCL_E_INTERNAL);
}
