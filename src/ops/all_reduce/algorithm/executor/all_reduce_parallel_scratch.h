/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_PARALLEL_SCRATCH_H
#define ALL_REDUCE_PARALLEL_SCRATCH_H

#include <algorithm>
#include <cstdint>

namespace ops_hccl {

// 所有容量及偏移均以元素为单位。count 须先由 CalcAllReduceParallelSliceCount 限定。
inline uint64_t CalcAllReduceParallelScratchCount(uint64_t count, uint32_t rankSize)
{
    const uint64_t maxRemainder = rankSize > 0 ? rankSize - 1 : 0;
    // count = q * rankSize + r，尾块为 q + r，接收区需 count + (rankSize - 1) * r。
    // 加上输出区，并覆盖第二阶段子块前的偏移，使用 r <= min(count, rankSize - 1) 的上界。
    return count * 2 + maxRemainder * std::min(count, maxRemainder);
}

inline uint64_t CalcAllReduceParallelSliceCount(uint64_t countUpperBound, uint64_t scratchCount, uint32_t rankSize)
{
    const uint64_t maxRemainder = rankSize > 0 ? rankSize - 1 : 0;
    uint64_t lower = 0;
    uint64_t upper = std::min(countUpperBound, scratchCount / 2);
    while (lower < upper) {
        const uint64_t count = lower + (upper - lower + 1) / 2;
        // 两路尾块余数之和不超过 min(count, 2 * (rankSize - 1))，分轮容量须包含两路预留。
        const uint64_t remainderCount = std::min(count, maxRemainder * 2);
        const uint64_t available = scratchCount - count * 2;
        if (maxRemainder == 0 || remainderCount <= available / maxRemainder) {
            lower = count;
        } else {
            upper = count - 1;
        }
    }
    return lower;
}

} // namespace ops_hccl

#endif // ALL_REDUCE_PARALLEL_SCRATCH_H
