/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// gert POD 上下文构造器 (Execute 链路 UT):
// EagerOpExecutionContext 全链标准布局无虚表, 以镜像 shadow 结构按 C 头同字段序覆写底层
// KernelRunContext/ComputeNodeInfo, 输入输出经 values[] 挂 Chain(Tensor), 属性经
// RuntimeAttrs::GetPointerByIndex 桩按 index 注册 — 生产代码零改动直驱 Execute 全路径。
#pragma once
#include <cstdint>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <vector>

#include "exe_graph/runtime/eager_op_execution_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/runtime_tensor.h"
#include "graph/types.h"

// 桩层可注入项 (op_deps_stub.cc / op_layer_stub.cc 定义)
extern const void* g_stubAttrs[16];                    // RuntimeAttrs 按 index
extern std::vector<gert::Tensor*> g_stubOutputTensors; // MallocOutputTensor 顺序弹出
extern size_t g_stubOutputCursor;
extern uint64_t g_stubCclBufSize; // CCL buffer 桩尺寸 (Refresh 多迭代控制)

namespace hcut {

// ---- C 头同字段序镜像 (标准布局, 勿改字段顺序) ----
// TypedContinuousVector<int64_t> 镜像 (gert::ContinuousVector 布局: capacity_/size_/
// reserved_[40]/elements[] 内联), 供 GetListInt 属性桩
struct StubCV {
    size_t capacity;
    size_t size;
    unsigned char reserved[40];
    static constexpr size_t kStubCVMaxElems = 4;
    int64_t elements[kStubCVMaxElems];
};
static_assert(sizeof(StubCV) >= 8 + 40 + 8, "StubCV 布局兜底");
struct StubAnyValue {
    void* pointer;
    void* deleter;
};
struct StubRunCtx {
    size_t input_size;
    size_t output_size;
    const void* compute_node_info;
    const void* kernel_extend_info;
    void** output_start;
    StubAnyValue* values[256];
};
struct StubCni {
    const char* node_type;
    const char* node_name;
    size_t ir_inputs_num;
    size_t inputs_num;
    size_t outputs_num;
    size_t ir_outputs_num;
    size_t runtime_attr_size;
    unsigned char reserved[24];
    uint64_t place_holder;
};

static gert::Tensor MkTensor(std::initializer_list<int64_t> shape, ge::DataType dt, const void* data)
{
    return gert::Tensor(
        gert::StorageShape(shape, shape), gert::StorageFormat(), gert::kOnHost, dt,
        reinterpret_cast<gert::TensorAddress>(const_cast<void*>(data)));
}

// 用例内栈对象: 构造输入/输出/属性 → Build() 得到的 ctx 生命期随本对象
class CtxBuilder {
public:
    ~CtxBuilder()
    {
        for (size_t i = 0; i < 16; i++) {
            g_stubAttrs[i] = nullptr;
        }
        // 输出队列清空: Output() 注册的全是指向本对象 outTensors_ 元素的指针,
        // builder 析构后即悬垂 — 不清则下个不经 fixture 复位的用例会经
        // MallocOutputTensor 桩弹出悬垂指针给生产代码 (无感知 use-after-free)
        g_stubOutputTensors.clear();
        g_stubOutputCursor = 0;
    }

    CtxBuilder& Input(std::initializer_list<int64_t> shape, ge::DataType dt, const void* data)
    {
        inTensors_.push_back(MkTensor(shape, dt, data));
        return *this;
    }

    CtxBuilder& Output(std::initializer_list<int64_t> shape, ge::DataType dt, void* data)
    {
        outTensors_.push_back(MkTensor(shape, dt, data));
        g_stubOutputTensors.push_back(&outTensors_.back());
        return *this;
    }

    CtxBuilder& Attr(size_t index, const char* str)
    {
        g_stubAttrs[index] = str;
        return *this;
    }

    CtxBuilder& Attr(size_t index, int64_t value)
    {
        int64s_.push_back(value);
        g_stubAttrs[index] = &int64s_.back();
        return *this;
    }

    // list int 属性 (shape attr 等): ContinuousVector 为 POD 且数据内联 —
    // 每次注册构造独立存储 (连续 push_back 不搬迁), 返回指针给 GetListInt 解引用
    CtxBuilder& AttrListInt(size_t index, const std::vector<int64_t>& values)
    {
        StubCV& cv = listInts_.emplace_back();
        cv.capacity = values.size();
        cv.size = values.size();
        for (size_t i = 0; i < values.size() && i < StubCV::kStubCVMaxElems; i++) {
            cv.elements[i] = values[i];
        }
        g_stubAttrs[index] = &cv;
        return *this;
    }

    gert::EagerOpExecutionContext* Build()
    {
        g_stubOutputCursor = 0;
        runBuf_.assign(sizeof(StubRunCtx), 0);
        // ComputeNodeInfo: 头 + inputs 个 CompileTimeTensorDesc 占位 + 64B 属性区
        cniBuf_.assign(sizeof(StubCni) + inTensors_.size() * sizeof(gert::CompileTimeTensorDesc) + 64, 0);
        auto* run = reinterpret_cast<StubRunCtx*>(runBuf_.data());
        auto* cni = reinterpret_cast<StubCni*>(cniBuf_.data());
        cni->inputs_num = inTensors_.size();
        const size_t total = inTensors_.size() + outTensors_.size();
        chains_.assign(total, StubAnyValue{nullptr, nullptr});
        for (size_t i = 0; i < inTensors_.size(); i++) {
            chains_[i].pointer = &inTensors_[i];
        }
        for (size_t i = 0; i < outTensors_.size(); i++) {
            chains_[inTensors_.size() + i].pointer = &outTensors_[i];
        }
        run->input_size = inTensors_.size();
        run->output_size = outTensors_.size();
        run->compute_node_info = cni;
        run->kernel_extend_info = nullptr;
        run->output_start = nullptr;
        for (size_t i = 0; i < total; i++) {
            run->values[i] = &chains_[i];
        }
        return reinterpret_cast<gert::EagerOpExecutionContext*>(run);
    }

    // InferShapeContext 同为 KernelContext 派生 POD, 复用同一底层布局
    gert::InferShapeContext* BuildAsInferShape()
    {
        Build();
        return reinterpret_cast<gert::InferShapeContext*>(runBuf_.data());
    }

private:
    std::vector<unsigned char> runBuf_;
    std::vector<unsigned char> cniBuf_;
    std::vector<StubAnyValue> chains_;
    std::deque<gert::Tensor> inTensors_; // deque: push_back 不搬移既有元素, 指针长期有效
    std::deque<gert::Tensor> outTensors_;
    std::deque<int64_t> int64s_;
    std::deque<StubCV>
        listInts_; // deque: push 不搬迁, 指针稳定 // deque: 连续 Attr(int64) 注册不搬迁既有元素, 指针不悬垂
};

} // namespace hcut
