/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// superkernel HCCL 逻辑 UT 的 GE 链接桩。仅模拟对象生命周期和最小返回契约，
// 不验证 GE 的真实注册、平台查询或 TaskDef protobuf 序列化行为。
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include "exe_graph/runtime/task_def_builder.h"
#include "graph/arg_desc_info.h"
#include "graph/ascend_string.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

namespace domi {
class KernelContext {
public:
    std::string argsFormat;
};
class KernelDef {
public:
    uint32_t blockDim = 0U;
    KernelContext context;
};
class TaskDef {
public:
    uint32_t streamId = 0U;
    uint32_t type = 0U;
    KernelDef kernel;
};
} // namespace domi

namespace gert {
domi::TaskDef* TaskDefBuilder::CreateOrGetTaskDef() const
{
    if (impl_ == nullptr) {
        impl_ = new domi::TaskDef;
    }
    return impl_;
}

void TaskDefBuilder::Destroy()
{
    delete impl_;
    impl_ = nullptr;
}

TaskDefBuilder& TaskDefBuilder::SetStreamId(uint32_t streamId)
{
    CreateOrGetTaskDef()->streamId = streamId;
    return *this;
}

TaskDefBuilder& TaskDefBuilder::SetType(uint32_t type)
{
    CreateOrGetTaskDef()->type = type;
    return *this;
}

KernelDefBuilder TaskDefBuilder::GetKernelBuilder() const { return KernelDefBuilder(&CreateOrGetTaskDef()->kernel); }

size_t TaskDefBuilder::ByteSizeLong() const
{
    return impl_ == nullptr ? 0U : sizeof(uint32_t) * 3U + impl_->kernel.context.argsFormat.size();
}

bool TaskDefBuilder::SerializeToArray(uint8_t* data, size_t size) const
{
    const size_t requiredSize = ByteSizeLong();
    if (data == nullptr || requiredSize == 0U || size < requiredSize || CreateOrGetTaskDef() == nullptr) {
        return false;
    }

    size_t offset = 0U;
    std::memcpy(data + offset, &impl_->streamId, sizeof(impl_->streamId));
    offset += sizeof(impl_->streamId);
    std::memcpy(data + offset, &impl_->type, sizeof(impl_->type));
    offset += sizeof(impl_->type);
    std::memcpy(data + offset, &impl_->kernel.blockDim, sizeof(impl_->kernel.blockDim));
    offset += sizeof(impl_->kernel.blockDim);
    const std::string& argsFormat = impl_->kernel.context.argsFormat;
    if (!argsFormat.empty()) {
        std::memcpy(data + offset, argsFormat.data(), argsFormat.size());
    }
    return true;
}

KernelDefBuilder::KernelDefBuilder(domi::KernelDef* impl) : impl_(impl) {}

KernelDefBuilder& KernelDefBuilder::SetBlockDim(uint32_t blockDim)
{
    impl_->blockDim = blockDim;
    return *this;
}

KernelContextBuilder KernelDefBuilder::GetContextBuilder() const { return KernelContextBuilder(&impl_->context); }

KernelContextBuilder::KernelContextBuilder(domi::KernelContext* impl) : impl_(impl) {}

KernelContextBuilder& KernelContextBuilder::SetArgsFormat(const ge::AscendString& argsFormat)
{
    impl_->argsFormat = argsFormat.GetString() == nullptr ? "" : argsFormat.GetString();
    return *this;
}

class OpImplRegisterV2Impl {};

OpImplRegisterV2::OpImplRegisterV2(const ge::char_t* opType) : impl_(nullptr) { (void)opType; }
OpImplRegisterV2::OpImplRegisterV2(OpImplRegisterV2&& registerData) noexcept : impl_(nullptr) { (void)registerData; }
OpImplRegisterV2::OpImplRegisterV2(const OpImplRegisterV2& registerData) : impl_(nullptr) { (void)registerData; }
OpImplRegisterV2::~OpImplRegisterV2() = default;

OpImplRegisterV2& OpImplRegisterV2::GenerateTask(OpGenTaskKernelFunc genTaskFunc)
{
    (void)genTaskFunc;
    return *this;
}
} // namespace gert

namespace ge {
class ArgDescInfoImpl {
public:
    ArgDescInfoImpl(ArgDescType argType, int32_t index, bool folded) : type(argType), irIndex(index), isFolded(folded)
    {}

    ArgDescType type;
    int32_t irIndex;
    bool isFolded;
    HiddenInputSubType hiddenType = HiddenInputSubType::kEnd;
};

ArgDescInfo::ArgDescInfo(ArgDescType argType, int32_t irIndex, bool isFolded)
    : impl_(std::make_unique<ArgDescInfoImpl>(argType, irIndex, isFolded))
{}
ArgDescInfo::~ArgDescInfo() = default;
ArgDescInfo::ArgDescInfo(const ArgDescInfo& other)
    : impl_(other.impl_ == nullptr ? nullptr : std::make_unique<ArgDescInfoImpl>(*other.impl_))
{}
ArgDescInfo::ArgDescInfo(ArgDescInfo&& other) noexcept = default;
ArgDescInfo& ArgDescInfo::operator=(const ArgDescInfo& other)
{
    if (this != &other) {
        impl_ = other.impl_ == nullptr ? nullptr : std::make_unique<ArgDescInfoImpl>(*other.impl_);
    }
    return *this;
}
ArgDescInfo& ArgDescInfo::operator=(ArgDescInfo&& other) noexcept = default;

ArgDescInfo ArgDescInfo::CreateHiddenInput(HiddenInputSubType hiddenType)
{
    ArgDescInfo desc(ArgDescType::kHiddenInput, 0);
    desc.impl_->hiddenType = hiddenType;
    return desc;
}

ArgDescType ArgDescInfo::GetType() const { return impl_ == nullptr ? ArgDescType::kEnd : impl_->type; }

HiddenInputSubType ArgDescInfo::GetHiddenInputSubType() const
{
    return impl_ == nullptr ? HiddenInputSubType::kEnd : impl_->hiddenType;
}

int32_t ArgDescInfo::GetIrIndex() const { return impl_ == nullptr ? -1 : impl_->irIndex; }

AscendString::AscendString(const char_t* const name)
    : name_(name == nullptr ? nullptr : std::make_shared<std::string>(name))
{}
AscendString::AscendString(const char_t* const name, size_t length)
    : name_(name == nullptr ? nullptr : std::make_shared<std::string>(name, length))
{}
const char_t* AscendString::GetString() const { return name_ == nullptr ? "" : name_->c_str(); }
size_t AscendString::GetLength() const { return name_ == nullptr ? 0U : name_->size(); }
} // namespace ge

namespace fe {
PlatformInfoManager::PlatformInfoManager() = default;
PlatformInfoManager::~PlatformInfoManager() = default;

PlatformInfoManager& PlatformInfoManager::Instance()
{
    static PlatformInfoManager manager;
    return manager;
}

uint32_t PlatformInfoManager::GetPlatformInfoWithOutSocVersion(PlatformInfo& platformInfo, OptionalInfo& optionalInfo)
{
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optionalInfo.soc_version = "Ascend910B";
    return 0U;
}

uint32_t PlatformInfoManager::InitializePlatformInfo(std::string socVersion)
{
    (void)socVersion;
    return 0U;
}

uint32_t PlatformInfoManager::GetPlatformInfo(
    const std::string socVersion, PlatformInfo& platformInfo, OptionalInfo& optionalInfo)
{
    platformInfo.str_info.short_soc_version = socVersion;
    optionalInfo.soc_version = socVersion;
    return 0U;
}
} // namespace fe
