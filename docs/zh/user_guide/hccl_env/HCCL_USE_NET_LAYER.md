# HCCL_USE_NET_LAYER

<!-- npu="950" id6 -->
## 功能描述

在Ascend 950PR&950DT系列产品上，指定通信域允许通信使用的网络拓扑层（Net Layer）。创建独立通信域（直接初始化、无需父通信域的通信域）时，若`HcclCommConfig.hcclUseNetLayer`非空，则优先使用该字段，跳过环境变量的读取和校验。字段为空时读取`HCCL_USE_NET_LAYER`。环境变量未设置或为空字符串时，采用默认选层策略。

变量值为最长31字节的字符串。层号为`0`至`7`，用英文逗号分隔；例如`0,2`表示只允许层0和层2，`not(0,2)`表示排除这两层。允许重复层号和标记之间的ASCII空白。层号顺序不影响选层结果。一个表达式只能是层号列表或用`not(...)`包住的一个层号列表。不支持中文标点、范围（如`0-2`）、嵌套或混用正选与排除（如`0,not(2)`）。

## 配置示例

先在一次独立运行中保持环境变量未设置、通信域选层字段为空，使用与目标应用相同的拓扑和rank建域。各rank可使用HCOMM算子开发接口`HcclRankGraphGetLayers`查询本rank的层号。以下代码中的`comm`为已成功创建的通信域句柄：

```cpp
#include <cinttypes>
#include <cstdio>
#include <hccl/hccl_rank_graph.h>

HcclResult PrintLayers(HcclComm comm)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    HcclResult ret = HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    for (uint32_t i = 0; i < netLayerNum; ++i) {
        std::printf("layer=%" PRIu32 "\n", netLayers[i]);
    }
    return HCCL_SUCCESS;
}
```

返回的层号数组由库管理，调用方无需释放。已配置选层的通信域只返回本rank允许使用的层。

例如，各rank查询到的层号均为0和2时，在启动新的应用进程前设置以下变量，即可排除层0，只使用层2：

```bash
export HCCL_USE_NET_LAYER='not(0)'
```

## 使用约束

- `HcclCommInitRootInfo`、`HcclCommInitRootInfoConfig`、`HcclCommInitClusterInfo`和`HcclCommInitClusterInfoConfig`支持显式选层。
- `HcclCommInitAll`没有通信域配置参数，选层配置来自进程环境变量。若该变量指定了合法的显式选层，即使只为跨机通信域设置，同一进程中的`HcclCommInitAll`也会返回`HCCL_E_NOT_SUPPORT`。混用两种建域方式时，可保持环境变量未设置，通过支持选层的Config接口单独配置目标通信域。
- 在多root建域流程中，`HcclGetRootInfoScalable`用于获取root信息。遇到合法的显式选层配置时，`HcclCommInitRootInfoScalable`返回`HCCL_E_NOT_SUPPORT`。
- 正选或排除的层都必须存在，结果至少保留一个可用层。语法非法、仅含空白、超长、引用不存在的层或结果为空时，返回`HCCL_E_PARA`。
- 使用显式选层时，同一通信域内各rank的有效层集合必须一致。普通RootInfo建域时，若各rank上报的有效层集合不同，root会广播`HCCL_E_PARA`，参与rank在通信域初始化阶段返回该错误。
- 使用`HcclCommInitClusterInfo`或`HcclCommInitClusterInfoConfig`时，每个rank按本进程的配置和rank table中自身的拓扑信息校验并建图。该路径不交换各进程的选层结果。即使各rank都建域成功，也不能据此确认有效层集合一致；应在各rank查询并核对层号，再执行集合通信。
- 创建子通信域时，HCOMM根据传入的子域成员rank从父域拓扑构建子图。候选层取自父图中本rank可见的拓扑层，无需额外传入候选层列表。子域继承父域允许范围内的层，显式子域参数只能进一步收窄该范围。创建子域时不会重新读取环境变量。
- 环境变量首次成功解析后，进程复用该结果，包括未配置状态。请在应用启动前设置。建域成功仅表示选层通过建域校验。如果允许层无法满足某个集合通信算子的要求，后续可能在算法匹配、选路或执行阶段失败，不会回退到已禁用的层。具体错误码取决于失败路径；排查时先核对各rank的有效层，再查看该算子首次报错的阶段和错误码。
<!-- end id6 -->

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas推理系列产品：不支持
<!-- end id5 -->
