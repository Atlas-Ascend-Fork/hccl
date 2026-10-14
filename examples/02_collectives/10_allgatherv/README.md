# 集合通信 - AllGatherV

## 样例介绍

本样例展示如何使用 `HcclAllGatherV()` 接口执行 AllGatherV 操作，包含以下功能点：

- 设备检测，通过 `aclrtGetDeviceCount()` 接口查询可用设备数量。
- 将 rank0 作为 root 节点，通过 `HcclGetRootInfo()` 接口生成 root 节点的 rootinfo 标识信息。

  > rootinfo 标识信息主要包含：Device IP、Device ID 等信息，此信息需广播至集群内所有 rank 用来初始化通信域。

- 在每个线程中，基于 rootinfo 标识信息通过 `HcclCommInitRootInfo()` 接口初始化通信域。
- 调用 `HcclAllGatherV()` 接口，将通信域内所有 rank 的输入数据按照 rank_id 顺序拼接起来，再将结果发送到所有 rank，并打印结果。

与 AllGather 不同的是，AllGatherV 允许通信域内不同 rank 配置不同大小的数据量。本样例将第 i 个 rank 的发送数据量设置为 `i + 1`，并通过 `recvCounts`、`recvDispls` 描述各 rank 数据在 `recvBuf` 中的长度与偏移，用于体现 V（Variable）语义。

## 目录结构

```
├── main.cc       # 样例源文件
├── Makefile      # 编译/构建配置文件
└── allgatherv    # 编译生成的可执行文件
```

## 环境准备

### 环境要求

本样例支持以下产品，组网为单机N卡（N>=2）：

- Ascend 950PR&950DT系列产品
- Atlas A3系列产品
- Atlas A2训练系列产品
- Atlas推理系列产品

> 注意：AllGatherV 算子不支持 Atlas训练系列产品，故本样例亦不支持该产品。详细支持情况请参见 [HcclAllGatherV](../../../docs/zh/api_ref/comm_op_interface/HcclAllGatherV.md)。

### 配置环境变量

```bash
# 设置 CANN 环境变量，以 root 用户默认安装路径为例
source /usr/local/Ascend/cann/set_env.sh
```

## 编译执行样例

在本样例代码目录下执行如下命令：

```bash
make
make test
```

> 注意：可通过设置 `HCCL_OP_EXPANSION_MODE` 环境变量配置通信算子的展开模式，不同产品型号支持的范围可参考[环境变量列表](https://hiascend.com/document/redirect/CannCommunityEnvRef)中该环境变量的使用方法。
>
> ```bash
> # 设置通信算子的展开模式为AI CPU通信引擎
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## 结果示例

第 i 个 rank 发送 `i + 1` 个元素，数据内容初始化为该 rank 的 rank_id。经过 AllGatherV 操作后，每个 rank 的结果是所有 rank 数据的拼接，因此输出长度为各 rank 数据量之和，内容为 rank0 重复 1 次、rank1 重复 2 次……依此类推。

以 8 卡为例，输出如下：

```text
Found 8 NPU device(s) available
rankId: 0, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
rankId: 1, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
rankId: 2, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
rankId: 3, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
rankId: 4, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
rankId: 5, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
rankId: 6, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
rankId: 7, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
```
