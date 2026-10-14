# 集合通信 - ReduceScatterV

## 样例介绍

本样例展示如何使用 `HcclReduceScatterV()` 接口执行 ReduceScatterV 操作，包含以下功能点：

- 设备检测，通过 `aclrtGetDeviceCount()` 接口查询可用设备数量。
- 将 rank0 作为 root 节点，通过 `HcclGetRootInfo()` 接口生成 root 节点的 rootinfo 标识信息。

  > rootinfo 标识信息主要包含：Device IP、Device ID 等信息，此信息需广播至集群内所有 rank 用来初始化通信域。

- 在每个线程中，基于 rootinfo 标识信息通过 `HcclCommInitRootInfo()` 接口初始化通信域。
- 调用 `HcclReduceScatterV()` 接口，将所有 rank 对应编号的输入数据相加后，再把结果按照编号分散到各个 rank，并打印结果。

与 ReduceScatter 不同的是，ReduceScatterV 允许同一 rank 内不同编号的数据块大小不一致。本样例将编号为 i 的数据块大小设置为 `i + 1`，并通过 `sendCounts`、`sendDispls` 描述各数据块的长度与偏移，用于体现 V（Variable）语义。注意所有 rank 的 `sendCounts`、`sendDispls` 需保持一致。

## 目录结构

```
├── main.cc            # 样例源文件
├── Makefile           # 编译/构建配置文件
└── reduce_scatterv    # 编译生成的可执行文件
```

## 环境准备

### 环境要求

本样例支持以下产品，组网为单机N卡（N>=2）：

- Ascend 950PR&950DT系列产品
- Atlas A3系列产品
- Atlas A2训练系列产品
- Atlas推理系列产品

> 注意：ReduceScatterV 算子不支持 Atlas训练系列产品，故本样例亦不支持该产品。详细支持情况请参见 [HcclReduceScatterV](../../../docs/zh/api_ref/comm_op_interface/HcclReduceScatterV.md)。

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

每个 rank 的输入数据全部初始化为该 rank 的 rank_id，编号为 i 的数据块大小为 `i + 1`。经过 ReduceScatterV 操作后，编号为 i 的部分是所有 rank 对应位置数据的和，第 i 个 rank 得到编号为 i、长度为 `i + 1` 的结果段。

以 8 卡为例，所有 rank 的 rank_id 之和为 `0+1+…+7 = 28`，因此各 rank 输出如下：

```text
Found 8 NPU device(s) available
rankId: 0, output: [ 28 ]
rankId: 1, output: [ 28 28 ]
rankId: 2, output: [ 28 28 28 ]
rankId: 3, output: [ 28 28 28 28 ]
rankId: 4, output: [ 28 28 28 28 28 ]
rankId: 5, output: [ 28 28 28 28 28 28 ]
rankId: 6, output: [ 28 28 28 28 28 28 28 ]
rankId: 7, output: [ 28 28 28 28 28 28 28 28 ]
```
