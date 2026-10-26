# HCOMM_AICPU_TASK_CACHE_ENABLE

## 功能描述

此环境变量用于控制HCCL算子的AI CPU任务缓存（Task Cache）开关，由HCCL解析。复用缓存的任务可减少算子展开开销，但会占用一定的显存。

- `true`或`1`：开启cache开关。
- `false`或`0`：关闭cache开关。

取值大小写不敏感，解析前会去除首尾空白。未设置、空串或仅含空白时均按未配置处理；其他取值会在环境变量初始化时报参数错误。

该变量没有独立默认值。未配置时，沿用[HCCL_OP_EXPANSION_MODE](HCCL_OP_EXPANSION_MODE.md)的cache开关结果；显式配置有效值时，覆盖该结果。例如，`HCCL_OP_EXPANSION_MODE=AICPU_CacheDisable`关闭cache开关，仍可被本变量的`1`覆盖为开启。

## 配置示例

在启动业务进程前配置：

```bash
export HCCL_OP_EXPANSION_MODE=AICPU_TS
export HCOMM_AICPU_TASK_CACHE_ENABLE=0
```

此配置保持AI CPU展开模式，并关闭cache开关。

## 使用约束

- 仅控制cache开关，不改变算子的展开模式。开启开关后，是否使用缓存仍取决于设备、算子、算法及组网等准入条件。
- `HCCL_OP_EXPANSION_MODE`取值非法时，初始化仍会报错，本变量不能使其变为有效配置。

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
