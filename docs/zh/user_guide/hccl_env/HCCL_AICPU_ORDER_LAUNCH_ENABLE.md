# HCCL_AICPU_ORDER_LAUNCH_ENABLE

## 功能描述

此环境变量用于控制AI CPU按序下发（OrderLaunch）功能的总开关。开启时，AI CPU通信算子按下发顺序依次占核执行；关闭时，关闭AI CPU通信算子的按序下发功能，展开过程不再保证跨算子顺序，回退到引入按序下发前的下发行为。

该开关适用于问题定位与版本回退场景，可一键关闭保序链。

HCCL_AICPU_ORDER_LAUNCH_ENABLE支持的取值如下：

- `1`（默认值）：开启AI CPU按序下发。
- `0`：关闭AI CPU按序下发。

仅接受精确的`1`/`0`，默认值为1（开启），不支持`TRUE`/`FALSE`及大小写变体。设置非法取值会导致通信域初始化失败并上报错误码EI0001。

## 配置示例

在启动业务进程前配置：

```bash
export HCCL_AICPU_ORDER_LAUNCH_ENABLE=0
```

此配置关闭AI CPU按序下发，恢复特性引入前的下发行为。

## 使用约束

- 仅HCOMM 9.2.0_beta.2（CANN 9.2.0 beta2）及之后版本支持该开关；低版本HCOMM下按序下发整链路自动跳过，该开关取值不生效，配置为`0`/`1`均无行为差异。
- 进程级环境变量，在通信域初始化时解析并缓存，进程内所有通信域共享同一取值，修改后需重启进程生效。
- 开关由HCOMM通信域管理侧解析，HCCL侧经HcclConfigGetInfo查询项`HCCL_CONFIG_TYPE_AICPU_ORDER_LAUNCH_ENABLE`（uint32_t，1=开启/0=关闭）消费，详见HCOMM的[HcclConfigGetInfo](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/zh/api_ref/comm_mgr_c/HcclConfigGetInfo.md)接口说明。
- 图模式（Ascend IR）Prepare阶段的展开线程覆盖不受该开关影响（属任务下发载体，非保序同步链）。

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
<!-- npu="910" id5 -->
- Atlas训练系列产品：不支持
<!-- end id5 -->
<!-- npu="310p" id4 -->
- Atlas推理系列产品：不支持
<!-- end id4 -->
