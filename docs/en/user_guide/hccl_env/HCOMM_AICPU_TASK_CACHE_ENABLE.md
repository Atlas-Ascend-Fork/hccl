# HCOMM_AICPU_TASK_CACHE_ENABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-06T09:49:31.388Z pushedAt=2026-11-16T06:28:10.225Z nl=2 -->

## Function

Controls the AI CPU task cache of HCCL operators. It is parsed by HCCL. Reusing cached tasks reduces operator expansion overhead but consumes a certain amount of device memory.

- `true` or `1`: Enables caching.
- `false` or `0`: Disables caching.

The value is case-insensitive. Leading and trailing whitespace is removed before parsing. If the variable is not set, is an empty string, or contains only whitespace, it is treated as unconfigured. Any other value reports a parameter error during environment variable initialization.

This variable has no independent default value. When it is not configured, the cache switch result of [HCCL_OP_EXPANSION_MODE](HCCL_OP_EXPANSION_MODE.md) is used. When a valid value is explicitly configured, it overrides that result. For example, `HCCL_OP_EXPANSION_MODE=AICPU_CacheDisable` disables caching, but it can still be overridden to be enabled by setting this variable to `1`.

## Configuration Example

Configure it before starting a service process:

```bash
export HCCL_OP_EXPANSION_MODE=AICPU_TS
export HCOMM_AICPU_TASK_CACHE_ENABLE=0
```

This configuration keeps the AI CPU expansion mode and disables caching.

## Constraints

- It only controls caching and does not change the operator expansion mode. After caching is enabled, whether the cache is used still depends on admission conditions such as the device, operator, algorithm, and networking.
- When the value of `HCCL_OP_EXPANSION_MODE` is invalid, initialization still reports an error. This variable cannot make it a valid configuration.

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
