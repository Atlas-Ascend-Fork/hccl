# HCCL_CONNECT_TIMEOUT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T03:50:42.415Z pushedAt=2026-10-15T08:46:37.155Z -->

## Function

In distributed training or inference tasks, processes on different devices may become desynchronized due to other factors before collective communication initialization. This environment variable limits the timeout wait period for socket connection establishment between different devices, during which each device process waits for other devices to establish their connections and synchronize.

This environment variable must be configured as an integer with a value range of [120, 7200] and a default value of 120, in seconds.

**Note**: The actual connection establishment timeout wait period is the value of this environment variable plus 20 seconds. For example, if this environment variable is set to 150 seconds, the actual wait time is 170 seconds. The additional 20 seconds are used to notify each node of the reason for communicator initialization failure.

> [!NOTE] Note
> The value of this environment variable affects the exception reporting time for connection failures.

## Configuration Example

```bash
export HCCL_CONNECT_TIMEOUT=200
```

## Constraints

None

## Applicable Products

<!-- npu="950" id5 -->
- Ascend 950PR&950DT products: Supported
<!-- end id5 -->
<!-- npu="A3" id4 -->
- Atlas A3 products: Supported
<!-- end id4 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="910" id1 -->
- Atlas training products: Supported
<!-- end id1 -->
<!-- npu="310p" id2 -->
- Atlas inference products: Supported
<!-- end id2 -->
