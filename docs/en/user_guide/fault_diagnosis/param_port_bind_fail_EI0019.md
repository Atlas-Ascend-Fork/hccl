# Parameter Plane Port Binding Failure (EI0019)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:53:39.994Z pushedAt=2026-10-13T11:47:45.010Z -->

## Symptoms

The CANN log contains the keyword "Please check the port status and whether the port is being used by other process.", as shown below. **In addition, note that port binding failures can also occur during the communicator cluster negotiation phase. You can determine the type based on the "socket type" in the error log.** If the type is 0 or 1, it is a parameter plane port binding failure. If the type is 2, it is a host-side NIC port binding failure during communicator cluster information negotiation. For details, see [Server Node Port Binding Failure (EI0019)](./server_node_port_bind_fail_EI0019.md#server-node-port-binding-failure-ei0019).

```text
[ERROR] HCCL(1009464,all_reduce_test):2025-03-15-00:41:48.470.172 [hccl_socket.cc:110] [1009464][InitGroupStage][RanktableDetect] socket type[0], listen on ip[192.168.2.199] and specific port[16666] fail. Please check the port status and whether the port is being used by other process.
```

## Possible Causes

During the Parameter Plane Setup of a communication operator, the current rank or process needs to bind a port on a device-side NIC, but the port is found to be already occupied by another process.

## Solution

When using device-side NIC ports, HCCL binds port 16666 by default. Therefore, if multiple processes run on the same device and all call the HCCL Communication Operator interface, the port will already be bound by another process, resulting in a failure.

In this case, first check whether running multiple processes on the same device aligns with the task expectations. If it does, the multi-process scenario can be enabled by configuring the `HCCL_NPU_SOCKET_PORT_RANGE` environment variable, for example:

```bash
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
```
