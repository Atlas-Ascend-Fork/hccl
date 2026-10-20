# Server Node Port Binding Failure (EI0019)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:54:56.287Z pushedAt=2026-10-14T03:02:11.866Z -->

## Symptoms

The log contains the EI0019 error. The error message is as follows:

```text
[PID: 2267203] 2025-11-21-11:38:29.575.404 Communication_Error_Bind_IP_Port(EI0019): Failed to enable listening for the host network adapter socket.Reason: The IP address 192.168.1.100 and port 50001 have already been bound.
```

The CANN log contains the keyword "socket type\[2\], \*\*\* Please check the port status and whether the port is being used by other process.", as shown below. **In addition, note that a port binding failure may also occur during the parameter plane link setup phase when communication operators are delivered. This can be determined based on the "socket type" in the error log. If the type is 2, the host-side NIC port binding failed during communicator cluster negotiation. For Atlas A3 products and Atlas A2 products, if the type is 0 or 1, the parameter plane port binding failed. For details, see** [Parameter Plane Port Binding Failure (EI0019)](./param_port_bind_fail_EI0019.md#parameter-plane-port-binding-failure-ei0019).

```text
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.860 [hccl_socket.cc:110] [3626636][InitChannelStage][RanktableDetect] socket type[2], listen on ip[192.168.1.100%enp53s0f2] and specific port[60000] fail. Please check the port status and whether the port is being used by other process.
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.869 [topoinfo_detect.cc:744] [3626636][InitGroupStage][RanktableDetect]StartRootNetwork failed, ret[7]
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.874 [topoinfo_detect.cc:233] [3626636][InitGroupStage][RanktableDetect]SetupServer failed, hostIP[192.168.1.100%enp53s0f2] and hostPort[60000] ret[7]
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.882 [op_base.cc:1071] [3626636][InitGroupStage][RanktableDetect]HcclGetRootInfo failed, ret[7]
```

## Possible Causes

HCCL port binding fails. During the communicator creation phase, HCCL needs to bind ports 60000–60031 by default. If these ports are already bound at that time, the HCCL port binding fails, which in turn causes the communicator creation to fail.

## Solution

The port range can be configured as follows:

- Specify the starting port number and port reservation range for the host network interface card through the [HCCL_IF_BASE_PORT](../hccl_env/HCCL_IF_BASE_PORT.md) environment variable.
- If the service requires multiple processes to run simultaneously on a single NPU, configure the Host-side communication port range used by HCCL through [HCCL_HOST_SOCKET_PORT_RANGE](../hccl_env/HCCL_HOST_SOCKET_PORT_RANGE.md) to avoid port conflicts between multiple processes. **This operation applies only to the following products**:
    <!-- npu="A3" id2 -->
  - Atlas A3 products
    <!-- end id2 -->
  <!-- npu="910b" id3 -->
  - Atlas A2 products
    <!-- end id3 -->
