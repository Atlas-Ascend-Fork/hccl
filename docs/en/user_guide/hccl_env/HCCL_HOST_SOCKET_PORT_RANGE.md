# HCCL_HOST_SOCKET_PORT_RANGE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T03:53:33.978Z pushedAt=2026-10-15T08:46:37.182Z -->

## Function

When a communicator is created based on root node information, use this environment variable to configure the communication ports used by HCCL on the host.

This environment variable can be set as a specific port, a port range, or the string `auto`.

- If a specific port number or port range is used, the number of planned ports should be no less than the number of HCCL processes on a single NPU. The port number ranges from 1 to 65535, and you must ensure that the specified ports are not occupied by other processes. Note that ports 1 to 1023 are reserved by the system and should not be used.

    Specific port numbers and port ranges can be used in combination, separated by commas (,). However, port numbers or port ranges between commas must not overlap. For details, see [Configuration Example](#configuration-example).

- If set to `auto`, the host communication port used by HCCL is dynamically allocated by the operating system.

## Configuration Example

```bash
# Method 1: Configure it as a port range.
export HCCL_HOST_SOCKET_PORT_RANGE="60000-60050"
# Method 2: Use a specific port number together with a port range, separated by commas (,).
export HCCL_HOST_SOCKET_PORT_RANGE="60000,60050-60100,60150-60160"
# Method 3: Specify specific port numbers, separated by commas (,).
export HCCL_HOST_SOCKET_PORT_RANGE="56000,56005,56007,56008,56100,56105,56107,56108"
# Method 4: The operating system dynamically allocates port numbers.
export HCCL_HOST_SOCKET_PORT_RANGE="auto"
```

## Constraints

- In single-device multi-process use cases (where multiple processes share one NPU), configure this environment variable. Otherwise, services may fail due to port conflicts. Note that multi-process running incurs resource overheads and affects communication performance.
- This environment variable takes precedence over [HCCL_IF_BASE_PORT](HCCL_IF_BASE_PORT.md). Setting this environment variable means it decides the communication ports used by HCCL on the host.

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR/950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
