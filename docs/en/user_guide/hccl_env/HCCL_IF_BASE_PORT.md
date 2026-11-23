# HCCL_IF_BASE_PORT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-06T09:47:02.606Z pushedAt=2026-11-16T06:28:10.164Z nl=2 -->

## Function

When a communicator is created based on root node information, use this environment variable to specify the start port number of the host NIC. After configuration, HCCL calculates the port used by each NPU based on "start port + physical device ID". These ports are used for both cluster information collection and control plane link establishment. The actual number of occupied ports depends on the product form and the number of NPUs per server.

This environment variable must be configured as an integer in the range of [1024, 65520]. Ensure that the ports to allocate are not occupied.

## Configuration Example

```bash
export HCCL_IF_BASE_PORT=50000
```

## Constraints

In distributed use cases, HCCL uses certain ports of the host server for cluster information collection and control plane link establishment. The operating system must reserve these ports.

- If you do not specify ports through the HCCL_IF_BASE_PORT environment variable, HCCL selects ports in the range 60000-60031 by default. Run the following command to reserve the operating system ports in this range:

    ```bash
    sysctl -w net.ipv4.ip_local_reserved_ports=60000-60031
    ```

- If the port is specified through HCCL_IF_BASE_PORT, for example, port 50000, HCCL uses ports 50000–50031. Run the following command to reserve this range of operating system ports:

    ```bash
    sysctl -w net.ipv4.ip_local_reserved_ports=50000-50031
    ```

## Applicable Products

<!-- npu="950" id3 -->
- Ascend 950PR&950DT products: Supported
<!-- end id3 -->
<!-- npu="A3" id4 -->
- Atlas A3 products: Supported
<!-- end id4 -->
<!-- npu="910b" id5 -->
- Atlas A2 products: Supported
<!-- end id5 -->
<!-- npu="910" id1 -->
- Atlas training products: Supported
<!-- end id1 -->
<!-- npu="310p" id2 -->
- Atlas inference products: Supported
<!-- end id2 -->
