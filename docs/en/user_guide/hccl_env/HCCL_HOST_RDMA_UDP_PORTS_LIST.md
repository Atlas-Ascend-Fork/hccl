# HCCL_HOST_RDMA_UDP_PORTS_LIST

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-06T09:47:01.393Z pushedAt=2026-11-16T06:28:10.160Z nl=2 -->

## Function

In RoCE communication over host NICs, if you need to specify the UDP source ports used by multiple QPs (Queue Pairs), you can use this environment variable to configure a UDP source port list based on the NPU physical device ID.

In addition, you can manually create the `/etc/hcomm.cfg` file to configure the QP count and UDP source port list.

This environment variable is configured in the following format:

```text
<phy_dev_id>:<src_port0>,<src_port1>,...,<src_portN>[;<phy_dev_id>:<src_port0>,<src_port1>,...,<src_portN>]
```

- `phy_dev_id` indicates the NPU physical device ID, configured as a non-negative decimal integer. Each NPU physical device ID can be configured only once.
- `src_port` indicates the UDP source port, configured as a decimal integer ranging from 1 to 65535. Ports 1 to 1023 are system-reserved ports and should be avoided. A maximum of 32 UDP source ports can be configured for each NPU physical device ID.
- Configurations of multiple NPU physical devices are separated by semicolons (`;`), and multiple UDP source ports of the same NPU physical device are separated by commas (`,`). The configuration must not contain spaces.
- The value of this environment variable must not exceed 32 x 1024 characters.

HCCL selects the corresponding UDP source port list based on the NPU physical device ID currently in use. If the number of QPs is greater than the number of UDP source ports, the ports are used cyclically starting from the first port in the list. If the number of QPs is smaller than the number of UDP source ports, only the first ports matching the number of QPs are used.

This environment variable applies only when specifying UDP source ports and does not change the number of QPs used between two ranks. The number of QPs can be configured through the environment variable [HCCL_RDMA_QPS_PER_CONNECTION](HCCL_RDMA_QPS_PER_CONNECTION.md). By default, this environment variable is not configured, and the system selects UDP source ports based on other configurations or the default mode.

## Configuration Example

```bash
export HCCL_HOST_RDMA_UDP_PORTS_LIST="0:10000,10015;1:10016,10031"
```

The preceding configuration indicates that NPU physical device 0 uses UDP source ports 10000 and 10015, and NPU physical device 1 uses UDP source ports 10016 and 10031.

## Constraints

- This environment variable takes effect only in RoCE communication use cases over host NICs.
- If the NPU physical device ID in use is not configured for this environment variable, this environment variable does not specify UDP source ports.
- This environment variable has lower priority than the UDP source port configuration in `/etc/hcomm.cfg` and higher priority than the environment variable [HCCL_RDMA_QP_PORT_CONFIG_PATH](HCCL_RDMA_QP_PORT_CONFIG_PATH.md). If `/etc/hcomm.cfg` does not provide the configuration for the NPU physical device currently in use, the UDP source port list matching that device in this environment variable is used first. If this environment variable does not provide the configuration for that device, the configuration matching the current IP pair in HCCL_RDMA_QP_PORT_CONFIG_PATH or the default method is used.
- If the configuration format or value is invalid, communicator initialization fails.

## Setting the Configuration File

In RoCE communication over host NICs, you can configure the QP count and UDP source port list by NPU physical device ID through `/etc/hcomm.cfg`, so that RDMA communication between two ranks uses multiple QPs.

HCCL selects the corresponding configuration based on the NPU physical device ID currently in use. The configuration of the same physical device applies to RoCE communication performed by that device over host NICs.

### Configuration Format

The configuration file path is fixed at `/etc/hcomm.cfg`. The configuration format for each NPU physical device is as follows:

```text
udp_port_mode_<phy_dev_id>=multi_qp
multi_qp_count_<phy_dev_id>=<qp_count>
multi_qp_udp_ports_<phy_dev_id>=<src_port0>,<src_port1>,...,<src_portN>
```

- `phy_dev_id` is the NPU physical device ID, configured as a non-negative decimal integer with a value range of \[0, 4294967295\]. Except for device ID 0, it cannot contain leading zeros. For example, device ID 1 cannot be written as `01`. The three configuration items of the same device use the same physical device ID.
- `udp_port_mode` is the UDP port configuration mode. Currently, only `multi_qp` is supported.
- `qp_count` is the QP count, configured as a decimal integer with a value range of \[1, 32\].
- `src_port` is the UDP source port, configured as a decimal integer with a value range of \[1, 65535\]. Ports 1 to 1023 are system-reserved ports and should be bypassed. Each NPU physical device must be configured with 1 to 32 UDP source ports, separated by commas (,), with no spaces inside the list.
- Each config occupies a single line. Empty lines and comment lines starting with `#` are ignored. Inline comments cannot be added after a config value.

The length of the UDP source port list can differ from the QP count. If the QP count is greater than the UDP source port count, ports are used cyclically starting from the first port in the list. If the QP count is less than the UDP source port count, only the first ports equal to the QP count are used.

### Configuration Example

Add the following content to `/etc/hcomm.cfg`:

```text
# NPU physical device 0
udp_port_mode_0=multi_qp
multi_qp_count_0=4
multi_qp_udp_ports_0=10000,10015,10030,10045

# NPU physical device 1
udp_port_mode_1=multi_qp
multi_qp_count_1=2
multi_qp_udp_ports_1=10016,10031
```

The preceding configuration indicates that NPU physical device 0 uses 4 QPs with UDP source ports 10000, 10015, 10030, and 10045 in sequence, and NPU physical device 1 uses 2 QPs with UDP source ports 10016 and 10031 in sequence.

### Constraints

- This configuration takes effect only in RoCE communication over host NICs.
- For each configured NPU physical device, `udp_port_mode`, `multi_qp_count`, and `multi_qp_udp_ports` must be provided together. If any device is missing a parameter, or it uses an invalid physical device ID format or value, communicator initialization fails and does not fall back to other configuration methods. If the same parameter is set repeatedly, its first configuration is used.
- If the file does not exist or cannot be opened, the file configuration is not used, and other configurations or the default method will be used. If the NPU physical device ID currently in use is not configured in the file, the device uses other configurations or the default method.
- In RoCE communication over host NICs, the QP count and the UDP source port list are selected in a descending order, and only the configuration matching the current device or IP pair is used.

    | Configuration | Order |
    | --- | --- |
    | QP count | QP count explicitly specified by the communication channel > `multi_qp_count` in `/etc/hcomm.cfg` > Port count in the file specified by [HCCL_RDMA_QP_PORT_CONFIG_PATH](HCCL_RDMA_QP_PORT_CONFIG_PATH.md) > [HCCL_RDMA_QPS_PER_CONNECTION](HCCL_RDMA_QPS_PER_CONNECTION.md) (default value: 1) |
    | UDP source port list | `multi_qp_udp_ports` in `/etc/hcomm.cfg` > `HCCL_HOST_RDMA_UDP_PORTS_LIST` > Port list in the file specified by [HCCL_RDMA_QP_PORT_CONFIG_PATH](HCCL_RDMA_QP_PORT_CONFIG_PATH.md) > Default method |

- The configuration is read when the configuration manager is initialized for the first time in the process and is cached after success. After modifying the file, restart the communication service process for the new configuration to take effect.

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
