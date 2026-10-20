# HCCL_HOST_RDMA_UDP_PORTS_LIST

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T03:53:06.687Z pushedAt=2026-10-15T08:46:37.177Z -->

## Function

In RoCE communication over host NICs, if you need to specify the UDP source ports used by multiple QPs, you can use this environment variable to configure a UDP source port list based on the NPU physical device ID.

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
- This environment variable has higher priority than [HCCL_RDMA_QP_PORT_CONFIG_PATH](HCCL_RDMA_QP_PORT_CONFIG_PATH.md): if this environment variable is configured, the UDP source port list in this environment variable is used; otherwise, the ports in the configuration file specified by HCCL_RDMA_QP_PORT_CONFIG_PATH are used.
- If the configuration format or value is invalid, communicator initialization fails.

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
