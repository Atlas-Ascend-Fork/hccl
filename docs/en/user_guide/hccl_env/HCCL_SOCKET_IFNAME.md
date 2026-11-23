# HCCL_SOCKET_IFNAME

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-06T09:49:10.711Z pushedAt=2026-11-16T06:28:10.218Z nl=2 -->

## Function

Configures the name of the communication NIC used by the host during HCCL initialization. HCCL obtains the host IP through this NIC name and communicates with the root node to complete communicator creation.

You can choose one of the following rules for configuration:

- `eth`: Use all NICs prefixed with eth.

    If multiple NIC prefixes are specified, separate them with commas.

    For example, `export HCCL_SOCKET_IFNAME=eth,enp` indicates that all NICs with the prefix `eth` or `enp` are used.

- `^eth`: Do not use NICs prefixed with eth.

    If multiple NIC prefixes are specified, separate them with commas.

    For example, `export HCCL_SOCKET_IFNAME=^eth,enp` indicates that no NICs with the prefix `eth` or `enp` are used.

- `=eth0`: Use the specified eth0 NIC.

    If multiple NICs are specified, separate them with commas.

    For example, `export HCCL_SOCKET_IFNAME==eth0,enp0` indicates using the eth0 NIC or the enp0 NIC.

- `^=eth0`: Do not use the specified eth0 NIC.

    If multiple NICs are specified, separate them with commas.

    For example, `export HCCL_SOCKET_IFNAME=^=eth0,enp0` indicates not using the eth0 and enp0 NICs.

> [!NOTE]
>
> - You can configure multiple NICs in HCCL_SOCKET_IFNAME. The first matched NIC is used as the communication NIC.
> - The [HCCL_IF_IP](HCCL_IF_IP.md) environment variable takes precedence over HCCL_SOCKET_IFNAME.
> - If neither HCCL_IF_IP nor HCCL_SOCKET_IFNAME is specified, the NIC is selected in the following priority order:
>    NICs other than docker/lo (in ascending lexicographic order of NIC names) \> docker NIC \> lo NIC
>
> If HCCL_IF_IP or HCCL_SOCKET_IFNAME is not configured, the system automatically selects a NIC based on the priority. If the NIC selected on the current node cannot communicate with the NIC selected on the root node, HCCL link establishment fails.

## Configuration Example

```bash
# Use the eth0 or endvnic NIC
export HCCL_SOCKET_IFNAME==eth0,endvnic
```

## Constraints

None.

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
