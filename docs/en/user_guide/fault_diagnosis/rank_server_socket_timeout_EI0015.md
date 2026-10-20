# Rank and Server Node Socket Establishment Timeout (EI0015)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:54:00.929Z pushedAt=2026-10-14T02:25:37.502Z -->

## Symptoms

The keyword "topo exchange agent get socket timeout!" is found in the CANN log, as shown below:

```text
[ERROR] HCCL(7988,all_reduce_test):2025-03-19-04:16:13.978.979 [topoinfo_exchange_agent.cc:190] [7988][InitGroupStage][RanktableDetect]topo exchange agent get socket timeout! timeout[120] 
[ERROR] HCCL(7988,all_reduce_test):2025-03-19-04:16:13.978.995 [topoinfo_exchange_agent.cc:41] [7988][TopoInfoExchangeAgent][Setup]TopoExchangeAgent: connect server[127.10.0.1 : 60000] failed
```

## Possible Causes

During the communicator creation phase through cluster negotiation, the current rank creates a socket with the IP address and port of the server node based on the rootInfo information. However, the socket establishment times out due to network connectivity issues.

## Solution

1. Check the connectivity of the host-side network and the corresponding port.

    Since the host side often has multiple network interface cards, HCCL selects a host NIC in lexicographical order for socket connection by default. As a result, an unreachable host NIC may be selected. You can specify the NIC to be used through the [HCCL_SOCKET_IFNAME](../hccl_env/HCCL_SOCKET_IFNAME.md) environment variable. If the NIC is correctly selected, further check whether the specified port is reachable. For example, based on the following log, check whether the faulty node can reach port 60000 of 127.10.0.1. You can run the following command to query the host NIC information obtained by the current process:

    ```bash
    grep -r "get host ip success\|find nic.*success"
    ```

2. Check whether the interval between the communicator creation delivery time of each agent and the server node within the communicator exceeds the timeout.

    In the run directory of the CANN log, run `grep -r "Entry-"` to confirm the delivery time of the communicator creation interface, or directly calculate the interval between the communicator creation delivery time of each agent and the server node based on the timestamp in the error log. If the interval exceeds the timeout, configure the connection establishment timeout through [HCCL_CONNECT_TIMEOUT](../hccl_env/HCCL_CONNECT_TIMEOUT.md), which defaults to 120 seconds.

    The time when each communicator creation interface is delivered can be queried using the following command:

    ```bash
    grep -r "Entry-HcclGetRootInfo\|Entry-HcclCommInitRootInfoInner" run/plog
    ```

    Based on the following query results, it can be seen that the communicator interface delivery time for rank\[3\] is approximately 200 seconds slower than that of other ranks, while the connection establishment timeout is 120 seconds. Therefore, the entire communicator creation process eventually failed due to a timeout. For this scenario, if there are preceding differences in service processes on different ranks, for example, some processes need to load more data and therefore start more slowly, this error can be resolved by increasing the timeout through the [HCCL_CONNECT_TIMEOUT](../hccl_env/HCCL_CONNECT_TIMEOUT.md) environment variable.

    ```text
    [INFO] HCCL(3079955,all_reduce_test):2025-11-20-11:59:56.716.583 [op_base.cc:1293] [3079955]Entry-HcclCommInitRootInfoInner:ranks[4], rank[3], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1763610996711234], deviceLogicId[3]
    [INFO] HCCL(3079952,all_reduce_test):2025-11-20-11:56:36.704.523 [op_base.cc:858] [3079952]Entry-HcclGetRootInfo:rootInfo[0xaaaae85c79a0], deviceLogicId[0]
    [INFO] HCCL(3079952,all_reduce_test):2025-11-20-11:56:36.711.546 [op_base.cc:1293] [3079952]Entry-HcclCommInitRootInfoInner:ranks[4], rank[0], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[9127.10.0.1%enp_60000_0_1763610996711234], deviceLogicId[0]
    [INFO] HCCL(3079953,all_reduce_test):2025-11-20-11:56:36.712.024 [op_base.cc:1293] [3079953]Entry-HcclCommInitRootInfoInner:ranks[4], rank[1], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1763610996711234], deviceLogicId[1]
    [INFO] HCCL(3079954,all_reduce_test):2025-11-20-11:56:36.712.065 [op_base.cc:1293] [3079954]Entry-HcclCommInitRootInfoInner:ranks[4], rank[2], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1763610996711234], deviceLogicId[2]
    ```
