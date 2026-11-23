# Communicator Link Setup Timeout Background

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-09T03:33:42.884Z pushedAt=2026-11-12T08:33:33.795Z nl=2 -->

During cluster information negotiation (the scenario where a communicator is created based on root node information), HCCL establishes socket connections between the server node and each rank node to exchange local information, thereby obtaining the cluster information of the entire communicator and completing communicator initialization.

![image](figures/topo_detect_theory.png)

1. The root node in the communicator calls HcclGetRootInfo, which starts a server thread that monitors a port and waits to receive messages from each rank in the communicator. At the same time, this API returns a rootInfo variable that represents the server's IP and port, which is then passed to the upper-layer framework to broadcast rootInfo to each rank in the communicator.
2. Each rank node in the communicator calls HcclCommInitRootInfo with rootInfo as an input parameter. This API starts a client thread that establishes a socket connection with the server through the **host NIC**, sends its own rankInfo to the server, and then enters the receiving state to wait for the server to return the complete rank table.
3. After collecting the rankInfo from all ranks, the server generates the complete rank table and sends it to each client, so that each rank has the complete rank information of the entire communicator.

Because the topology detection method requires establishing socket connections between ranks, the **host-side NIC** is used for socket connection and communication, and all ranks in the communicator must execute synchronously within the timeout duration. **Therefore, if the interconnected host-side NIC is not selected correctly, inconsistent delivery behavior within the communicator is a common cause of timeout failures.**

# Communicator Initialization Topo Detection Timeout (EI0015)

## Timeout Issue Quick Reference Table<a id="timeout-issue-quick-reference-table"></a>

When the error log contains the `topoinfo_exchange_server.cc` or `topoinfo_exchange_agent.cc` field, the system is in the communicator creation phase. Based on the error log, you can jump to the corresponding section for troubleshooting:

| Common Keywords / Clues                                | Troubleshooting Section                                   | Description                          |
| ----------------------------------------- | -------------------------------------- | --------------------------- |
| `topo exchange server get socket timeout` | [Some Ranks Not Connected to the Server Node](#some-ranks-not-connected-to-the-server-node)        | The server waits for all ranks to connect via sockets.     |
| `Wait timeout for sockets recv`           | [Client Receiving Rank Table Timeout](#client-receiving-rank-table-timeout)       | The client times out while waiting for the server to send the rank table. |
| `topo exchange agent get socket timeout`  | [Rank-Server Socket Establishment Timeout](#rank-server-socket-establishment-timeout) | The client times out while connecting to the server socket.     |

**Error example:**

```text
Server node:
[ERROR] HCCL(340,python3.9):2026-07-15-19:50:49.176.080 [topoinfo_exchange_server.cc:314] [5590][Get][Connection]topo exchange server get socket timeout! timeout[3600 s]

Client node:
[ERROR] HCCL(347,python3.9):2026-07-15-19:50:49.253.479 [topoinfo_exchange_base.cc:75] [347][Recv][ClusterInfoMsg]receive msg length from fdhandle failed, ret[9]
[ERROR] HCCL(347,python3.9):2026-07-15-19:50:49.253.490 [topoinfo_exchange_agent.cc:161] [347][DetectClusterTopoInfo]call trace: hcclRet -> 4
```

## Troubleshooting Steps

First, locate the server node. After calling the HcclGetRootInfo API, the server node starts a background thread that waits for all ranks to connect within the configured timeout duration. Therefore, if all ranks in the communicator fail to connect to the server thread within the timeout duration, the server thread reports a timeout error. In addition, after reporting the timeout error, the server thread prints the list of currently connected ranks. Based on this information, identify the ranks that failed to connect, and then further troubleshoot why the corresponding ranks failed to connect.

### Some Ranks Not Connected to the Server Node<a id="some-ranks-not-connected-to-the-server-node"></a>

When some ranks are not connected to the server node, the server node waits for the agent node socket connection to time out. The problem symptom is as follows:

```text
[ERROR] HCCL(340,python3.9):2026-07-15-19:50:49.176.080 [topoinfo_exchange_server.cc:314] [5590][Get][Connection]topo exchange server get socket timeout! timeout[3600 s] 
[ERROR] HCCL(340,python3.9):2026-07-15-19:50:49.176.770 [topoinfo_exchange_server.cc:499] [5590][TopoInfoExchangeServer][DisplayConnectionedRank]total connected num is [2004],line num is [251] [ERROR] HCCL(340,python3.9):2026-07-15-19:50:49.176.782 [topoinfo_exchange_server.cc:512] [5590][TopoInfoExchangeServer][DisplayConnectionedRank]connected rankinfo[LINE 0]: [0000000000000000],[0000000000000001],[0000000000000002],[0000000000000003],[0000000000000004],[0000000000000005],[0000000000000006],[0000000000000007]; 
... ...
[ERROR] HCCL(340,python3.9):2026-07-15-19:50:49.231.208 [op_base.cc:1150] [340][InitCommRootInfo]Init failed, return[0x0000000005000004], rankNum[2048], rank[0], rootInfo identifier[group_name_3], server[172.16.154.221%eth0], logicDevId[-1]
 ... ... 
[ERROR] HCCL(340,python3.9):2026-07-15-19:50:49.450.492 [topoinfo_exchange_server.cc:115] [5590][TopoInfoExchangeServer][Setup]Broadcast Rank Basic Infos failed,connectFailedAgentIdList[10,17,40,79,119,310,358,682,755,812,850,860,865,925,961,989,1025,1054,1077,1164,1170,1178,1203,1224,1341,1375,1422,1469,1562,1563,1583,1629,1634,1637,1641,1650,1738,1775,1829,1840,1866,1951,1975,2007,]
```

#### Confirming the Unconnected Rank<a id="confirming-the-unconnected-rank"></a>

Find the rank that failed to connect from the error log. Take the sample log as an example:
connectFailedAgentIdList: the list of unconnected rank IDs.
connected rankinfo: the detailed IDs of each connected rank. In the example, `[0000000000000003]` indicates that rank3 has connected successfully.
In the example, `connectFailedAgentIdList[10,17,40,79,119, ... ...]`, rank10 is used as the entry point to analyze why it failed to connect.

#### Locating the Rank That Failed to Connect<a id="locating-the-rank-that-failed-to-connect"></a>

Locate the log position of the rank that failed to connect, using rank10 as an example:
First, find the communicator that reported the error from the error log. Using the sample log as an example: `identifier` indicates the communicator name, and in the sample, the communicator name is `group_name_3`.
Then, based on the communicator name and rankID, check whether the corresponding rank has a communicator creation delivery record. You can search the **entire cluster's logs** by executing the following command to find the corresponding rank log path:
`grep -rn "xxx" plog/ | grep -rn "Entry-HcclCommInit" | grep -rn "rank\[10\]"`, where `xxx` is the communicator name and `plog/` is the cluster plog log directory.
If the communicator creation API delivery log for the corresponding rank is not found in the cluster logs, **you need to troubleshoot from the service side why the rank was not delivered**.

#### Troubleshooting the Communicator Initialization Time Difference Between the Failed Rank and the Server Node

If the corresponding rank has delivered the communicator creation API, a log record appears in the run/plog log of that node:

```text
[INFO] HCCL(343,python3.9):2026-07-15-18:50:49.284.233 [op_base.cc:1281] [343]Entry-HcclCommInitRootInfoConfigInner:ranks[2048], rank[10], rootinfo: host ip[172.16.154.221] port[64000] nicDeploy[1] identifier[group_name_3], deviceLogicId[2]
```

The communicator delivery time point of the client node is obtained: `2026-07-15-18:50:49.284.233`.

1. Confirm the time when the server node delivers the communicator creation.

   - Confirm the log location of the server node. Generally, rank0 is the server node: `grep -rn "xxx" plog/ | grep -rn "Entry-HcclCommInit" | grep -rn "rank\[0\]"`, where `xxx` is the communicator name and `plog/` is the cluster plog log directory.

      ```text
      hcclLog/run/plog/plog-340_20260715182300676.log:17282:[INFO] HCCL(340,python3.9):2026-07-15-18:50:49.175.969 [op_base.cc:1281] [340]Entry-HcclCommInitRootInfoConfigInner:ranks[2048], rank[0], rootinfo: host ip[172.16.154.221] port[64000] nicDeploy[1] identifier[group_name_3], deviceLogicId[0]
      ```

   - Find the time point when `HcclGetRootInfo` is called in the server node log path: execute `grep -rn "Entry-HcclGetRootInfo" xxxxx`, where `xxxxx` is the log path found in the previous step:
     Multiple records may be found. Take the timestamp of the last record, `2026-07-15-18:50:49.174.231` (the communicator creation API is serial, so the current communicator is the last communicator).

      ```text
      # grep -rn "Entry-HcclGetRootInfo" hcclLog/run/plog/plog-340_20260715182300676.log
      8849:[INFO] HCCL(340,python3.9):2026-07-15-18:49:48.236.358 [op_base.cc:806] [340]Entry-HcclGetRootInfo:rootInfo[0xffffe8730d48], deviceLogicId[0]
      8913:[INFO] HCCL(340,python3.9):2026-07-15-18:50:30.271.209 [op_base.cc:806] [340]Entry-HcclGetRootInfo:rootInfo[0xffffe8731048], deviceLogicId[0]
      17271:[INFO] HCCL(340,python3.9):2026-07-15-18:50:49.174.231 [op_base.cc:806] [340]Entry-HcclGetRootInfo:rootInfo[0xffffe8731068], deviceLogicId[0]
      ```

2. Check whether the time interval between the communicator creation delivery of this rank and that of the server node exceeds the timeout duration.

   - Obtain the timeout duration for communicator link setup by searching the run log: `grep -rn "HCCL_CONNECT_TIMEOUT" run/plog`. In this example, the communicator delivery time point of the client node is `2026-07-15-18:50:49.284.233`, and that of the server node is `2026-07-15-18:50:49.174.231`. The delivery time difference is less than the configured timeout duration of 3600s.
     If the delivery time difference of the communicator creation API is greater than the timeout duration, **you need to troubleshoot the cause of the excessive time difference of the communicator creation API from the service side, or adjust the [HCCL_CONNECT_TIMEOUT](../hccl_env/HCCL_CONNECT_TIMEOUT.md) environment variable configuration to increase the timeout duration to resolve this error**.

      ```text
      [INFO] HCCL(353,python3.9):2026-07-15-18:41:23.173.030 [externalinput.cc:382] [353]HCCL_CONNECT_TIMEOUT set by environment to [3600]s
      ```

#### Troubleshooting the Cause of the Error Reported by the Failed Rank

When the third-step troubleshooting is normal, locate the debug log based on [the run log location of the rank](#locating-the-rank-that-failed-to-connect). For the relationship between the debug log and the run log, see [reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/logreference/logreference_0002.html). Further [troubleshoot](#timeout-issue-quick-reference-table) based on the error symptom in the debug log. The correspondence between the run log and the debug log is as follows:

```text
├── debug
│   ├── device-1
│   │   ├── device-2849436_20260722090447711.log
│   └── plog
│       ├── plog-2849436_20260722090443026.log
├── run
│   ├── device-1
│   │   ├── device-2849436_20260722090449878.log
│   └── plog
│       ├── plog-2849436_20260722090443227.log
```

### Rank-Server Socket Establishment Timeout<a id="rank-server-socket-establishment-timeout"></a>

The problem symptom of a rank-server socket establishment timeout is as follows:

```text
[ERROR] HCCL(7988,all_reduce_test):2025-03-19-04:16:13.978.979 [topoinfo_exchange_agent.cc:190] [7988][Get][Connection]topo exchange agent get socket timeout! timeout[120]
[ERROR] HCCL(7988,all_reduce_test):2025-03-19-04:16:13.978.995 [topoinfo_exchange_agent.cc:41] [7988][TopoInfoExchangeAgent][Setup]TopoExchangeAgent: connect server[172.16.0.0 : 1888] failed
```

1. In a multi-server scenario, check whether the host NIC to be used is specified through the `HCCL_SOCKET_IFNAME` environment variable. If multiple host NICs exist in the environment, HCCL selects a NIC in lexicographic order by default for socket connection establishment, which may easily select a disconnected host NIC. Run `ifconfig` to view the name of the interconnected host NIC, and configure the `HCCL_SOCKET_IFNAME` environment variable to specify the host NIC, then restart the job.

2. If the NIC is correctly selected, check whether the specified port is reachable. For example, based on the log `TopoExchangeAgent: connect server`, check whether the node reporting the error can connect to port 1888 of 172.16.0.0 (if `ping 172.16.0.0` is used, port 22 is used by default).

### Client Receiving Rank Table Timeout<a id="client-receiving-rank-table-timeout"></a>

When the client times out while receiving the rank table, the problem symptom is as follows:

```text
[ERROR] HCCL(347,python3.9):2026-07-15-19:50:49.253.079 [adapter_hccp.cc:1389] [347][Recv][RaSocket]errNo[0x0000000005000013] Wait timeout for sockets recv, data[0xffffeb38eb60], size[4 Byte], recvSize[0 Byte]. Peerrank did not send the data in time. Check whether the peerrank is abnormal.
[ERROR] HCCL(347,python3.9):2026-07-15-19:50:49.253.471 [hccl_socket.cc:397] [347][Recv]call trace: hcclRet -> 9
[ERROR] HCCL(347,python3.9):2026-07-15-19:50:49.253.479 [topoinfo_exchange_base.cc:75] [347][Recv][ClusterInfoMsg]receive msg length from fdhandle failed, ret[9]
... ...
[ERROR] HCCL(347,python3.9):2026-07-15-19:50:49.304.157 [op_base.cc:1150] [347][InitCommRootInfo]Init failed, return[0x0000000005000004], rankNum[2048], rank[1132], rootInfo identifier[group_name_3], server[0.0.0.0], logicDevId[-1]
```

#### Confirming the Server Node

Based on the `identifier` field of the failed communicator in the error log, locate the server node of the communicator. Generally, rank0 is the server node of the communicator.
Run the following command in the full cluster logs to search for the run log path of the server node. In the command, `group_name_3` is the communicator name in this example:

```text
grep -rn "group_name_3" | grep -rn "Entry-HcclCommInit" | grep -rn "rank\[0\]"
```

Locate the debug log based on the run log path, and then perform further troubleshooting based on the error in the debug log.

```text
Correspondence between the run log and the debug log:
├── debug
│   ├── device-1
│   │   ├── device-2849436_20260722090447711.log
│   └── plog
│       ├── plog-2849436_20260722090443026.log
├── run
│   ├── device-1
│   │   ├── device-2849436_20260722090449878.log
│   └── plog
│       ├── plog-2849436_20260722090443227.log
```

#### Troubleshooting Server Node Errors

1. Based on the error information on the server node, [confirm the unconnected rank](#confirming-the-unconnected-rank).
2. If the unconnected ranks on the server do not include this rank, the local end has connected normally, and you need to troubleshoot the other ranks that failed to connect.
3. If the unconnected ranks on the server include the local end, the socket connection between the local end and the server is abnormal. This is usually caused by an exception on the host-side network, such as ARP drift or port interception. You need to further check whether there is any abnormal configuration on the host-side network.
   In large-scale cluster scenarios, the number of concurrent link setups that the Master node can handle is limited by the Linux kernel parameters `somaxconn` and `tcp_max_syn_backlog`. Therefore, if the values of these two parameters are too small, some clients may exit abnormally ahead of time with a certain probability during link setup, which in turn causes cluster initialization to fail. You can adjust the connection number limit by using the following configuration (the OS of all machines must be configured, including bare-metal and image environments). Usually, set the value to `65535`:

   ```text
   sysctl -w net.core.somaxconn=65535
   sysctl -w net.ipv4.tcp_max_syn_backlog=65535
   ```
