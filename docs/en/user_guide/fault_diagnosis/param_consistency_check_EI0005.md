# Parameter Consistency Check (EI0005)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:53:12.566Z pushedAt=2026-10-13T10:55:38.297Z -->

## Symptoms

The console log contains the keyword "The arguments for collective communication are inconsistent between ranks", as shown below:

```text
EI0005: 2024-04-24-06:32:27.781.599 The arguments for collective communication are inconsistent between ranks:parameter count, local end 16512, remote end 8320
        TraceBack (most recent call last):
        Transport init error. Reason: [Create] [DestLink]Create Dest error! createLink para:rank[5]-localUserrank[4]-localIpAddr[127.10.0.1], dst_rank[6]-remoteUserrank[7]-remote_ip_addr[127.10.0.1]
        Transport init error. Reason: [Create] [DestLink]Create Dest error! createLink para:rank[5]-localUserrank[4]-localIpAddr[127.10.0.1], dst_rank[4]-remoteUserrank[5]-remote_ip_addr[127.10.0.1]
        call hccl op:HcomAllReduce(HcomAllReduce) load task fail[FUNC:Distribute][FILE:hccl_task_info.cc] [LINE:329]
        [[{[node Ge0p3_0]}]]
```

Or the CANN log contains the keyword "CMD information *** check fail", as shown below:

```text
[ERROR] HCCL(3743927,all_reduce_test):2025-10-25-16:11:16.831.640 [rank_consistentcy_checker.cc:429] [3743951][InitChannelStage][ParameterConflict]CMD information tag check fail. local[AllGather_127.10.0.1%enp_60000_0_1761379874757928], remote[AllReduce_127.10.0.1%enp_60000_0_1761379874757928]
[ERROR] HCCL(3743927,all_reduce_test):2025-10-25-16:11:16.831.666 [rank_consistentcy_checker.cc:439] [3743951][InitChannelStage][ParameterConflict]CMD information cmdType check fail. local[6], remote[2]
[ERROR] HCCL(3743927,all_reduce_test):2025-10-25-16:11:16.831.679 [rank_consistentcy_checker.cc:439] [3743951][InitChannelStage][ParameterConflict]CMD information op check fail. local[255], remote[0]
```

## Possible Causes

During parameter plane link setup, after the socket is established, a parameter consistency check is performed between the two endpoints. The check covers the operator identifier tag, operator type cmdType, reduction type op, data volume count, HCCL buffer size cclbufferSize, data type dataType, and other parameters. The inconsistent data can be identified based on the information in the error message. For example, in the following case, the operator identifier tags on the two endpoints are inconsistent, causing the communication operator to fail the consistency check during link setup. The data in local and remote represent the inconsistent values on the two endpoints.

The node information of the two endpoints with inconsistent parameters can be confirmed through the "Transport init error! createLink para:" error log. For example, execute `grep -r "Transport init error! createLink para:"` and the result is as follows:

```text
[ERROR] HCCL(3215542,all_reduce_test):2025-11-20-18:18:03.114.306 [transport_manager.cc:886] [3215599][InitChannelStage][Timeout]Transport init error! createLink para:rank[2]-localUserrank[2]-localIpAddr[127.10.0.1/2], remoteRank[1]-remoteUserrank[1]-remoteIpAddr[127.10.0.1/1], machineType[1], linkMode[1], isUsedRdma[0], tag[AllReduce_127.10.0.1%enp_60000_0_1763633852475745
```

- localUserrank: the rank number of the local endpoint.
- localIpAddr: the node IP information of the local endpoint.
- remoteUserrank: Rank number of the peer.
- remoteIpAddr: Node IP information of the peer.
- tag: Communication operator identifier.

## Solution

1. If the functionality works properly when SuperKernel is not enabled but an initialization inconsistency occurs after SuperKernel is enabled, it is recommended to move the HCCL operator out of the SuperKernel calibration scope. For specific instructions, see the "max-autotune Mode Features > Calibrating the SuperKernel Scope Within a Graph" section in [*PyTorch Graph Mode Usage Guide*](https://hiascend.com/document/redirect/pttorchairuseguide).
2. Based on the error information, troubleshoot the root cause of the inconsistency between the operators delivered by the two ends that failed the parameter consistency check from the service perspective.

    **NOTE** Some values printed in the log are enumeration values, where cmdType indicates the operator type and op indicates the reduction type. The mapping of enumeration values is shown in the following table:

    | cmdType Enumeration Value | Operator Type |
    | --- | --- |
    | 1 | BroadCast |
    | 2 | AllReduce |
    | 3 | Reduce |
    | 4 | Send |
    | 5 | Receive |
    | 6 | AllGather |
    | 7 | ReduceScatter |
    | 8 | AlltoAllV |
    | 9 | AlltoAllVC |
    | 10 | AlltoAll |
    | 11 | Gather |
    | 12 | Scatter |
    | 13 | BatchSendRecv |
    | 16 | AllGatherV |
    | 17 | ReduceScatterV |

    The reduction types corresponding to the op enumeration values are shown in the following table:

    | op Enumeration Value | Reduction Type |
    | --- | --- |
    | 0 | SUM |
    | 1 | PROD |
    | 2 | MAX |
    | 3 | MIN |
    | 255 | Non-Reduce Operator |
