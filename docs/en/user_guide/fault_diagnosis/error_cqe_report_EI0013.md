# ERROR CQE Error Report (EI0013)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:52:23.926Z pushedAt=2026-10-13T09:33:09.590Z -->

ERROR CQE in HCCL indicates a retransmission timeout of RoCE packets. When this error occurs, it is inevitably accompanied by a cluster hang that leads to a timeout. HCCL periodically polls the RoCE driver to obtain its events. Users can query whether an ERROR CQE error report has occurred through the **HcclGetCommAsyncError** API.

## Symptoms

The error code EI0013 is printed in the console logs, with the keyword "Error ROCE CQE", as shown below:

```text
[PID: 3448331] 2025-12-04-21:59:08.232.310 Execution Error ROCE CQE(EI0013): An error CQE occurred during operator execution. Local information: server 127.0.0.1, device ID 0, device IP 127.10.0.1. Peer information: server 127.0.0.2, device ID 1, device IP 127.10.0.2.
Possible Cause: 1. The network between two devices is abnormal. For example, the network port is intermittently disconnected. 2. The peer process exits abnormally in advance. As a result, the local end cannot receive the response from the peer end.
Solution: 1. Check whether the network devices between the two ends are abnormal.2. Check whether the peer process exits first. If yes, check the cause of the process exit.
```

In addition, the keyword "error cqe status" appears in the CANN logs, as shown below:

```text
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.612 [hns_roce_lite.c:630]hns_roce_lite_handle_error_cqe(630) : error cqe status: 0x15
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.622 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000000): 0x00041580
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.627 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000001): 0x00000000
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.630 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000002): 0x00000000
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.634 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000003): 0x1500047c
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.637 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000004): 0x00000000
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.640 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000005): 0x00000000
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.644 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000006): 0x00000000
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.647 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000007): 0x00000000
[ERROR] HCCP(2040034,alltoall_test):2025-09-15-08:38:12.776.650 [ra_hdc_lite.c:794]tid:2040458,ra_hdc_lite_period_poll_cqe : [create][ra_hdc_period_poll]failed CQE status[12], wr[0]
[ERROR] HCCL(2040034,alltoall_test):2025-09-15-08:38:13.607.432 [heartbeat.cc:1229] [2040666][TaskExecStage][HeartbeatAbnormal][ROCE CQE ERROR]cqe error status[12], time:[2025-09-15 08:38:12.776654],localInfo{server[127.10.0.1],deviceId[127.10.0.1],deviceIp[127.11.0.1]}, remoteIP{server[127.10.0.2],deviceId[127.10.0.2],deviceIp[127.11.0.2]}
```

## Possible Causes

The root cause of an ERROR CQE event is that after the local end sends packets to the peer, it does not receive an acknowledgment reply from the peer within a specified time period, which triggers an ERROR CQE error report on the local end. This indicates that the network channel between the local end and the peer is abnormal, or the peer has disconnected, or the connection quality is poor and unable to respond. In addition to network factors, abnormal process exit on the peer side can also prevent the local end from receiving a reply, thereby causing an ERROR CQE error report.

## Solution

First, identify the remote end of the ERROR CQE based on the error information.

```text
[ERROR] HCCL(2040034,alltoall_test):2025-09-15-08:38:13.607.432 [heartbeat.cc:1229] [2040666][TaskExecStage][HeartbeatAbnormal][ROCE CQE ERROR]cqe error status[12], time:[2025-09-15 08:38:12.776654],localInfo{server[127.10.0.1],deviceId[127.10.0.1],deviceIp[127.11.0.1]}, remoteIP{server[127.10.0.2],deviceId[127.10.0.2],deviceIp[127.11.0.2]}
```

Here, localIP and remoteIP represent the device IPs of the local end and the remote end, respectively. Locate the compute node or log where the corresponding rank resides based on the hardware resource information.

1. Check for network issues. Use the hccn_tool to query whether there are any network port flapping records. The following result indicates that a port link-down event occurred at 10:13:50 2025. If a collective communication operator was being executed at that time, an ERROR CQE would be generated, and further investigation into the cause of the port flapping is required.

    ```bash
    $ hccn_tool -i 0 -link_stat -g
    [devid 0]current time        : Tue Oct 28 21:46:46 2025
    [devid 0]link up count       : 2
    [devid 0]link down count     : 1
    [devid 0]link change records :
    [devid 0]    Sun Oct  5 10:13:51 2025    LINK UP
    [devid 0]    Sun Oct  5 10:13:50 2025    LINK DOWN
    [devid 0]    Sun Oct  5 10:13:35 2025    LINK UP
    ```

2. Check whether the peer service process exited abnormally or had already entered the resource destruction process before the local ERROR CQE occurred. This can be determined by examining the peer's service logs or plog to confirm whether the abnormal exit time of the peer process precedes the local ERROR CQE occurrence.
3. If the HCCL_RDMA_TIMEOUT retransmission timeout and HCCL_RDMA_RETRY_CNT retry count configured in the environment variables are relatively small, ERROR CQE errors are likely to occur when the link status is poor. Increase these environment variable values accordingly.

Among them, status\[12\] indicates a RoCE packet retransmission timeout. Other status codes are extremely rare. If encountered, contact technical support.
