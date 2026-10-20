# Troubleshooting Approach

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:51:54.342Z pushedAt=2026-10-13T08:50:04.902Z -->

HCCL establishes independent maintenance and diagnostic links with adjacent ranks based on existing communicator information, thereby providing the cluster with single-point fault broadcast and diffusion capabilities. (Note: HCCL controls the number of maintenance and diagnostic links and the communication data volume, so users do not need to worry about performance loss on communication links.) This ensures that the plog of any rank contains the fault root node information. For the currently supported fault detection capabilities, see the following table.

| Priority | Exception Type | Status (Run Log - run/plog) | ExceptionType (Debug Log - debug/plog) | Criteria |
| --- | --- | --- | --- | --- |
| 1 | Network issue | ERROR CQE | Error cqe Occurred | Periodically polls ROCE driver retransmission timeout events and maps the remote IP through QPN. |
| 2 | Process hang | STUCK | Stuck Occurred | Polls the ingress/egress counts of all operators at intervals of 1/3 of HCCL_EXEC_TIMEOUT to analyze whether a hang has occurred. |
| 3 | Process exit | LOST | Heartbeat Lost Occurred | No heartbeat packet received from the remote end within 30 seconds. |

To control the number of printed entries, the current Cluster Exception ERROR log prints only the first three valid events, with the priority order being ERROR CQE \> STUCK \> LOST. If all heartbeat events need to be confirmed, search the logs in the run directory.

- After detecting an exception event, HCCL diffuses and forwards the information within the cluster. During operation, HCCL prints the received exception events to the run log.

    **The log format is:** [HeartbeatAbnormal]local rank[IP/ID]:crimer rank[IP/ID] status[4exception event type]by informer rank[IP/ID].

  - HeartbeatAbnormal: Represents a heartbeat exception event.
  - local rank: Information about the current node.
  - crimer rank: Information about the root node.
  - status: Exception event type.
  - by informer rank: The reporter of the cluster fault.

    You can search using the keyword "HeartbeatAbnormal" combined with the status value. The following shows a log example:

    ```text
    [INFO] HCCL(686,python):2025-10-23-07:52:59.191.363 [heartbeat.cc:951] [8970][TaskExecStage][HeartbeatAbnormal]local rank [127.10.0.1/1]: crimer rank [127.10.0.2/2] status[LOST] by informer rank [127.10.0.3/3]
    ```

- If an operator execution error report occurs subsequently and the task exception callback function is invoked to notify HCCL, HCCL infers the most likely single point of failure based on the received exception events and timeout configurations such as HCCL_EXEC_TIMEOUT, and prints the result in the ERROR log.

    **The log format is**: \[TaskExecStage\]\[HeartbeatAbnormal\]Cluster Exception Location\[IP/ID\], Arrival Time:\[Day Mon DD HH:MM:SS YYYY\], Discoverer:\[IP/ID\], ExceptionType:\[Exception Type\], Possible Reason:possible cause.

  - \[TaskExecStage\]\[HeartbeatAbnormal\]: indicates that the cluster fault occurred during the operator execution stage and is a heartbeat exception event.
  - Cluster Exception Location: Location where the cluster fault occurred.
  - Arrival Time: Time when the cluster fault occurred.
  - Discoverer: Node that discovered the cluster fault.
  - ExceptionType: Exception type of the cluster fault.
  - Possible Reason: Possible cause of the cluster fault.

    Users can search for the keyword "HeartbeatAbnormal". A log example is as follows:

    ```text
    [ERROR]HCCL(835695,all_reduce_test):2025-10-23-17:28:06.049.385[task_exception_handler.cc:610][835695][TaskExecStage][HeartbeatAbnormal]Cluster Exception Location[IP/ID]:[127.10.0.1/1], Arrival Time:[Thu Oct 23 17:25:58 2025], Discoverer:[127.10.0.1/2], ExceptionType:[Heartbeat Lost Occurred], Possible Reason:1. Process has exited, 2. Network Disconnected
    ```

If no exception event accompanies the timeout, the issue may be caused by cluster behavior consistency problems. Prioritize troubleshooting factors such as scripts, versions, and datasets. If necessary, enable the HCCL_ENTRY_LOG_ENABLE environment variable to perform operator-level behavior tracking.

> [!NOTE]Note
>
> 1. If the training/inference task is killed before the notify timeout, or the task exception mechanism fails to call the callback function to notify HCCL in time for some reason, HCCL does not print the exception information. Users can still locate the root node through the abnormal events during system running in the run log. In this case, the abnormal events need to be identified. Generally, the LOST/ERROR CQE events near the time when the system is stuck are considered the cause of the system stop, and the STUCK detection time is (1/3\~2/3 )\* HCCL_EXEC_TIMEOUT, which requires attention.
> 2. Both network exceptions and process exits may cause LOST and ERROR CQE events at the same time. Analyze them in combination with the specific heartbeat events. For example, the ranks at both ends report LOST to each other.
