# Process Hang or Peer Heartbeat Loss

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:53:56.532Z pushedAt=2026-10-14T02:19:58.423Z -->

## Symptoms

The keyword "Cluster Exception Location" appears in CANN logs, as shown below:

Peer heartbeat loss:

```text
[ERROR]HCCL(835695,all_reduce_test):2025-10-23-17:28:06.049.385[task_exception_handler.cc:610][835695][TaskExecStage][HeartbeatAbnormal]Cluster Exception Location[IP/ID]:[127.10.0.1/1], Arrival Time:[Thu Oct 23 17:25:58 2025], Discoverer:[127.10.0.1/2], ExceptionType:[Heartbeat Lost Occurred], Possible Reason:1. Process has exited, 2. Network Disconnected
```

Process hang:

```text
[ERROR]HCCL(1219039,all_reduce_test):2025-10-23-21:05:09.859.568[task_exception_handler.cc:610] [1219039][TaskExecStage][HeartbeatAbnormal]Cluster Exception Location[IP/ID]:[127.10.0.1/1], Arrival Time:[Thu Oct 23 21:03:19 2025], ExceptionType:[Stuck Occurred], Possible Reason:1.Host process is stuck, 2.Device task is stuck
```

## Possible Causes and Troubleshooting Approach

The exception type and the node information where the exception occurred can be identified from the error log:

- Cluster Exception Location: Indicates the node information where the exception occurred.
- Arrival Time: Indicates the time when the exception was broadcast to the local end.
- ExceptionType: Exception type, including heartbeat loss (Heartbeat Lost Occurred), process hang (Stuck Occurred), network packet loss (Error CQE Occurred), and others.
- Possible Reason: Possible causes of the exception and troubleshooting approach:
  - Heartbeat Lost Occurred: Check whether the node where the exception occurred has already exited early before the exception was broadcast to the local end, or whether a network anomaly between nodes prevents connection.
  - Stuck Occurred: Check whether the service process on the node where the exception occurred is stuck on a certain node or has encountered a deadlock.
  - Error CQE Occurred: Check whether a CQE error has occurred on the node where the exception occurred.
