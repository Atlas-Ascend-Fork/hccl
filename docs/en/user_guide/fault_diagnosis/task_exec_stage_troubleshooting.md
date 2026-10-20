# Troubleshooting Approach

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:55:20.271Z pushedAt=2026-10-14T03:06:51.308Z -->

After communicator initialization and parameter plane link setup are completed, HCCL performs task orchestration and dispatch of communication operators. During communication operator task orchestration, a notify synchronization mechanism is used before data communication to ensure that the peer is ready to receive data from the local end. Therefore, if any rank experiences a process hang or exit due to an exception, a network fault, or inconsistent communication operator calls, most ranks will encounter execution wait timeout. When such issues occur, the primary task for troubleshooting is to locate the fault point. The overall troubleshooting approach is shown in the following figure.

![Troubleshooting Approach for Task Dispatch and Execution Phase Errors](figures/task_exec_error_debug.png)

During the communication operator task dispatch and execution phase, HCCL provides the following DFX mechanisms to facilitate rapid issue locating:

- HCCL provides a cluster heartbeat mechanism. When a rank node detects an exception, the exception is propagated to every node in the cluster through the heartbeat mechanism. Therefore, you can first search the CANN logs of any node in the cluster for printed heartbeat exception event information. For details about the mechanism and log information, see [Cluster Heartbeat Mechanism](./_dump_cluster_heartbeat_mechanism.md#cluster-heartbeat-mechanism).
- If no heartbeat exception event information is found in the logs, you can use the task exception error information to check whether inconsistent cluster behavior exists. For the troubleshooting method, see [Task Exception Mechanism](./_dump_task_exception_mechanism.md#task-exception-mechanism).
