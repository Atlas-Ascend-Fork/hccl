# Troubleshooting Approach

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:55:21.471Z pushedAt=2026-10-14T03:03:40.001Z -->

After the task orchestration of an HCCL communication operator is complete, the tasks are delivered to the device side for asynchronous execution. If a task delivered by HCCL fails to execute, HCCL is notified of the abnormal task information (stream and taskId) by calling the callback function. HCCL then uses this information to retrieve the task information at the time of delivery and prints the detailed information of the failed task and the operator to which it belongs.

For the following products, to track task-level information, manually enable it through [HCCL_DIAGNOSE_ENABLE](../hccl_env/HCCL_DIAGNOSE_ENABLE.md).

<!-- npu="A3" id4 -->
- Atlas A3 products
<!-- end id4 -->
<!-- npu="910b" id1 -->
- Atlas A2 products
<!-- end id1 -->

In this case, the key log entries for task exception printed in the CANN log are **"Task run failed"** or **"TaskExecStage"**, as shown below:

```text
[ERROR] HCCL(2111667,all_reduce_test):2025-10-24-11:18:29.597.044 [task_exception_handler.cc:908] [2111667][TaskExecStage][Timeout][Host]Task run failed, base information is streamID:[2], taskID[21], tag[AllReduce_127.10.0.1%enp_60000_0_1761275812718970], AlgType(level 0-1-2):[fullmesh-ring-NHR].
[ERROR] HCCL(2111667,all_reduce_test):2025-10-24-11:18:29.597.054 [task_exception_handler.cc:771] [2111667][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[127.10.0.1%enp_60000_0_1761275812718970], user define information[], rankSize[4], rankId[2].
[ERROR] HCCL(2111667,all_reduce_test):2025-10-24-11:18:29.597.083 [task_exception_handler.cc:704] [2111667][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-10-24-11:16:55.490.253], deviceId[2], index[21], count[256], reduceType[sum], src[0x12c0c0013000], dst[0x12c0c0014000], dataType[float32].
```

First, the key information of the communication operator can be identified from the task exception information:

- base information: The stream and task ID where the HCCL operator resides, as well as the operator tag. The HCCL operator that reported the error can be identified based on the tag.
- groupRank information: The communicator name (group), the size of the communicator (rankSize), and the rankId of the current card within the communicator.
- opData information: The input parameter information of the current operator, including the deviceId where it resides, the index of the operator within the communicator (index), the data volume (count), the reduce type (reduceType), and the addresses of the source (src) and destination (dst).

Generally, only the following two types of tasks may fail:

- Notify: Commonly occurs when waiting for a remote peer times out during the operator execution phase.
- SDMA: Typically occurs in scenarios such as HCCS link anomalies or multi-bit ECC errors, and may also be triggered with a low probability during a remote core dump.
