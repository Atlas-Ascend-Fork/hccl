# Background of Operator Execution Timeout

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:53:26.982Z pushedAt=2026-10-13T10:55:21.854Z -->

A single inter-card data transfer consists of three phases: **pre-sync, data transfer, and post-sync** (using SDMA as an example).

![op_execute_sequence](figures/op_execute_sequence.png)

* **Pre-sync (Post / Wait Ack)**: Before Rank0 reads data from Rank1, it must wait for Rank1 to send an Ack, indicating that the data is ready.
* **Data transfer**: Rank0 starts reading data from Rank1.
* **Post-sync (Post / Wait DataSignal)**: Rank1 waits for Rank0 to send a DataSignal, indicating that the data has been read.

If Rank1 does not execute Post Ack, the Wait Ack task of Rank0 keeps waiting until it times out. Therefore, **the essence of operator execution timeout is a mismatch in the synchronization relationship between the two communicating parties**.

Common causes of execution timeout include:

* A rank does not issue the communication operator.
* The communication operators issued by the ranks are inconsistent.
* Inconsistent parameters such as data volume, data type, and algorithm selection

# Operator Execution Timeout (EI0002)

When an operator execution timeout occurs, the CANN log usually contains keywords such as `Task run failed` and `task_exception_handler.cc`:

```text
[ERROR] HCCL(2111665,all_reduce_test):2025-10-24-11:18:29.499.235 [task_exception_handler.cc:908] [2111665][TaskExecStage][Timeout][Host]Task run failed, base information is streamID:[2], taskID[21], tag[AllReduce_127.10.0.1%enp_60000_0_1761275812718970], AlgType(level 0-1-2):[fullmesh-ring-NHR].
[ERROR] HCCL(2111665,all_reduce_test):2025-10-24-11:18:29.499.247 [task_exception_handler.cc:771] [2111665][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[127.10.0.1%enp_60000_0_1761275812718970], user define information[], rankSize[4], rankId[0].
[ERROR] HCCL(2111665,all_reduce_test):2025-10-24-11:18:29.499.283 [task_exception_handler.cc:704] [2111665][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-10-24-11:16:55.493.816], deviceId[0], index[21], count[256], reduceType[sum], src[0x12c0c0013000], dst[0x12c0c0014000], dataType[float32].
```

The `group` field in the error log is the communicator name (that is, `127.10.0.1%enp_60000_0_1761275812718970`). This parameter applies to all subsequent troubleshooting steps.

## Troubleshooting Steps

### Obtaining the Locations of All Ranks in a Communicator

To retrieve all participating ranks by communicator name, run the command `grep -rn "Entry-HcclCommInit" run/plog | grep "<communicator name>"` to locate the positions of all ranks in the communicator. **The example shows the communicator size `ranks[4]` and the run log locations of `rank0` to `rank3`**, based on which the corresponding debug log locations can be found.

```text
run/plog/plog-2111667_20251024111652406.log:[INFO] HCCL(2111667,all_reduce_test):2025-10-24-11:16:52.725.226 [op_base.cc:1292] [2111668]Entry-HcclCommInitRootInfoInner:ranks[4], rank[3], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[3]
run/plog/plog-2111665_20251024111652405.log:[INFO] HCCL(2111665,all_reduce_test):2025-10-24-11:16:52.724.374 [op_base.cc:1292] [2111667]Entry-HcclCommInitRootInfoInner:ranks[4], rank[2], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[2]
run/plog/plog-2111668_20251024111652406.log:[INFO] HCCL(2111668,all_reduce_test):2025-10-24-11:16:52.719.213 [op_base.cc:1292] [2111665]Entry-HcclCommInitRootInfoInner:ranks[4], rank[0], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[0]
run/plog/plog-2111666_20251024111652405.log:[INFO] HCCL(2111666,all_reduce_test):2025-10-24-11:16:52.719.502 [op_base.cc:1292] [2111666]Entry-HcclCommInitRootInfoInner:ranks[4], rank[1], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[1]
```

### Determining Whether It Is a Full Timeout

Check the debug logs of all ranks in the communicator. If all ranks report a communication operator execution timeout, it is a full timeout. If only some ranks report a communication operator timeout, it is a partial timeout. The following figure shows the approach to locating timeout issues:

![op_execute_timeout_debug](figures/op_execute_timeout_debug.png)

#### Partial Timeout

* **Some ranks have no abnormal logs**
  
  If a rank in the communicator has no debug log, or its debug log contains no ERROR information, the rank did not deliver the communication operator.
  **This scenario is not an HCCL issue. The service side needs to investigate why the communication operator was not delivered.**
* **Some ranks have other errors**

  If the debug log of a rank in the communicator contains an ERROR, but it is not a communication operator execution timeout error, **prioritize analyzing the first error of that rank**. Usually the first error is the root cause, and the timeouts on the other ranks are a secondary effect. This scenario is not an HCCL issue.

#### Full Timeout

If all ranks in the communicator report a communication operator timeout, further analysis is required.

1. Check whether the communication parameters are consistent.

   Check whether the operator parameters in the error logs of all ranks are consistent: whether the communication operator, count, dataType, reduceType, and communicator name are exactly the same. **If they are inconsistent, this scenario is not an HCCL issue, and the service side needs to further analyze the cause of the inconsistent operator dispatch.**
   In the following case, rank0 in the same communicator reports an error in the AllReduce operator, while rank1 reports an error in the Allgather operator. In this case, further investigation is required on the service side to identify the root cause of the inconsistent operator dispatch among different ranks in the same communicator.

   ```text
   # rank0 error log:
   tag[AllReduce_127.10.0.1%enp_60000_0_1761275812718970], AlgType(level 0-1-2):[fullmesh-ring-NHR].
   [ERROR] HCCL(2111665,all_reduce_test):2025-10-24-11:18:29.499.247 [task_exception_handler.cc:771] [2111665][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[127.10.0.1%enp_60000_0_1761275812718970], user define information[], rankSize[4], rankId[0].
   [ERROR] HCCL(2111665,all_reduce_test):2025-10-24-11:18:29.499.283 [task_exception_handler.cc:704] [2111665][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-10-24-11:16:55.493.816], deviceId[0], index[21], count[256], reduceType[sum], src[0x12c0c0013000], dst[0x12c0c0014000], dataType[float32].

   # rank1 error log:
   tag[AllGather_127.10.0.1%enp_60000_0_1761275812718970], AlgType(level 0-1-2):[fullmesh-ring-NHR].
   [ERROR] HCCL(2111666,all_reduce_test):2025-10-24-11:18:29.513.764 [task_exception_handler.cc:771] [2111666][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[127.10.0.1%enp_60000_0_1761275812718970], user define information[], rankSize[4], rankId[1].
   [ERROR] HCCL(2111666,all_reduce_test):2025-10-24-11:18:29.513.787 [task_exception_handler.cc:704] [2111666][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-10-24-11:16:55.489.331], deviceId[1], index[21], count[256], src[0x12c0c0013000], dst[0x12c0c0014000], dataType[float32].
   ```

2. Check the operator dispatch time.

    If the operator parameters are consistent across all ranks, check the operator delivery time discrepancy between ranks within the communicator by comparing the **timeStamp** in each rank's error logs.
    If the operator delivery time interval between different ranks exceeds **HCCL_EXEC_TIMEOUT** (1836 seconds by default), all ranks will time out while waiting. In this case, check whether the operator delivery time interval between ranks exceeding the timeout is expected from the service perspective. If it is expected, specify an appropriate timeout value through the `HCCL_EXEC_TIMEOUT` environment variable. You can search the logs for the currently configured timeout value:

    ```bash
    grep -r "HCCL_EXEC_TIMEOUT" run/plog
   ```

## Troubleshooting Scenarios

For operator execution errors that are difficult to troubleshoot, when all the preceding checks meet expectations, enable the "HCCL_ENTRY_LOG_ENABLE" environment variable and reproduce the test case. After this environment variable is enabled, a log entry recording the input parameters of the communication operator dispatch is printed in the log files under the log/run/plog directory each time a communication operator is dispatched. If the test case fails, you can then check whether the communication operators dispatched on each rank are abnormal. To enable it:

```bash
export HCCL_ENTRY_LOG_ENABLE=1
```

After it is enabled, detailed input parameters are printed each time a communication operator is dispatched:

```text
[INFO] HCCL(3015875,python):2025-03-07-11:43:32.305.623 [hccl_opbase_atrace_info.cc:56][3017221]Entry-HcclAllReduce: tag[AllReduce_127.10.0.1%eth_60000_0_1741318944927847], sendBuf[0x1241d3dcdc00], recvBuf[0x124702f40200], count[10746295], dataType[float32], op[sum], localRank[0], streamId[7],comm[0xfffe380078d0], deviceLogicId[0]
[INFO] HCCL(3015875,python):2025-03-07-11:43:32.306.413 [hccl_opbase_atrace_info.cc:56][3017183]Entry-HcclAllReduce: tag[AllReduce_127.10.0.1%eth_60000_0_1741318944927847], sendBuf[0x1244bfffe000], recvBuf[0x1244bfffb400], count[1024], dataType[float32], op[sum], localRank[0], streamId[2],comm[0xfffe380078d0], deviceLogicId[0]
```

You can check the following based on:

* Whether all ranks have dispatched the communication operator
* Whether the operator order is consistent
* Whether count, dataType, and reduceType are consistent
* Whether an abnormal stream exists

The preceding log indicates that the service dispatches two AllReduce operators in the 127.10.0.1%eth_60000_0_1741318944927847 communicator, but on two different streams, streamId[7] and streamId[2]. Multiple streams execute concurrently on the NPU. If the service does not correctly implement the synchronization mechanism for stream execution, the two AllReduce operators in the same communicator execute concurrently. Because HCCL reuses communication operator resources in the same communicator, the concurrent execution of the two AllReduce operators causes resources such as notify to be consumed incorrectly, resulting in unexpected errors such as execution timeout or precision exceptions.
