# SDMA ERROR (EI0012)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:54:37.141Z pushedAt=2026-10-14T02:54:56.867Z -->

## Symptoms

The console log contains the error code EI0012 with the keyword "Execution_Error_SDMA", as shown below:

```text
[PID: 3480365] 2025-12-24-14:10:31.094.189 Execution_Error_SDMA(EI0012): SDMA memory copy task exception occurred. Remote rank: [4800]. Base information: [streamID:[351], taskID[5], taskType[Memcpy], tag[], AlgType(level 0-1-2):[null-null-null].]. Task information: [src:[0x12c180000000], dst:[0x12c041800000], size:[0x80], notify id:[0xffffffffffffffff], link type:[HCCS], remote rank:[0]]. Communicator information: [group:[], user define information[], rankSize[0], rankId[0]].
```

In addition, the CANN log contains the keyword "**fftsplus sdma error**", as shown below:

```text
[ERROR] RUNTIME(57096,python3.10):2025-05-12-20:55:44.705.025 [task_info.cc:1170]57288 PrintSdmaErrorInfoForFftsPlusTask:fftsplus task execute failed, dev_id=0, stream_id=50, task_id=21, context_id=18, thread_id=0, err_type=4[fftsplus sdma error]
[ERROR] RUNTIME(57096,python3.10):2025-05-12-20:55:44.705.031 [task_info.cc:1270]57288 TaskFailCallBackForFftsPlusTask:fftsplus streamId=50, taskId=21, context_id=18, expandtype=1, rtCode=0x715006c,[fftsplus task exception], psStart=0x0, kernel_name=not found kernel name, binHandle=(nil), binSize=0.
[ERROR] HCCL(57096,python3.10):2025-05-12-20:55:44.706.132 [task_exception_handler.cc:947] [57288][TaskExecStage][Timeout][Host]Task run failed, base information is streamID:[32], taskID[21], tag[AllGather_group_name_0], AlgType(level 0-1-2):[fullmesh-ring-H-D].
[ERROR] HCCL(57096,python3.10):2025-05-12-20:55:44.706.140 [task_exception_handler.cc:810] [57288][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[group_name_0], user define information[Unspecified], rankSize[8], rankId[0].
[ERROR] HCCL(57096,python3.10):2025-05-12-20:55:44.706.163 [task_exception_handler.cc:737] [57288][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-05-12-20:54:51.268.778], deviceId[0], index[4], count[3397632], src[0x12c25487ac00], dst[0x12c255000000], dataType[uint8].
```

## Possible Causes

A page table translation failure occurred during the execution of an SDMA memory copy task. Specifically, the input or output address of the memory copy was not allocated memory, the allocated memory was smaller than the memory copy size, or the allocated memory had already been freed.

Common root causes include the following scenarios:

- After a communication operator is dispatched, the communicator is destroyed without performing stream synchronization to confirm that the communication operator has completed execution. Since destroying the communicator releases the HCCL buffer addresses used for collective communication, this causes a page table translation failure during SDMA memory copy.

    The time when the communicator is destroyed can be retrieved from the `run` directory of the CANN logs:

    ```bash
    grep -r "Entry-HcclCommDestroy" log/run/plog
    ```

  <!-- npu="A3" id2 -->
- On Atlas A3 products, a network link failure can also cause an SDMA ERROR. In this case, check the link status between the two ends.
  <!-- end id2 -->
- When an HCCL communication operator is called, the actual allocated memory size of the passed input or output address is smaller than the passed data volume Count.
