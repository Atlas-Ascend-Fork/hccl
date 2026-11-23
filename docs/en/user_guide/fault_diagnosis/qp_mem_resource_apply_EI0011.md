# QP Memory Resource Allocation (EI0011)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-09T03:27:43.900Z pushedAt=2026-11-11T08:53:32.185Z nl=2 -->

During the parameter plane link setup phase, HCCL creates QPs. If the device-side memory is insufficient, an OOM error is reported. To resolve this issue, adjust the service configuration, reduce the number of RoCE links in use, or release some memory.

## Symptoms

The keyword "EI0011" or "Resource_Error_Insufficient_Device_Memory" appears in the console log, as shown below:

```text
[PID: 2103452] 2025-11-03-20:18:46.447.213 Resource_Error_Insufficient_Device_Memory(EI0011): Failed to allocate [size: [0.25MB, 3MB], Affected by QP depth configuration.] bytes of NPU memory.
        Possible Cause: Allocation failure due to insufficient NPU memory.
        Solution: Stop unnecessary processes and ensure the required memory is available.
```

## Solution

Adjust the service configuration (such as batchSize), reduce the number of ROCE links in use, or release some memory to resolve the issue.

**NOTE:**

If an OOM error occurs during other HCCL memory allocations, such as cclBuffer memory allocation, the drv component reports the error code and prints error information. Whether the failure is caused by HCCL memory allocation can be determined based on the error information or the stack in CANN logs. If the failure is caused by HCCL memory allocation, the requested memory size can be adjusted by configuring the HCCL_BUFFSIZE environment variable.
