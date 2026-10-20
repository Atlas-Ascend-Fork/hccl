# HcclAlltoAll

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-19T07:22:27.367Z pushedAt=2026-10-19T10:17:19.044Z -->

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
<!-- end id5 -->

## Function

The operation API for the collective communication operator AlltoAll, which sends data of the same size to all ranks in the communicator and receives data of the same size from all ranks.

![alltoall](figures/alltoall.png)

The AlltoAll operation splits the input data into a specific number of blocks along a specific dimension, sends them to other ranks in order, and simultaneously receives input data from other ranks, concatenating the data along the specific dimension in order.

## Prototype

```c
HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf, uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| sendCount | Input | Amount of data sent to each rank. |
| sendType | Input | Data type of the data to send. The value is of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Supported data types vary by model. For details, see [Supported Data Types](#supported-data-types).|
| recvBuf | Output | Destination data buffer address, to which the collective communication result is output.<br>The addresses configured for recvBuf and sendBuf cannot be the same, and their memory ranges cannot overlap. |
| recvCount | Input | Amount of data received from each rank. The value must be the same as that of sendCount. |
| recvType | Input | Data type of the data to be received, of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type. The value must be the same as that of sendType.|
| comm | Input | Communicator where the collective communication operation is performed. |
| stream | Input | Stream used by the current rank. |

### Supported Data Types

<!-- npu="950" id10 -->
- For Ascend 950PR&950DT products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float8-e5m2, float8-e4m3, float8-e8m0, hifloat8, float16, float32, float64, bfp16.
<!-- end id10 -->
<!-- npu="A3" id11 -->
- For Atlas A3 products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.
<!-- end id11 -->
<!-- npu="910b" id12 -->
- For Atlas A2 products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.
<!-- end id12 -->
<!-- npu="910" id13 -->
- For Atlas training products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.
<!-- end id13 -->
<!-- npu="310p" id6 -->
- For Atlas 300I Duo,  supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.
<!-- end id6 -->

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | The API call succeeds. |
| HCCL_E_PTR | An input pointer parameter is null, e.g., comm, sendBuf, recvBuf, and stream are nullptr. |
| HCCL_E_PARA | The input parameter is invalid, such as sendCount being inconsistent with recvCount, sendType being inconsistent with recvType, or sendBuf and recvBuf having the same address. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, the current model does not support it or dataType is invalid. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- The sendCount, sendType, recvCount, and recvType of all ranks must be the same.
- The performance of the AlltoAll operation depends on the buffer size for data sharing between NPUs. When the communication data volume exceeds the buffer size, performance will degrade significantly. If the AlltoAll communication data volume in your service is large, you are advised to configure the environment variable [HCCL_BUFFSIZE](../../user_guide/hccl_env/HCCL_BUFFSIZE.md) to appropriately increase the buffer size to improve communication performance.
<!-- npu="910" id7 -->
- For Atlas training products, the AlltoAll communicator must meet the following constraints:

    For a single server, 1p and 2p communicators must be within the same cluster (devices 0-3 and devices 4-7 in a server each form a cluster). For single-server 4p and 8p and multi-server communicators, ranks must be organized by cluster as the basic unit, and the cluster selection across servers must be consistent.
    
- For Atlas training products, in single-server use cases, the NIC status must be `up`; otherwise, this API will fail to execute.
<!-- end id7 -->
<!-- npu="310p" id14 -->
- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of two Atlas 300I Duo inference cards (i.e., four NPUs) per server.
<!-- end id14 -->
- All communication operators in multiple communicators must be dispatched serially on each device. Out-of-order dispatch, multi-thread concurrent dispatch, and thread reentrancy are not supported.
- On the same device, the dispatch threads of all communication operators in the same communicator must use the same context.

## Example

```c
// Allocate device memory for the collective communication operation.
void *sendBuf = nullptr;
void *recvBuf = nullptr;
uint64_t count = 8;
size_t mallocSize = count * sizeof(float);
aclrtMalloc((void **)&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc((void **)&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Initialize the communicator.
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Create a task stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Execute AlltoAll. Send the same amount of data to and receive the same amount of data from all ranks in the communicator.
size_t perCount = count / rankSize;
HcclAlltoAll(sendBuf, perCount, HCCL_DATA_TYPE_FP32, recvBuf, perCount, HCCL_DATA_TYPE_FP32, hcclComm, stream);
// Block until the collective communication tasks in the task stream are complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free device memory.
aclrtFree(recvBuf);          // Free device memory.
aclrtDestroyStream(stream);  // Destroy the task stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
