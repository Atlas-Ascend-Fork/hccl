# HcclAllGather

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-06T09:43:14.507Z pushedAt=2026-11-16T06:28:10.065Z nl=2 -->

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

<!-- npu="910b,310p" id7 -->
> [!NOTE]
>
> <!-- npu="910b" id8 -->
> - For Atlas A2 products, only Atlas 800T A2, Atlas 900 A2 PoD, and Atlas 200T A2 Box16 are supported.
> <!-- end id8 -->
> <!-- npu="310p" id9 -->
> - For Atlas inference products, only Atlas 300I Duo is supported.
<!-- end id9 -->
<!-- end id7 -->

## Function

The operation API of the collective communication operator AllGather, which reorders the inputs of all nodes in the communicator by rank ID, concatenates them, and then sends the result to the outputs of all nodes.

![allgather](figures/allgather.png)

> [!NOTE]
> For AllGather operations, each node receives the dataset reordered by rank ID, meaning that the AllGather output is the same for every node.

## Prototype

```c
HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| recvBuf | Output | Destination data buffer address, to which the collective communication result is output. |
| sendCount | Input | Data size of sendBuf participating in the AllGather operation. Data size of recvBuf = sendCount * rank size |
| dataType | Input | Data type of the AllGather operation, of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different product models support different data types. For details, see [Supported Data Types](#supported-data-types).|
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
- For Atlas inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.
<!-- end id6 -->

## Return Values

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | The API call succeeds. |
| HCCL_E_PTR | An input pointer parameter is null, e.g., comm, sendBuf, recvBuf, and stream are nullptr. |
| HCCL_E_PARA | A parameter is invalid, such as sendCount exceeding the upper limit. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, the current model does not support it or dataType is invalid. |
| HCCL_E_INTERNAL | Internal error. |
| HCCL_E_UNAVAIL | Resources are unavailable, such as a CCU resource reuse exception (null reuse thread or the number of threads exceeding the limit). |
| HCCL_E_RUNTIME | A call to an underlying runtime API (such as memory copy or memory zeroing) fails. |
| HCCL_E_MEMORY | Memory allocation or operation fails. |

## Constraints

- The sendCount and dataType must be the same across all ranks.
<!-- npu="310p" id14 -->
- For Atlas inference products, only single-server use cases are supported, with a maximum of 16 inference devices (32 NPUs) deployed in a single server.
<!-- end id14 -->
- All communication operators in multiple communicators must be dispatched serially on each device. Out-of-order dispatch, multi-thread concurrent dispatch, and thread reentrancy are not supported.
- On the same device, the dispatch threads of all communication operators in the same communicator must use the same context.

## Example

```c
// Allocate device memory for the collective communication operation.
void *sendBuf = nullptr, *recvBuf = nullptr;
uint32_t rankSize = 8;
uint64_t sendCount = 1;  // Number of data elements sent by each node.
size_t sendSize = sendCount * sizeof(float);
size_t recvSize = rankSize * sendCount * sizeof(float);
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Initialize the communication domain and stream.
uint32_t deviceId = 0;  // Logical device ID used by the current process.
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Create a task stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Execute AllGather to concatenate sendBuf of all ranks in the communication domain in rank_id order, and then send the result to recvBuf of all ranks.
HcclAllGather(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream);
// Block and wait until the collective communication task in the task stream completes.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free the device memory.
aclrtFree(recvBuf);          // Free the device memory.
aclrtDestroyStream(stream);  // Destroy the task stream.
HcclCommDestroy(hcclComm);   // Destroy the communication domain.
```
