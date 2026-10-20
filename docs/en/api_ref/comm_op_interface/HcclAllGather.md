# HcclAllGather

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-19T07:22:32.057Z pushedAt=2026-10-19T10:17:19.047Z -->

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
| dataType | Input | Data type of the AllGather operation, of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different product models support different data types. For details, see [Supported Data Types](#supported-data-types).|
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
- For Atlas 300I Duo, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.
<!-- end id6 -->

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | The API call succeeds. |
| HCCL_E_PTR | An input pointer parameter is null, e.g., comm, sendBuf, recvBuf, and stream are nullptr. |
| HCCL_E_PARA | An input parameter is invalid, such as count exceeding the upper limit. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, the current model does not support it or dataType is invalid. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- The sendCount and dataType must be the same across all ranks.
- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of 16 Atlas 300I Duo inference cards (i.e., 32 NPUs) per server.
- All communication operators in multiple communicators must be dispatched serially on each device. Out-of-order dispatch, multi-thread concurrent dispatch, and thread reentrancy are not supported.
- On the same device, the dispatch threads of all communication operators in the same communicator must use the same context.

## Example

```c
// Allocate device memory for the collective communication operation.
void *sendBuf = nullptr, *recvBuf = nullptr;
uint32_t rankSize = 8;
uint64_t sendCount = 1;  // Number of data elements sent by each node
size_t sendSize = sendCount * sizeof(float);
size_t recvSize = rankSize * sendCount * sizeof(float);
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Initialize the communicator and stream.
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, devId, &hcclComm);

// Create a task stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Perform AllGather to concatenate the sendBuf of all ranks in the communicator in rank_id order and send the result to the recvBuf of all ranks.
HcclAllGather(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream);
// Block until the collective communication tasks in the task stream are complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free device memory.
aclrtFree(recvBuf);          // Free device memory.
aclrtDestroyStream(stream);  // Destroy the task stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
