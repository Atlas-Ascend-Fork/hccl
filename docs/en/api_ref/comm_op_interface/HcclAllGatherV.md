# HcclAllGatherV

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-19T07:23:16.742Z pushedAt=2026-10-19T10:17:19.052Z -->

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
- Atlas training products: Not supported
<!-- end id5 -->

## Function

The operation API of the collective communication operator AllGatherV, which reorders the inputs of all nodes in the communicator by rank ID, concatenates them, and then sends the result to the outputs of all nodes.

Unlike the AllGather operator, the AllGatherV operator supports configuring different data sizes for the inputs of different nodes in the communicator.

![allgatherv](figures/allgatherv.png)

> [!NOTE]
> For AllGatherV operations, each node receives the dataset reordered by rank ID, meaning that the AllGatherV output is the same for every node.

## Prototype

```c
HcclResult HcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts, const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| sendCount | Input | Data size of sendBuf participating in the AllGatherV operation. |
| recvBuf | Output | Destination data buffer address, to which the collective communication result is output.<br>recvBuf and sendBuf cannot be set to the same address. |
| recvCounts | Input | Data size of each rank participating in the AllGatherV operation in recvBuf, in an array of the uint64 type.<br>The i-th element of the array indicates the amount of data to be received from rank i, which must be the same as the sendCount value of rank i. |
| recvDispls | Input | Offset in recvBuf of the data of each rank participating in the AllGatherV operation (in units of dataType), in an array of the uint64 type.<br>The i-th element of the array indicates the start offset in recvBuf for the data received from rank i. |
| dataType | Input | Data type of the AllGatherV operation. The value is of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different product models support different data types. For details, see [Supported Data Types](#supported-data-types).|
| comm | Input | Communicator where the collective communication operation is performed. |
| stream | Input | Stream used by the current rank. |

### Supported Data Types

<!-- npu="950" id11 -->
- For Ascend 950PR&950DT products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float8-e5m2, float8-e4m3, float8-e8m0, hifloat8, float16, float32, float64, bfp16.
<!-- end id11 -->
<!-- npu="A3" id12 -->
- For Atlas A3 products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.
<!-- end id12 -->
<!-- npu="910b" id13 -->
- For Atlas A2 products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.
<!-- end id13 -->
<!-- npu="310p" id6 -->
- For Atlas 300I Duo, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.
<!-- end id6 -->

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | The API call succeeds. |
| HCCL_E_PTR | The input pointer parameter is null, such as comm, recvCounts, recvDispls, or stream is nullptr (sendBuf cannot be nullptr when sendCount > 0, and recvBuf cannot be nullptr when recvCounts is not all zeros). |
| HCCL_E_PARA | An input parameter is invalid, such as count exceeding the upper limit. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, the current model does not support it or dataType is invalid. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- The recvCounts, recvDispls, and dataType must be identical across all ranks.
<!-- npu="A3" id15 -->
- For Atlas A3 products, only single-server use cases are supported.
<!-- end id15 -->
<!-- npu="910b" id16 -->
- For Atlas A2 products, only multi-server symmetric deployment use cases are supported. Asymmetric deployment (that is, an asymmetric number of devices) is not supported.
<!-- end id16 -->
<!-- npu="310p" id10 -->
- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of two Atlas 300I Duo inference cards (i.e., four NPUs) per server.
<!-- end id10 -->
- All communication operators in multiple communicators must be dispatched serially on each device. Out-of-order dispatch, multi-thread concurrent dispatch, and thread reentrancy are not supported.
- On the same device, the dispatch threads of all communication operators in the same communicator must use the same context.

## Example

```c
// Allocate device memory for the collective communication operation.
uint32_t rankSize = 8;
uint64_t sendCount = 1;  // Number of data items sent by each rank
size_t sendSize = sendCount * sizeof(float);
size_t recvSize = rankSize * sendCount * sizeof(float);

void *sendBuf = nullptr;
void *recvBuf = nullptr;
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Set recvCounts and recvDispls so that each rank receives the same amount of data.
std::vector<uint64_t> recvCounts(rankSize, sendCount);
std::vector<uint64_t> recvDispls(rankSize);
for (uint32_t i = 0; i < rankSize; ++i) {
    recvDispls[i] = i * sendCount;
}

// Initialize the communicator.
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Create a stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Execute AllGatherV to sort the sendBuf of all ranks in the communicator by rank ID, concatenate the data, and then send the result to the recvBuf of all ranks.
HcclAllGatherV(sendBuf, sendCount, recvBuf, recvCounts.data(), recvDispls.data(), HCCL_DATA_TYPE_FP32, hcclComm, stream);
// Block until the collective communication tasks in the stream are complete.
aclrtSynchronizeStream(stream);

// Release resources
aclrtFree(sendBuf);          // Free device memory
aclrtFree(recvBuf);          // Free device memory
aclrtDestroyStream(stream);  // Destroy the stream
HcclCommDestroy(hcclComm);   // Destroy the communicator
```
