# HcclReduceScatterV

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-19T07:27:26.627Z pushedAt=2026-10-19T10:17:19.083Z -->

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

The operation API of the collective communication operator ReduceScatterV, which is similar to ReduceScatter, except that it allows different nodes within the communicator to be configured with different data sizes (the data size for different indices on the same rank can be set, but the data size for the same index across different ranks must be consistent). It performs a reduction operation (supporting sum, prod, max, and min) on the data corresponding to each index across all ranks, and then scatters the results to the output buffer of each rank by index.

![reducescatterv](figures/reducescatterv.png)

## Prototype

```c
HcclResult HcclReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| sendCounts | Input | Size of the data in sendBuf of each rank participating in the ReduceScatterV operation, as an array of the uint64 type.<br>The i-th element of the array indicates the amount of data to send to rank i. |
| sendDispls | Input | Offset (in units of dataType) of the data in sendBuf of each rank participating in the ReduceScatterV operation, as an array of the uint64 type.<br>The i-th element of the array indicates the offset in sendBuf of the data to send to rank i. |
| recvBuf | Output | Destination data buffer address, to which the collective communication result is output.<br>recvBuf and sendBuf cannot be set to the same address. |
| recvCount | Input | Size of the data in recvBuf of the rank participating in the ReduceScatterV operation.<br>If the current rank ID is i, the value of recvCount must be the same as that of the element with index i in the sendCounts array. |
| dataType | Input | Data type of the ReduceScatterV operation, of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>The supported data types vary by product model. For details, see [dataType Description](#datatype-description).|
| op | Input | Reduce operation type.<br>The supported operation types vary by product model. For details, see [Operation Types](#operation-types).|
| comm | Input | Communicator where the collective communication operation is performed. |
| stream | Input | Stream used by the current rank. |

### dataType Description

<!-- npu="950" id10 -->
- For Ascend 950PR&950DT products, supported data types: int8, int16, int32, int64, float16, float32, and bfp16.
<!-- end id10 -->
<!-- npu="A3" id11 -->
- For Atlas A3 products, supported data types: int8, int16, int32, int64, float16, float32, and bfp16.
<!-- end id11 -->
<!-- npu="910b" id12 -->
- For Atlas A2 products, supported data types: int8, int16, int32, float16, float32, and bfp16.
<!-- end id12 -->
<!-- npu="310p" id6 -->
- For Atlas 300I Duo, supported data types: int16, float16, float32.
<!-- end id6 -->

### Operation Types

<!-- npu="950" id14 -->
- For Ascend 950PR&950DT products, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 or bfp16 data types.
<!-- end id14 -->
<!-- npu="A3" id15 -->
- For Atlas A3 products, the supported operation types are sum, max, and min.
<!-- end id15 -->
<!-- npu="910b" id16 -->
- For Atlas A2 products, the supported operation types are sum, max, and min.
<!-- end id16 -->
<!-- npu="310p" id7 -->
- For Atlas 300I Duo, only sum is supported.
<!-- end id7 -->

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | The API call succeeds. |
| HCCL_E_PTR | An input pointer parameter is null, for example, comm, sendCounts, sendDispls, or stream is nullptr. (recvBuf cannot be nullptr when recvCount is greater than 0.) |
| HCCL_E_PARA | An input parameter is invalid, such as count exceeding the upper limit. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, it is not supported by the current product model, dataType is invalid, or the prod operation does not support the int16/bfp16 data type. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- The sendCounts, sendDispls, dataType, and op must be the same across all ranks.
<!-- npu="A3" id18 -->
- For Atlas A3 products, only single-server use cases are supported.
<!-- end id18 -->
<!-- npu="910b" id19 -->
- For Atlas A2 products, only multi-server symmetric deployment use cases are supported. Asymmetric deployment (that is, an asymmetric number of devices) is not supported.
<!-- end id19 -->
<!-- npu="310p" id8 -->
- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of two Atlas 300I Duo inference cards (i.e., four NPUs) per server.
<!-- end id8 -->
- The input and output addresses (sendBuf and recvBuf) of the operator must meet the following alignment requirements based on the data type:
  - int8: 1-byte address alignment.
  - int16, float16, and bfp16: 2-byte address alignment.
  - int32 and float32: 4-byte address alignment.
  - int64: 8-byte address alignment.
- All communication operators in multiple communicators must be dispatched serially on each device. Out-of-order dispatch, multi-thread concurrent dispatch, and thread reentrancy are not supported.
- On the same device, the dispatch threads of all communication operators in the same communicator must use the same context.

## Example

```c
// Allocate device memory for the collective communication operation.
uint32_t rankSize = 8;
uint64_t recvCount = 1;  // Number of data elements received by each rank.
size_t recvSize = recvCount * sizeof(float);
size_t totalSendCount = rankSize * recvCount;
size_t sendSize = totalSendCount * sizeof(float);

void *sendBuf = nullptr;
void *recvBuf = nullptr;
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Set sendCounts and sendDispls. Each rank sends the same amount of data.
std::vector<uint64_t> sendCounts(rankSize, recvCount);
std::vector<uint64_t> sendDispls(rankSize);
for (uint32_t i = 0; i < rankSize; ++i) {
    sendDispls[i] = i * recvCount;
}

// Initialize the communicator.
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Create a task stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Execute ReduceScatterV: Sum sendBuf of all ranks, then scatter the result to recvBuf of each rank according to rank ID.
HcclReduceScatterV(sendBuf, sendCounts.data(), sendDispls.data(), recvBuf, recvCount, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
// Block and wait for the collective communication tasks in the task stream to complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free device-side memory.
aclrtFree(recvBuf);          // Free device-side memory.
aclrtDestroyStream(stream);  // Destroy the task stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
