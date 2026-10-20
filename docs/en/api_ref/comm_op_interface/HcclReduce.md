# HcclReduce

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-19T07:26:01.041Z pushedAt=2026-10-19T10:17:19.067Z -->

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
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
<!-- end id5 -->

## Function

The operation API for the collective communication operator Reduce, which adds (or performs other reduce operations on) data from all ranks and then sends the result to the specified position on the root node.

![reduce](figures/reduce.png)

## Prototype

```c
HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| recvBuf | Output | Destination data buffer address, to which the collective communication result is output. |
| count | Input | Number of data elements participating in the reduce operation. For example, if only one int32 data is involved, count=1. |
| dataType | Input | Data type of the reduce operation, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Supported data types vary by product. For details, see [dataType description](#datatype-description).|
| op | Input | Operation type of the reduce operation.<br>Supported operation types vary by product. For details, see [Operation Types](#operation-types).|
| root | Input | Rank ID of the reduce root. |
| comm | Input | Communicator where the collective communication operation is performed. |
| stream | Input | Stream used by the current rank. |

### dataType Description

<!-- npu="950" id10 -->
- For Ascend 950PR&950DT products, supported data types: int8, int16, int32, int64, uint64, float16, float32, float64, bfp16.
<!-- end id10 -->
<!-- npu="A3" id11 -->
- For Atlas A3 products, supported data types: int8, int16, int32, int64, float16, float32, and bfp16.
<!-- end id11 -->
<!-- npu="910b" id12 -->
- For Atlas A2 products, supported data types: int8, int16, int32, int64, float16, float32, and bfp16. Note that the performance of the int64 data type degrades to some extent.
<!-- end id12 -->
<!-- npu="910" id13 -->
- For Atlas training products, supported data types: int8, int32, int64, float16, and float32.
<!-- end id13 -->

### Operation Types

<!-- npu="950" id14 -->
- For Ascend 950PR&950DT products, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 or bfp16 data types.
<!-- end id14 -->
<!-- npu="A3" id15 -->
- For Atlas A3 products, the supported operation types are sum, prod, max, and min, where the prod operation does not support the int16 or bfp16 data types.
<!-- end id15 -->
<!-- npu="910b" id16 -->
- For Atlas A2 products, the supported operation types are sum, prod, max, and min, where the prod operation does not support the int16 or bfp16 data types.
<!-- end id16 -->
<!-- npu="910" id17 -->
- For Atlas training products, the supported operation types are sum, prod, max, and min.
<!-- end id17 -->

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | The API call succeeds. |
| HCCL_E_PTR | An input pointer parameter is null, e.g., comm, sendBuf, recvBuf, and stream are nullptr. |
| HCCL_E_PARA | An input parameter is invalid, for example, count exceeds the upper limit, or root is out of range. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, it is not supported by the current product model, dataType is invalid, or the prod operation does not support the int16/bfp16 data type. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- The count, dataType, and op must be the same across all ranks.
- The input and output addresses (sendBuf and recvBuf) of the operator must meet the following alignment requirements based on the data type:
  - int8: 1-byte address alignment.
  - int16, float16, and bfp16: 2-byte address alignment.
  - int32 and float32: 4-byte address alignment.
  - int64, uint64, and float64: 8-byte address alignment.
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

// Execute Reduce. Add the sendBuf values at corresponding positions of all ranks, and send the result to recvBuf on the root node.
HcclReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, rootRank, hcclComm, stream);
// Block until the collective communication tasks in the task stream are complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free device-side memory.
aclrtFree(recvBuf);          // Free device-side memory.
aclrtDestroyStream(stream);  // Destroy the task stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
