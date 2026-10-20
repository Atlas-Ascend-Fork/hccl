# HcclReduceScatter

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-19T07:26:12.020Z pushedAt=2026-10-19T10:17:19.072Z -->

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

The operation API of the collective communication operator ReduceScatter, which divides the input data of all ranks in the communicator into *{ranksize}* portions, then takes one of the *{ranksize}* portions from each rank for a reduction operation (such as sum, prod, max, min). Finally, the results are scattered to the output buffer of each rank according to the rank indices.

![reducescatter](figures/reducescatter.png)

## Prototype

```c
HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| recvBuf | Output | Destination data buffer address, to which the collective communication result is output. |
| recvCount | Input | Data size of recvBuf in the ReduceScatter operation. Data size of sendBuf = recvCount * rank size |
| dataType | Input | Data type of the ReduceScatter operation. The value is of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different products support different data types. For details, see [dataType Description](#datatype-description).|
| op | Input | Reduce operation type.<br>The supported operation types vary by product model. For details, see [Operation Types](#operation-types).|
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
<!-- npu="310p" id6 -->
- For Atlas 300I Duo, supported data types: int8, int16, int32, float16, float32.
<!-- end id6 -->

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
<!-- npu="310p" id7 -->
- For Atlas 300I Duo, the supported operation types are sum, prod, max, and min. The prod, max, and min operations do not support the int16 data type.
<!-- end id7 -->

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | The API call succeeds. |
| HCCL_E_PTR | An input pointer parameter is null, e.g., comm, sendBuf, recvBuf, and stream are nullptr. |
| HCCL_E_PARA | An input parameter is invalid, such as count exceeding the upper limit. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, it is not supported by the current product model, dataType is invalid, or the prod operation does not support the int16/bfp16 data type. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- The recvCount, dataType, and op must be the same across all ranks.
<!-- npu="310p" id8 -->
- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of 16 Atlas 300I Duo inference cards (i.e., 32 NPUs) per server.
<!-- end id8 -->
- The input and output addresses (sendBuf and recvBuf) of the operator must meet the following alignment requirements based on the data type:

  - int8: 1-byte address alignment.
  - int16, float16, and bfp16: 2-byte address alignment.
  - int32 and float32: 4-byte address alignment.
  - int64, uint64, and float64: 8-byte address alignment.
- All communication operators in multiple communicators must be dispatched serially on each device. Out-of-order dispatch, multi-thread concurrent dispatch, and thread reentrancy are not supported.
- On the same device, the dispatch threads of all communication operators in the same communicator must use the same context.

- When a communicator has registered symmetric memory, the input buffer pointed to by sendBuf is used as a symmetric window for direct read/write by remote ranks. During ReduceScatter, some algorithms perform in-place reduction (read+reduce) on sendBuf, modifying (corrupting) the data in sendBuf. Therefore, make sure that:
  - After calling HcclReduceScatter, the data in sendBuf is no longer used as the original input;
  - If the original input data needs to be retained, back up sendBuf before the call.

## Example

```c
uint32_t rankSize = 8;
uint64_t recvCount = 1;  // Number of data elements received by each node
uint64_t sendSize = rankSize * recvCount * sizeof(float);
uint64_t recvSize = recvCount * sizeof(float);

// Allocate device memory for the collective communication operation.
void *sendBuf = nullptr, *recvBuf = nullptr;
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Initialize the communicator and stream.
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Execute ReduceScatter. Sum the sendBuf of all ranks, and evenly scatter the result to the recvBuf of each rank in rank_id order.
HcclReduceScatter(sendBuf, recvBuf, recvCount, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
// Block until the collective communication tasks in the task stream are complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free device-side memory.
aclrtFree(recvBuf);          // Free device-side memory.
aclrtDestroyStream(stream);  // Destroy the task stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
