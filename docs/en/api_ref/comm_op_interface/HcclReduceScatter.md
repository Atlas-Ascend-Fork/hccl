# HcclReduceScatter

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-06T09:44:19.488Z pushedAt=2026-11-16T06:28:10.090Z nl=2 -->

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

<!-- npu="910b,310p" id9 -->
> [!NOTE]
>
<!-- npu="910b" id17 -->
> - For Atlas A2 products, only Atlas 800T A2, Atlas 900 A2 PoD, and Atlas 200T A2 Box16 are supported.
<!-- end id17 -->
<!-- npu="310p" id18 -->
> - For Atlas inference products, only Atlas 300I Duo is supported.
<!-- end id18 -->
<!-- end id9 -->

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
| dataType | Input | Data type of the ReduceScatter operation, of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Supported data types vary by product. For details, see [dataType Description](#datatype-description).|
| op | Input | Reduce operation type.<br>Supported operation types vary by product. For details, see [Operation Types](#operation-types).|
| comm | Input | Communicator where the collective communication operation is performed. |
| stream | Input | Stream used by the current rank. |

### dataType Description

<!-- npu="950" id10 -->
- For Ascend 950PR&950DT products, supported data types: int8, int16, int32, int64, uint64, float16, float32, float64, bfp16, hifloat8 (hifloat8 is currently supported only in AIV expansion mode).
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
- For Atlas inference products, supported data types: int8, int16, int32, float16, float32.
<!-- end id6 -->

### Operation Types

<!-- npu="950" id14 -->
- For Ascend 950PR&950DT products, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 and bfp16 data types.
<!-- end id14 -->
<!-- npu="A3" id15 -->
- For Atlas A3 products, the supported operation types are sum, prod, max, and min. The prod operation does not support the int16 and bfp16 data types.
<!-- end id15 -->
<!-- npu="910b" id16 -->
- For Atlas A2 products, the supported operation types are sum, prod, max, and min. The prod operation does not support the int16 and bfp16 data types.
<!-- end id16 -->
<!-- npu="310p" id7 -->
- For Atlas inference products, the supported operation types are sum, prod, max, and min. The prod, max, and min operations do not support the int16 data type.
<!-- end id7 -->

## Return Values

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | The API call succeeds. |
| HCCL_E_PTR | An input pointer parameter is null, e.g., comm, sendBuf, recvBuf, and stream are nullptr. |
| HCCL_E_PARA | A passed parameter is invalid, such as recvCount exceeding the upper limit. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, it is not supported by the current product model, dataType is invalid, or the prod operation does not support the int16/bfp16 data type. |
| HCCL_E_INTERNAL | Internal error. |
| HCCL_E_UNAVAIL | A resource is unavailable, for example, a CCU resource reuse exception (null reuse thread or excess threads). |
| HCCL_E_RUNTIME | A call to an underlying runtime API (such as memory copy or memory zeroing) fails. |
| HCCL_E_MEMORY | Memory allocation or operation fails. |

## Constraints

- The recvCount, dataType, and op must be the same across all ranks.
<!-- npu="310p" id8 -->
- For Atlas inference products, only single-server use cases are supported. A single server supports a maximum of 16 inference devices (32 NPUs).
<!-- end id8 -->
- The input and output addresses (sendBuf and recvBuf) of the operator must meet the following alignment requirements based on the data type:

  - int8 and hifloat8: 1-byte address alignment.
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
uint64_t recvCount = 1;  // Number of data elements received by each node.
uint64_t sendSize = rankSize * recvCount * sizeof(float);
uint64_t recvSize = recvCount * sizeof(float);

// Allocate device memory for the collective communication operation.
void *sendBuf = nullptr, *recvBuf = nullptr;
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Initialize the communicator and stream.
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Perform ReduceScatter: Add sendBuf of all ranks, and evenly scatter the result to recvBuf of each rank in rank_id order.
HcclReduceScatter(sendBuf, recvBuf, recvCount, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
// Block until the collective communication tasks in the stream complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free device memory.
aclrtFree(recvBuf);          // Free device memory.
aclrtDestroyStream(stream);  // Destroy the stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
