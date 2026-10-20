# HcclBroadcast

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-19T07:25:07.567Z pushedAt=2026-10-19T10:17:19.062Z -->

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

The operation API of the collective communication operator Broadcast, which broadcasts data from the root node in the communicator to other ranks.

![broadcast](figures/broadcast.png)

## Prototype

```c
HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| buf | Input/Output | Data buffer address. For the root node, the address of the source data buffer; for non-root nodes, the address of the buffer for receiving data. |
| count | Input | Number of data elements involved in the broadcast operation. For example, if only one int32 data element is involved, count=1. |
| dataType | Input | Data type of the broadcast operation, of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Supported data types vary by product model. For details, see [Supported Data Types](#supported-data-types).|
| root | Input | Rank ID that acts as the broadcast root. |
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

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | The API call succeeds. |
| HCCL_E_PTR | An input pointer parameter is null, such as comm, buf, or stream being nullptr. |
| HCCL_E_PARA | An input parameter is invalid, for example, count exceeds the upper limit, or root is out of range. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, the current model does not support it or dataType is invalid. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- All ranks must have the same count, dataType, and root.
- There can be only one root node globally.
- All communication operators in multiple communicators must be dispatched serially on each device. Out-of-order dispatch, multi-thread concurrent dispatch, and thread reentrancy are not supported.
- On the same device, the dispatch threads of all communication operators in the same communicator must use the same context.

## Example

```c
// Allocate device memory for the collective communication operation.
void *buf = nullptr;    // For the root node, it is the data source; for non-root nodes, it is the buffer for receiving data.
uint64_t count = 8;     // Number of data elements participating in the broadcast operation
size_t mallocSize = count * sizeof(float);
aclrtMalloc(&buf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Construct the input data on the root node.
if (deviceId == rootRank) {    
    aclrtMemcpy(buf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE);
}

// Initialize the communicator.
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Create a task stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Perform the broadcast operation to broadcast the data on the root node in the communicator to other ranks.
HcclBroadcast(buf, count, HCCL_DATA_TYPE_FP32, rootRank, hcclComm, stream);
// Block and wait for the collective communication tasks in the task stream to complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(buf);              // Free device-side memory.
aclrtDestroyStream(stream);  // Destroy the task stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
