# HcclScatter

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-19T07:27:43.791Z pushedAt=2026-10-19T10:17:19.086Z -->

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

The operation API of the collective communication operator Scatter, which evenly scatters data from the root node to other ranks.

## Prototype

```c
HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| recvBuf | Output | Destination data buffer address, to which the collective communication result is output. |
| recvCount | Input | Number of data elements in recvBuf that participate in the scatter operation. For example, if only one int32 data element participates, recvcount=1. |
| dataType | Input | Data type of the Scatter operation. The value is of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Supported data types vary by product model. For details, see [Supported Data Types](#supported-data-types).|
| root | Input | Rank ID that acts as the scatter root. |
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
| HCCL_E_PTR | An input pointer parameter is null, for example, comm and recvBuf are nullptr (sendBuf of the root node cannot be nullptr either). |
| HCCL_E_PARA | An input parameter is invalid, for example, count exceeds the upper limit, or root is out of range. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, it is not supported by the current model, dataType is invalid, or hybrid networking does not support Scatter. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- recvCount, dataType, and root must be the same across all ranks.
- There can be only one root node globally.
- The sendBuf of a non-root node can be null. The sendBuf of the root node cannot be null.
- All communication operators in multiple communicators must be dispatched serially on each device. Out-of-order dispatch, multi-thread concurrent dispatch, and thread reentrancy are not supported.
- On the same device, the dispatch threads of all communication operators in the same communicator must use the same context.

## Example

```c
void *sendBuf = nullptr;
void *recvBuf = nullptr;
uint64_t sendCount = 8;
uint64_t recvCount = 1;
size_t sendSize = sendCount * sizeof(float);
size_t recvSize = recvCount * sizeof(float);

// Allocate device memory to receive the Scatter result.
ACLCHECK(aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY));
// On the root node, allocate device memory to store data to be sent.
if (device == rootRank) {
    ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));
}

// Initialize the communicator.
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, device, &hcclComm);

// Create a task stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Execute Scatter. The root node's data in the communicator is evenly divided and scattered to other ranks.
HcclScatter(sendBuf, recvBuf, recvCount, HCCL_DATA_TYPE_FP32, rootRank, hcclComm, stream);
// Block until the collective communication tasks in the task stream are complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free device-side memory.
aclrtFree(recvBuf);          // Free device-side memory.
aclrtDestroyStream(stream);  // Destroy the task stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
