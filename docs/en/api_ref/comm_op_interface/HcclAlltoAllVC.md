# HcclAlltoAllVC

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-19T07:24:32.925Z pushedAt=2026-10-19T10:17:19.056Z -->

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
- Atlas training products: Not supported
<!-- end id5 -->

## Function

The operation API of the collective communication operator AlltoAllVC, which sends data (with customizable data volume) to all ranks in the communicator and receives data from all ranks. Compared with AlltoAllV, AlltoAllVC passes the send/receive parameters of all ranks through the input parameter `sendCountMatrix`.

![alltoallvc](figures/alltoallvc.png)

## Prototype

```c
HcclResult HcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType, const void *recvBuf, HcclDataType recvType, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| sendCountMatrix | Input | Two-dimensional uint64 array indicating the amount of data to send. The array shape is `[rankSize][rankSize]`, where `sendCountMatrix[i][j] = n` indicates that the amount of data sent from rank i to rank j is n.<br>For example, if `sendType` is float32, `sendCountMatrix[i][j] = n` indicates that rank i sends n float32 values to rank j. |
| sendType | Input | Data type of the data to send. The value is of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Supported data types vary by model. For details, see [Supported Data Types](#supported-data-types).|
| recvBuf | Output | Destination data buffer address, to which the collective communication result is output.<br>The addresses configured for recvBuf and sendBuf cannot be the same, and their memory ranges cannot overlap. |
| recvType | Input | Data type of the data to receive. The value is of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>The supported data types vary by product model. For details, see [Supported Data Types](#supported-data-types). |
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

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | The API call succeeds. |
| HCCL_E_PTR | An input pointer parameter is null, for example, comm, sendBuf, recvBuf, stream, or sendCountMatrix is nullptr. |
| HCCL_E_PARA | An input parameter is invalid, for example, sendBuf and recvBuf have the same address. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, the current model does not support it or dataType is invalid. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

The performance of the AlltoAllVC operation depends on the buffer size for sharing data between NPUs. When the communication data volume exceeds the buffer size, performance will degrade significantly. If the AlltoAllVC communication data volume is large in your service, you are advised to configure the environment variable [HCCL_BUFFSIZE](../../user_guide/hccl_env/HCCL_BUFFSIZE.md) to appropriately increase the buffer size to improve communication performance.

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

// Create a stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Set the amounts of data to send and receive. The send and receive amounts are the same.
std::vector<uint64_t> sendCountMatrix(rankSize * rankSize);
for (uint32_t i = 0; i < rankSize; ++i) {
    for (uint32_t j = 0; j < rankSize; ++j) {
        sendCountMatrix[i * rankSize + j] = count / rankSize;
    }
}

// Execute AlltoAllVC to send the same amount of data to all ranks in the communicator and receive the same amount of data from all ranks. The data amounts are customizable.
HcclAlltoAllVC(sendBuf, sendCountMatrix.data(), HCCL_DATA_TYPE_FP32, recvBuf, HCCL_DATA_TYPE_FP32, hcclComm, stream);

// Block and wait for the collective communication tasks in the stream to complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free device memory.
aclrtFree(recvBuf);          // Free device memory.
aclrtDestroyStream(stream);  // Destroy the stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
