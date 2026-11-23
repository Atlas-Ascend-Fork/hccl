# HcclRecv

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-06T09:44:08.378Z pushedAt=2026-11-16T06:28:10.082Z nl=2 -->

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

<!-- npu="910b" id6 -->
> [!NOTE]
> For Atlas A2 products, only Atlas 800T A2, Atlas 900 A2 PoD, and Atlas 200T A2 Box16 are supported.
<!-- end id6 -->

## Function

Point-to-point communication Receive operation API, which receives data from the source node to the specified location on the current node.

## Prototype

```c
HcclResult HcclRecv(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| recvBuf | Output | Address of the buffer to receive data. |
| count | Input | Number of data elements to receive. |
| dataType | Input | Type of the data to receive, of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Supported data types vary by product. For details, see [Supported Data Types](#supported-data-types).|
| srcRank | Input | Rank ID of the data sender in the communicator. |
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

## Return Values

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | The API call succeeds. |
| HCCL_E_PTR | An input pointer parameter is null, for example, comm or recvBuf is nullptr. |
| HCCL_E_PARA | An input parameter is invalid, for example, count exceeds the upper limit or srcRank is out of range. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, the current product model does not support it, dataType is invalid, or self-send and self-receive is not supported when srcRank equals the local rank. |
| HCCL_E_INTERNAL | Internal error. |
| HCCL_E_UNAVAIL | A resource is unavailable, for example, a CCU resource reuse exception (null reuse thread or excess threads). |
| HCCL_E_RUNTIME | A call to an underlying runtime API (for example, memory copy or memory zeroing) fails. |
| HCCL_E_MEMORY | Memory allocation or operation fails. |

## Constraints

The HcclSend and HcclRecv APIs use a synchronous call approach. They work in pairs. That is, after a process calls the HcclSend API, it must wait for the paired HcclRecv API to receive the data before making the next API call, as shown in the following figure.

![](figures/send_recv.png)

- All communication operators in multiple communicators must be dispatched serially on each device. Out-of-order dispatch, multi-thread concurrent dispatch, and thread reentrancy are not supported.
- On the same device, the dispatch threads of all communication operators in the same communicator must use the same context.
<!-- npu="950" id14 -->
- For Ascend 950PR&950DT products, self receive is not supported. `HCCL_E_NOT_SUPPORT` is returned when `srcRank` equals the current rank.
<!-- end id14 -->


## Example

```c
void *sendBuf = nullptr;
void *recvBuf = nullptr;
uint64_t count = 8;
size_t mallocSize = count * sizeof(float);

// Initialize the communicator.
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Create a task stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Perform Send/Recv operations. Devices 0/2/4/6 send data, and devices 1/3/5/7 receive data.
// HcclSend and HcclRecv are synchronous APIs and must be used in pairs.
if (deviceId % 2 == 0) {
    // Allocate device memory for storing input data.
    aclrtMalloc(&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
    // Initialize the input data.
    aclrtMemcpy(sendBuf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE);
    // Perform the Send operation.
    HcclSend(sendBuf, count, HCCL_DATA_TYPE_FP32, deviceId + 1, hcclComm, stream);
} else {
    // Allocate device memory for receiving data.
    aclrtMalloc(&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
    // Perform the Recv operation.
    HcclRecv(recvBuf, count, HCCL_DATA_TYPE_FP32, deviceId - 1, hcclComm, stream);
}

// Block and wait until the collective communication tasks in the task stream complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free the device memory.
aclrtFree(recvBuf);          // Free the device memory.
aclrtDestroyStream(stream);  // Destroy the stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
