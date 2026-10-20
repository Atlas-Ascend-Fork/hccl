# HcclAlltoAllV

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-19T07:24:51.515Z pushedAt=2026-10-19T10:17:19.059Z -->

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

The operation API for the collective communication operator AlltoAllV, which sends data (with customizable data volume) to all ranks in the communicator and receives data from all ranks.

![alltoallv](figures/alltoallv.png)

## Prototype

```c
HcclResult HcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| sendCounts | Input | uint64 array indicating the amount of data to send, where `sendCounts[i] = n` indicates that the amount of data sent by this rank to rank i is n.<br>For example, if `sendType` is float32, `sendCounts[i] = n` indicates that this rank sends n float32 data elements to rank i. |
| sdispls | Input | uint64 array indicating the send offset, where `sdispls[i] = n` indicates the offset, relative to sendBuf, of the start position in sendBuf of the data that this rank sends to rank i, in units of sendType. |
| sendType | Input | Data type of the data to send. The value is of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Supported data types vary by model. For details, see [Supported Data Types](#supported-data-types).|
| recvBuf | Output | Destination data buffer address, to which the collective communication result is output.<br>The addresses configured for recvBuf and sendBuf cannot be the same, and their memory ranges cannot overlap. |
| recvCounts | Input | uint64 array indicating the amount of data to receive, where `recvCounts[i] = n` indicates that the amount of data received by this rank from rank i is n.<br>For example, if `recvType` is float32, `recvCounts[i] = n` indicates that this rank receives n float32 data items from rank i. |
| rdispls | Input | uint64 array indicating the receive offset, where `rdispls[i] = n` indicates the offset, relative to recvBuf, of the start position in recvBuf where the data that this rank receives from rank i is stored, in units of recvType. |
| recvType | Input | Data type of the data to receive. The value is of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Supported data types vary by model. For details, see [Supported Data Types](#supported-data-types).|
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
| HCCL_E_PTR | An input pointer parameter is null, for example, comm, sendBuf, recvBuf, stream, sendCounts, sdispls, recvCounts, or rdispls is nullptr. |
| HCCL_E_PARA | An input parameter is invalid, for example, sendBuf and recvBuf have the same address. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, the current model does not support it or dataType is invalid. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- The performance of the AlltoAllV operation depends on the buffer size for data sharing between NPUs. When the communication data volume exceeds the buffer size, performance degrades significantly. If the AlltoAllV communication data volume is large in your service, configure the environment variable [HCCL_BUFFSIZE](../../user_guide/hccl_env/HCCL_BUFFSIZE.md) to increase the buffer size appropriately to improve communication performance.
<!-- npu="910" id7 -->
- For Atlas training products, the communicator of AlltoAllV must meet the following constraints:

  In cluster networking, single-server 1p and 2p communicators must be within the same cluster (devices 0-3 and devices 4-7 within a server each form a cluster). For single-server 4p, single-server 8p, and multi-server communicators, ranks must be organized by cluster as the basic unit, and the cluster selection must be consistent across servers.
  
- For Atlas training products, in single-server use cases, the NIC status must be `up`; otherwise, this API will fail to execute.
<!-- end id7 -->
<!-- npu="310p" id14 -->
- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of two Atlas 300I Duo inference cards (i.e., four NPUs) per server.
<!-- end id14 -->
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

// Set the send and receive data sizes, which are the same.
std::vector<uint64_t> sendCounts(rankSize, 1);
std::vector<uint64_t> recvCounts(rankSize, 1);
std::vector<uint64_t> sdispls(rankSize);
std::vector<uint64_t> rdispls(rankSize);
for (size_t i = 0; i < rankSize; ++i) {
    sdispls[i] = i;
    rdispls[i] = i;
}
// Execute AlltoAllV to send the same amount of data to all ranks in the communicator and receive the same amount of data from all ranks. The data size can be customized.
HcclAlltoAllV(sendBuf, sendCounts.data(), sdispls.data(), HCCL_DATA_TYPE_FP32,
              recvBuf, recvCounts.data(), rdispls.data(), HCCL_DATA_TYPE_FP32, hcclComm, stream);
// Block and wait until the collective communication tasks in the task stream are complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free device memory.
aclrtFree(recvBuf);          // Free device memory.
aclrtDestroyStream(stream);  // Destroy the task stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
