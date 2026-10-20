# HcclBatchSendRecv

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-19T07:25:35.772Z pushedAt=2026-10-19T10:17:19.064Z -->

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

Asynchronous batch point-to-point communication API. A single API call can complete multiple send and receive tasks on the current rank. The send and receive operations on the current rank are asynchronous, meaning send and receive tasks do not block each other.

## Prototype

```c
HcclResult HcclBatchSendRecv(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendRecvInfo | Input | Pointer to the list of send/receive tasks to be issued by this rank.<br>HcclSendRecvItem type. For details, see [HcclSendRecvItem](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclSendRecvItem.md). |
| itemNum | Input | Number of tasks to be received and sent by this rank. |
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
| HCCL_E_PTR | An input pointer parameter is null, for example, comm, sendRecvInfo, or stream is nullptr. |
| HCCL_E_PARA | An input parameter is invalid. For example, the count of an item in sendRecvInfo exceeds the upper limit, or remoteRank is out of range. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, the current model does not support it or dataType is invalid. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- "Asynchronous" means that the receive and send tasks on the same device are asynchronous and do not block each other. However, between devices, send and receive tasks are still synchronous. Therefore, inter-device send and receive tasks must be in one-to-one correspondence, just like HcclSend and HcclRecv.
- For Atlas A2 products, when this API is used in a large-scale cluster (ranksize > 500), the number of concurrent executions cannot exceed 3.
- For [Atlas 200T A2 Box16](https://support.huawei.com/enterprise/en/doc/EDOC1100318274/287e0458), if a link establishment failure occurs between devices within a server (error code: EI0010), set the environment variable `HCCL_INTRA_ROCE_ENABLE` to `1` and `HCCL_INTRA_PCIE_ENABLE` to `0`, so that the RoCE loop is used for multi-device communication within the server (ensure that a RoCE NIC exists on the server and that the RDMA links between devices with send/recv relationships are reachable). Example environment variable configuration:

    ```bash
    export HCCL_INTRA_ROCE_ENABLE=1
    export HCCL_INTRA_PCIE_ENABLE=0
    ```
    
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

// Execute Send/Recv to send data to the next node and receive data from the previous node.
// HcclBatchSendRecv can issue multiple send/recv tasks on the current rank at the same time.
uint32_t next = (deviceId + 1) % count;
uint32_t prev = (deviceId - 1 + count) % count;
HcclSendRecvItem sendRecvInfo[2];
sendRecvInfo[0] = HcclSendRecvItem{HCCL_SEND, sendBuf, count, HCCL_DATA_TYPE_FP32, next};
sendRecvInfo[1] = HcclSendRecvItem{HCCL_RECV, recvBuf, count, HCCL_DATA_TYPE_FP32, prev};
HcclBatchSendRecv(sendRecvInfo, 2, hcclComm, stream);

// Block and wait for collective communication tasks in the task stream to complete.
ACLCHECK(aclrtSynchronizeStream(stream));

// Release resources.
aclrtFree(sendBuf);          // Free device-side memory.
aclrtFree(recvBuf);          // Free device-side memory.
aclrtDestroyStream(stream);  // Destroy the task stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
