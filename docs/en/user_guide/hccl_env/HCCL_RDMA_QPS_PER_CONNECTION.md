# HCCL_RDMA_QPS_PER_CONNECTION

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-06T09:48:27.275Z pushedAt=2026-11-16T06:28:10.197Z nl=2 -->

## Function

During RDMA communication between two ranks, one QP (Queue Pair) is created by default for data transmission. If you want to use multiple QPs for RDMA communication between two ranks, you can configure this environment variable.

This environment variable specifies the number of QPs to be used between two ranks. It must be configured as an integer. Value range: [1, 32]. Recommended range: [1, 8]. When the number of QPs exceeds 8, performance gains cannot be guaranteed, and excess memory use may cause service execution failures. Default value: 1.

Assume `HCCL_RDMA_QPS_PER_CONNECTION` is set to `N1`. N1 QPs will be created between every two ranks, and the service data transmitted between the two ranks via RDMA will be evenly distributed across the N1 QPs for parallel sending and receiving.

After enabling multi-QP transmission, you can set the minimum threshold of data volume per QP through [HCCL_MULTI_QP_THRESHOLD](HCCL_MULTI_QP_THRESHOLD.md). If you want to specify the source port number used by each QP, you can do so through [HCCL_RDMA_QP_PORT_CONFIG_PATH](HCCL_RDMA_QP_PORT_CONFIG_PATH.md).

## Configuration Example

```bash
export HCCL_RDMA_QPS_PER_CONNECTION=4
```

## Constraints

- This environment variable supports only single-operator calls and does not support static graphs.
- The priorities of QP-related configurations are as follows:

    Environment variable HCCL_RDMA_QP_PORT_CONFIG_PATH \> environment variable HCCL_RDMA_QPS_PER_CONNECTION.

<!-- npu="A3,910b" id6 -->
- For Atlas A3 products and Atlas A2 products, QP-related configuration can also be set through the hccn_tool. The priority between the hccn_tool configuration and this environment variable is as follows:

    Management-plane multi-QP configuration (configured via the "-s multi_qp" parameter of hccn_tool) \> NSLB QP configuration (configured via the "-t nslb-dp" parameter of hccn_tool)\> environment variable HCCL_RDMA_QP_PORT_CONFIG_PATH \> environment variable HCCL_RDMA_QPS_PER_CONNECTION.
<!-- end id6 -->

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id3 -->
- Atlas A3 products: Supported
<!-- end id3 -->
<!-- npu="910b" id2 -->
- Atlas A2 products: Supported
<!-- end id2 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
