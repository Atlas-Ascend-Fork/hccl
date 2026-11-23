# HCCL_RDMA_SL

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-06T09:48:34.309Z pushedAt=2026-11-16T06:28:10.202Z nl=2 -->

## Function

Configures the service level for RDMA NICs. This value must be consistent with the PFC priority configured on the NIC. Inconsistent configuration may cause performance degradation.

This environment variable must be set to an integer ranging from 0 to 7 and defaults to 4.

## Configuration Example

```bash
# Set the priority to 3.
export HCCL_RDMA_SL=3
```

## Constraints

If you call the HCCL C API to initialize a communicator with specific configurations, and the RDMA NIC service level is configured through `hcclRdmaServiceLevel` of `HcclCommConfig`, the communicator-level configuration takes precedence.

## Applicable Products

<!-- npu="950" id3 -->
- Ascend 950PR&950DT products: Supported
<!-- end id3 -->
<!-- npu="A3" id4 -->
- Atlas A3 products: Supported
<!-- end id4 -->
<!-- npu="910b" id5 -->
- Atlas A2 products: Supported
<!-- end id5 -->
<!-- npu="910" id1 -->
- Atlas training products: Supported
<!-- end id1 -->
<!-- npu="310p" id2 -->
- Atlas inference products: Supported
<!-- end id2 -->
