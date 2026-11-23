# HCCL_INTER_HCCS_DISABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-06T09:47:18.682Z pushedAt=2026-11-16T06:28:10.169Z nl=2 -->

## Function

This environment variable is used to configure the communication link type within a SuperPoD in SuperPoD networking. The following values are supported:

- `TRUE`: AI nodes within the SuperPoD use RoCE for RDMA communication.
- `FALSE`: AI nodes within the SuperPoD use HCCS communication links for SDMA communication.

The default value is `FALSE`.

## Configuration Example

```bash
export HCCL_INTER_HCCS_DISABLE=FALSE
```

## Constraints

None.

## Applicable Products

<!-- npu="950" id2 -->
- Ascend 950PR&950DT products: Not supported
<!-- end id2 -->
<!-- npu="A3" id1 -->
- Atlas A3 products: Supported
<!-- end id1 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
