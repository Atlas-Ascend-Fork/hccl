# HCCL_RDMA_RETRY_CNT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T03:59:07.982Z pushedAt=2026-10-15T08:46:37.218Z -->

## Function

Configures the retry count for RDMA NICs. The value must be an integer ranging from 1 to 7, and defaults to 7.

## Configuration Example

```bash
# Set the retry count to 5.
export HCCL_RDMA_RETRY_CNT=5
```

## Constraints

None

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
