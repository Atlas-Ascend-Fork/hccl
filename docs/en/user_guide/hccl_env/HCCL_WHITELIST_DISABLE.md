# HCCL_WHITELIST_DISABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T04:03:05.818Z pushedAt=2026-10-15T08:46:37.234Z -->

## Function

Whether to enable the communication trustlist when using HCCL.

- `0`: Yes. Only IP addresses in the trustlist are allowed to perform collective communication.
- `1`: No. The trustlist is not verified.

The default value is `1`. If trustlist verification is enabled, specify the trustlist configuration file path through [HCCL_WHITELIST_FILE](HCCL_WHITELIST_FILE.md).

## Configuration Example

```bash
export HCCL_WHITELIST_DISABLE=1
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
