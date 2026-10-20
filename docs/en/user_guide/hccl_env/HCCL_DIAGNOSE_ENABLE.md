# HCCL_DIAGNOSE_ENABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T03:51:48.485Z pushedAt=2026-10-15T08:46:37.163Z -->

## Function

This environment variable is used to configure whether to cache detailed information of some tasks during collective communication, so that when a task fails, detailed logs are printed for issue locating.

The following values are supported:

- `1`: Enables collective communication caching.
- `0`: Disables collective communication caching.

Defaults to `0`.

Note that enabling this environment variable affects performance.

## Configuration Example

```bash
export HCCL_DIAGNOSE_ENABLE=1
```

## Constraints

A maximum of 2,000 latest operator information entries can be saved.

## Applicable Products

<!-- npu="950" id3 -->
- Ascend 950PR&950DT products: Not supported
<!-- end id3 -->
<!-- npu="A3" id1 -->
- Atlas A3 products: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2 products: Supported
<!-- end id2 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
