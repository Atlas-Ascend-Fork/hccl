# HCOMM_TA_CTP_UB_TIMEOUT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T04:05:05.190Z pushedAt=2026-10-15T08:46:37.244Z -->

## Function

Sets the coefficient (`timeout`) for configuring the jetty timeout under the UB_CTP protocol.

For Ascend 950PR&950DT products, this environment variable is configured as an integer ranging from 0 to 31, which defaults to `8`.

The jetty timeout under the UB_CTP protocol falls into four levels. Levels are calculated as: timeout / 8 (integer division), where `timeout` is the value of this environment variable. Level 0: 512 ms; level 1: 4s; level 2: 8s; level 3: 32s. The UB_CTP protocol directly uses the value of this environment variable, without comparing it with the total TP timeout. Values 0, 8, 16, and 24 are recommended.

## Configuration Example

```bash
# If the timeout coefficient for the UB_CTP protocol is set to 8, the timeout level is: 8 / 8 = 1, corresponding to 4s.
export HCOMM_TA_CTP_UB_TIMEOUT=8
```

## Constraints

None

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
