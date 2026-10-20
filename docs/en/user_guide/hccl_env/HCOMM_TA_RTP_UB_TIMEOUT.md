# HCOMM_TA_RTP_UB_TIMEOUT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T04:07:45.693Z pushedAt=2026-10-15T08:46:37.264Z -->

## Function

Sets the coefficient (`timeout`) of the jetty timeout duration under the UB_RTP protocol.

For Ascend 950PR&950DT products, this environment variable is set to an integer ranging from 0 to 31, which defaults to `16`.

The jetty timeout duration under the UB_RTP protocol falls into 4 levels. Levels are calculated as: timeout / 8 (integer division), where `timeout` is the value of this environment variable. Level 0: 512 ms; Level 1: 4s; Level 2: 8s; Level 3: 32s. The software has a built-in interception check mechanism. Before a jetty is created, the TP timeout configuration is queried. If the timeout duration configured by this environment variable is less than or equal to the TP total timeout, the jetty timeout duration is automatically raised to the minimum level greater than the TP total timeout. If the timeout duration configured by this environment variable is greater than the TP total timeout, the level matching this environment variable is used directly. Values 0, 8, 16, and 24 are recommended.

## Configuration Example

```bash
# If the timeout coefficient for the UB_RTP protocol is set to 16, the timeout duration level is: 16 / 8 = 2, corresponding to 8s.
export HCOMM_TA_RTP_UB_TIMEOUT=16
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
