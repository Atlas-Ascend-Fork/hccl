# HCCL_UB_MULTI_CHANNEL_NUM

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T04:03:13.340Z pushedAt=2026-10-15T08:46:37.236Z -->

## Function

For communication using the CLOS topology and UB protocol (CTP/RTP), you can use this environment variable to configure the number of channels established between the same EID pair.

This environment variable must be set to an integer ranging from 1 to 16, which defaults to `1`.

Each channel corresponds to an independent jetty at the underlying layer. Multiple channels move data in parallel to improve communication bandwidth (similar to the RoCE multi-QP mechanism). The default value `1` means multi-channel is disabled.

## Configuration Example

```bash
export HCCL_UB_MULTI_CHANNEL_NUM=4
```

## Constraints

- This environment variable takes effect only when CLOS topology and UB protocol (CTP/RTP) are used and the algorithm runs on the AICPU engine;
- This environment variable supports only collective communication operators (the communicator configuration is read through HcclConfigGetInfo);
- If the value is out of the range \[1,16\], an error is reported during HCCL initialization or value reading (fault code EI0001 is reported during parsing, and HCCL_E_PARA is returned during reading).
- Each channel occupies hardware resources such as jetties. Setting an excessively large number of channels may squeeze the resources available to other communicators. Set the value as needed (for example, 2 or 4).
- Performance notice: Increasing the number of channels brings benefits only when a single jetty throttles the communication bandwidth. If a single jetty has reached the link bandwidth limit, multiple channels bring no gain. Set this environment variable based on performance verification results.

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
