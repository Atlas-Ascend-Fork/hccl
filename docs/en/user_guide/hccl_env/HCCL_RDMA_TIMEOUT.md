# HCCL_RDMA_TIMEOUT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-06T09:48:55.447Z pushedAt=2026-11-16T06:28:10.210Z nl=2 -->

## Function

Configures the coefficient timeout for the RDMA NIC retry timeout.

The formula for the minimum RDMA NIC retry timeout is: *4.096 μs x 2^timeout*, where `timeout` is the configured value of this environment variable, and the actual retry timeout depends on your network conditions.

<!-- npu="950" id10 -->
- For Ascend 950PR&950DT products:
  - For Atlas 350 accelerator cards, when using a custom RDMA NIC, this environment variable is an integer that ranges from 0 to 31 and defaults to 20. Setting it to `0` or >=32 indicates no timeout.
    > The value of this environment variable is the exponential value of the NACK retry interval for verbs, consistent with the algorithm defined by the verbs API. It is configured by you based on the specifications of the selected RDMA NIC.
  - For non-Atlas 350 accelerator cards, when using an SP561 NIC based on Hi1825, set this environment variable to an integer, with a value range from 18 to 31. The default value is 20.
    > If the value is set within 0 to 17 or is greater than or equal to 32, it is processed as 18.
<!-- end id10 -->
<!-- npu="A3" id9 -->
- For Atlas A3 products, set this environment variable to an integer, with a value range from 5 to 20. The default value is 20.
<!-- end id9 -->
<!-- npu="910b" id8 -->
- For Atlas A2 products, set this environment variable to an integer, with a value range from 5 to 20. The default value is 20.
<!-- end id8 -->
<!-- npu="910" id1 -->
- For Atlas training products, set this environment variable to an integer, with a value range from 5 to 24. The default value is 20.
<!-- end id1 -->
<!-- npu="310p" id2 -->
- For Atlas inference products, set this environment variable to an integer, with a value range from 5 to 24. The default value is 20.
<!-- end id2 -->

## Configuration Example

```bash
# Set the coefficient of the RDMA NIC retransmission timeout to 6. When the NIC enables RDMA, the minimum retransmission timeout is: 4.096μs * 2^6
export HCCL_RDMA_TIMEOUT=6
```

## Constraints

None.

## Applicable Products

<!-- npu="950" id5 -->
- Ascend 950PR&950DT products: Supported
<!-- end id5 -->
<!-- npu="A3" id6 -->
- Atlas A3 products: Supported
<!-- end id6 -->
<!-- npu="910b" id7 -->
- Atlas A2 products: Supported
<!-- end id7 -->
<!-- npu="910" id3 -->
- Atlas training products: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Supported
<!-- end id4 -->
