# Rank Table File device_ip Field Verification Failure (EI0014)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:54:15.330Z pushedAt=2026-10-14T02:35:13.432Z -->

## Symptoms

The CANN log contains the keyword `the IP address(***) in the ranktable is inconsistent with the IP(***)address of the network adapter`, as shown below:

```text
[ERROR] HCCP(166192,eExecutor):2025-01-21-16:59:39.962.565 [ra_host.c:480]tid:167056,ra_rdev_init_check_ip(480) : [check][ip]fail, ret(129) the IP address(127.10.0.0) in the ranktable is inconsistent with the IP address(127.10.0.1) of the network adapter, please make sure they're consistent. num(2)
```

## Possible Causes

During device IP verification, HCCL detects that the device IP obtained on the current device side is inconsistent with the device IP configured for the current rank in the rank table, causing the verification to fail.

For example, on rank0, the device IP corresponding to the bound device is 127.10.0.1, but the device IP configured for rank0 in the rank table is 127.10.0.0, which causes the HCCL verification to fail.

## Solution

Check whether the device IP configured in the rank table is consistent with the device IP used by each rank in the communicator.
