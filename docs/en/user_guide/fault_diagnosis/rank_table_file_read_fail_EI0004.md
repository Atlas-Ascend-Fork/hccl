# Rank Table File Read Failure (EI0004)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:54:14.903Z pushedAt=2026-10-14T02:47:15.902Z -->

## Symptoms

The CANN log contains the keyword "is not a valid real path", as shown below:

```text
[ERROR] HCCL(1104629,test_one_side):2025-10-28-17:45:13.679.684 [param_check.cc:66] [1104629][InitGroupStage][RanktableConfig]errNo[0x0000000005010001] path /ranktable.json is not a valid real path
```

## Possible Causes

When a communicator is initialized based on a rank table file, the specified rank table file path does not exist or has insufficient permissions.

## Solution

Correct the rank table file path or configure the correct read permission.
