# Rank Table Field Configuration Error (EI0004)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-09T03:28:12.925Z pushedAt=2026-11-11T09:05:03.688Z nl=2 -->

## Symptoms

For Atlas A3 products, the CANN log contains the keyword "RanktableCheck", as shown below:

```text
[ERROR] HCCL(1265,):2025-10-21 07:56:47.198.454 [topoinfo_ranktableConcise.cc:727][15326][InitGroupStage][RanktableCheck]errNo[0x0000000005010001] super_device_id[] is invalid
```

## Possible Causes

The "version" field in the rank table is "1.2", but the "super_device_id" field in the rank table is left empty, causing the rank table verification to fail.

## Solution

Add the "super_device_id" field in the rank table file. For configuration instructions, see [Rank Table Configuration Resource Information (Atlas A3 Products)](../cluster_info_config/rank_table_config_a3.md).
