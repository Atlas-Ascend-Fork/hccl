# HCCL_LOGIC_SUPERPOD_ID

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T03:55:40.548Z pushedAt=2026-10-15T08:46:37.193Z -->

## Function

In SuperPoD networking on Atlas A3 products, if the rank table file is not used to configure cluster resource information, set this environment variable to specify the SuperPoD ID to which the process running on the current node belongs. This divides a physical SuperPoD into multiple logical SuperPoDs.

The value of this environment variable is a string with fewer than 128 characters in length, and defaults to null.

If this environment variable is not configured, the value of `Super Pod ID` in the environment is obtained as the SuperPoD ID. You can query this value running `npu-smi info -t spod-info -i <id> -c <chip_id>`.

## Configuration Example

```bash
export HCCL_LOGIC_SUPERPOD_ID=super_pod_id_1
```

## Constraints

- This environment variable applies only when no rank table is used to configure cluster information in SuperPoD networking. If a rank table is used, the configuration in the rank table takes precedence.
- This environment variable is used to divide a physical SuperPoD into multiple logical SuperPoDs. It does not support configuring ranks that belong to different physical SuperPoDs into a single logical SuperPoD.

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
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
