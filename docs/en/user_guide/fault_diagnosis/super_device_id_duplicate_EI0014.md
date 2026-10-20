# Duplicate superDeviceId (EI0014)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:55:00.928Z pushedAt=2026-10-14T03:02:37.863Z -->

## Symptoms

The CANN log contains the keyword "superDeviceId\[\*\*\*\] in superPod\[\*\*\*\]is already exist", as shown below:

```text
[ERROR] HCCL(169030,alltoall_test):2025-10-23-16:28:59.392.635 [topoinfo_exchange_agent.cc:695] [169030][InitGroupStage][RanktableCheck]devices have same superDeviceId[0x3000000] in superPod[super_pod_id_0]. Current device info: serverId[127.10.0.1], rankId[0], group[hccl_world_group]. Another device info: rankId[1].
```

## Possible Causes

The superDeviceId is the physical ID of a device within an Atlas A3 product in a SuperPoD system, serving as the unique identifier of the device in the SuperPoD system. During the consistency check, HCCL detected duplicate superDeviceId values within a SuperPoD, causing the check to fail. The superDeviceId can be queried using the npu-smi command:

```bash
npu-smi info -t spod-info -i id -c chip_id
```

- id: Device ID. The NPU ID obtained through the `npu-smi info -l` command is the device ID.
- chip_id: Chip ID. The Chip ID obtained through the `npu-smi info -m` command is the chip ID.

The "SDID" field in the command output is the superDeviceId.

Possible causes of this issue:

- Abnormal hardware configuration.
- Different physical SuperPoDs are configured in the same logical SuperPoD through the [HCCL_LOGIC_SUPERPOD_ID](../hccl_env/HCCL_LOGIC_SUPERPOD_ID.md) environment variable, causing duplicate superDeviceId.

### Solution

Modify the hardware configuration or correctly configure the [HCCL_LOGIC_SUPERPOD_ID](../hccl_env/HCCL_LOGIC_SUPERPOD_ID.md) environment variable to prevent devices with the same superDeviceId from appearing within the same superpod.
