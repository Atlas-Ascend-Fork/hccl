# HCCL_INTRA_ROCE_ENABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T03:56:53.900Z pushedAt=2026-10-15T08:46:37.200Z -->

## Function

Configures whether to use RoCE links for intra-server or intra-SuperPoD communication.

  <!-- npu="910b,910" id5 -->
- For Atlas training products and Atlas A2 products, this environment variable specifies whether RoCE links are used for communication within a server, and defaults to `0`. It can be configured separately or together with HCCL_INTRA_PCIE_ENABLE. The following table lists the supported configuration combinations and the communication links used within a server for each combination:

   Supported combinations of HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE

  | HCCL_INTRA_PCIE_ENABLE | HCCL_INTRA_ROCE_ENABLE | Intra-Server Communication Link |
  | --- | --- | --- |
  | 1 | Not set | PCIe |
  | 1 | 0 | PCIe |
  | 0 | 1 | RoCE |
  | Not set | 1 | RoCE |
  | 0 | 0 | PCIe |
  | Not set | Not set | PCIe |

    > [!NOTE]
    > - HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE can't both be 1.
    > - HCCL_INTRA_PCIE_ENABLE can't be 0 when HCCL_INTRA_ROCE_ENABLE is not set.
    > - HCCL_INTRA_PCIE_ENABLE must be set when HCCL_INTRA_ROCE_ENABLE is 0.
  <!-- end id5 -->

  <!-- npu="A3" id1 -->
- For Atlas A3 products, this environment variable takes effect only when LLM-DataDist is used as the cluster management component. It specifies whether RoCE links are used for communication within a SuperPoD, and defaults to `0`. The configuration is as follows:
  - `0`: The default HCCS links or PCIe links are used for intra-SuperPoD communication (including both LLM-DataDist communication and HCCL communication).
  - `1`: For Atlas 800T A3, Atlas 800I A3, and Atlas 900 A3 SuperPoD, LLM-DataDist communication within a SuperPoD uses RoCE links, and HCCL communication is not affected. For A200T A3 Box8, both LLM-DataDist and HCCL communication use RoCE links.
  <!-- end id1 -->

## Configuration Example

```bash
export HCCL_INTRA_ROCE_ENABLE=1
```

<!-- npu="910b" id6 -->
## Constraints

[Atlas 200T A2 Box16](https://support.huawei.com/enterprise/en/doc/EDOC1100318274/287e0458) has left and right modules, which house devices 0-7 and devices 8-15, respectively. For this product:

**In single-node use cases**, when PCIe links are used for intra-server communication, if devices from both modules need to be used simultaneously, the two modules must use the same number of devices and be in the same plane, that is, device 0 and device 8, device 1 and device 9 (and so on) must be used together. When RoCE links are used for intra-server communication, this restriction does not apply.
<!-- end id6 -->

## Applicable Products

<!-- npu="950" id7 -->
- Ascend 950PR&950DT products: Not supported
<!-- end id7 -->
<!-- npu="A3" id3 -->
- Atlas A3 products: Supported only when LLM-DataDist is used as the cluster management component.
<!-- end id3 -->
<!-- npu="910b" id4 -->
- Atlas A2 products: Only [Atlas 200T A2 Box16](https://support.huawei.com/enterprise/en/doc/EDOC1100318274/287e0458) is supported.
<!-- end id4 -->
<!-- npu="910" id2 -->
- Atlas training products: Only [Atlas 300T Pro](https://support.huawei.com/enterprise/en/ascend-computing/atlas-300t-pro-pid-256118195) is supported.
<!-- end id2 -->
<!-- npu="310p" id8 -->
- Atlas inference products: Not supported
<!-- end id8 -->
