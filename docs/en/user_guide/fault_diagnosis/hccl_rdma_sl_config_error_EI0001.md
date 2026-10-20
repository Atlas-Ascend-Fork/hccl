# HCCL_RDMA_SL Configuration Error (EI0001)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:52:34.189Z pushedAt=2026-10-13T09:34:46.731Z -->

## Symptoms

The log contains the keyword `EI0001` or `Value *** for environment variable *** is invalid`, as shown below:

```text
[PID:3729526]2025-10-23-17:30:40.098.984Config_Error_Invalid_Environment_Variable(EI0001): Value 1000 for environment variable HCCL_RDMA_SL is invalid. Expected value : range[0, 7].
```

<!-- npu="A3,910b,910,310p" id4 -->
For the following products, the keyword "externalinput.cc" in the ERROR logs of the CANN logs indicates that an error is reported when reading the environment variable configuration.

  <!-- npu="A3" id1 -->
- Atlas A3 products
  <!-- end id1 -->
  <!-- npu="910b" id2 -->
- Atlas A2 products
  <!-- end id2 -->
  <!-- npu="910" id5 -->
- Atlas training products
  <!-- end id5 -->
  <!-- npu="310p" id3 -->
- Atlas inference products
  <!-- end id3 -->

An example of the error is as follows:

```text
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.098.973 [externalinput.cc:963] [3729526][Parse][rdmaServerLevel]HCCL_RDMA_SL[1000] is invalid. except: [0, 7]
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.099.058 [externalinput.cc:169] [3729526][InitGroupStage][EnvConfig]errNo[0x0000000005000001] In init env variable param, parse HCCL_RDMA_SL failed. errno[1]
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.099.063 [externalinput.cc:47] [3729526][InitExternalInput]call trace: hcclRet -> 1
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.099.068 [op_base.cc:866] [3729526][HcclGetRootInfo]call trace: hcclRet -> 1
```
<!-- end id4 -->

## Possible Causes and Solutions

If the environment variable configuration parameter does not meet the requirements, adjust the value range based on the suggestions printed in the log. If questions remain, refer to the corresponding [Environment Variable Reference](../hccl_env/README.md).
