# HCCL_SOCKET_IFNAME Configuration Error (EI0001)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:52:40.497Z pushedAt=2026-10-13T09:43:05.203Z -->

## Symptoms

In the CANN log, the keyword "get host ip fail by socket Ifname" is found, as shown below:

```text
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.432 [sal.cc:501] [925892][InitGroupStage][EnvConfig]set ifname to [abc] by HCCL_SOCKET_IFNAME, but not found in the environment, ifnames in the environment is as follows
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.437 [sal.cc:504] [925892][InitGroupStage][EnvConfig]get host ip fail by socket Ifname. name[lo] ip[127.10.0.1%lo]
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.441 [sal.cc:504] [925892][InitGroupStage][EnvConfig]get host ip fail by socket Ifname. name[enp] ip[127.10.0.2%enp]
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.447 [sal.cc:504] [925892][InitGroupStage][EnvConfig]get host ip fail by socket Ifname. name[docker0] ip[172.17.0.1%docker0]
```

## Root Cause

The Host NIC was specified through the HCCL_SOCKET_IFNAME environment variable, but the corresponding NIC was not found in the current environment. (In a container scenario, a Host NIC available inside the container must be specified.) The error log lists the Host NICs detected in the current environment.

## Solution

Modify the HCCL_SOCKET_IFNAME environment variable and set it to a Host NIC that exists in the environment.
