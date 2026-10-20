# IP Family Verification Inconsistency (EI0001)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:52:52.102Z pushedAt=2026-10-13T09:44:26.943Z -->

## Symptoms

The CANN log contains the keyword "rank\[\*\] device ip family\[2\] is not same as others\[\*\]." as shown below:

```text
[ERROR] HCCL(144905,python):2025-04-20-00:26:54.435.048 [config.cc:413] [145735][InitGroupStage][RanktableCheck]rank[0] device ip family[2] is not same as others[10].
```

## Possible Causes

The two ranks obtain different IP families. For example, one side uses IPv4 while the other uses IPv6.

## Solution

Query whether IPv4 is configured:

```bash
hccn_tool -i {deviceId} -ip -g
```

Query whether IPv6 is configured:

```bash
hccn_tool -i {deviceId} -ip -inet6 -g
```

All ranks in the same job must use the same IP Family. By default, HCCL uses the IPv4 protocol first. If no IP address with the IPv4 protocol is configured on the device side, the IP address corresponding to the IPv6 protocol is used instead. You can use the HCCL_SOCKET_FAMILY environment variable to specify the NIC IP protocol to be used.

**Note**: The family value is printed as an enumeration value. The enumeration values and their corresponding relationships are shown in the following table.

| IP Family Enumeration Value | IP Protocol |
| --- | --- |
| 2 | IPv4 |
| 10 | IPv6 |
