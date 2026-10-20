# Link Setup Timeout Background

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:53:17.658Z pushedAt=2026-10-13T11:38:27.829Z -->

When a communication operator is invoked, HCCL creates a Socket connection over the **parameter plane network** based on the TCP protocol to exchange initialization information such as addresses and communication topology.

If any of the following situations occurs, the other ranks waiting to establish a connection report a Socket link setup timeout error.

- Some ranks have not reached the corresponding communication operator and therefore cannot initiate a link setup request.
- A network connectivity fault prevents the link setup request from reaching the peer.
- The communication behavior of the two ends is inconsistent (for example, inconsistent TLS configurations or unsynchronized execution of communication operators), so the peer cannot correctly respond to the link setup request.

Because communication operators in HCCL are processed serially in the order of service execution, a link setup blockage of one communication operator prevents subsequent communication operators from continuing. Therefore, **a link setup timeout usually propagates in a cascading manner across multiple ranks**.

For example:

- rank4 executes communication operator 3 and times out while establishing a link with rank3.
- rank3 is actually blocked by communication operator 2 and is waiting to complete link setup with rank2.
- rank2 is in turn blocked by communication operator 1 and is waiting to establish a link with rank1.
- The root cause is that the link setup between rank1 and rank2 failed, causing the ranks along the entire dependency chain to time out one by one.

![link_create_multi_rank_debug](figures/link_create_multi_rank_debug.png)

Therefore, when troubleshooting a link setup timeout, do not focus only on the rank that currently reports the error. Instead, trace forward along the link setup relationships to find the **rank pair where the link setup failure first occurred**, and analyze its failure cause first.

# Link Setup Timeout (EI0006)

When a link setup timeout occurs on the parameter plane, the following keyword usually appears in the CANN logs: `wait socket establish timeout`.

```text
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.403 [hccl_socket_manager.cc:797] [18744][Wait][LinkEstablish]wait socket establish timeout, role[1] rank[1] timeout[120 s]
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.454 [hccl_socket_manager.cc:861] [18744][Wait][LinksEstablishCompleted] is failed. ret[9].
```

## Troubleshooting Process

### Finding the Link Setup Peer

When a link setup timeout occurs, information about the link setup peer is printed:

```text
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.646 [hccl_socket_manager.cc:623] [18744]   _________________________LINK_ERROR_INFO___________________________
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.650 [hccl_socket_manager.cc:624] [18744]   |  comm error, device[1]
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.653 [hccl_socket_manager.cc:626] [18744]   |  dest_ip(user_rank)  |   dest_port   |  src_ip(user_rank)   |   src_port   |   MyRole   |   Status   |    TlsStatus   |
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.655 [hccl_socket_manager.cc:628] [18744]   |----------------------|---------------|----------------------|--------------|------------|------------|----------------|
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.706 [hccl_socket_manager.cc:583] [18744]   |  192.0.2.199(0)   |  16666  |   192.0.3.198(1)   |  3234403008  |  client  | time out |   DISABLE  | LinkInfo
```

 Obtain the device IP addresses of both ends of the link from `LINK_ERROR_INFO`. `dest_ip` is the peer IP address, and `src_ip` is the local IP address. In the example, the peer IP address of the link setup is `192.0.2.199`. Use the peer IP address to find the peer log path, and run the `grep -rni "localIp\[192.xx.xx.xxx\]"` command to globally search for the peer run log path (where `192.xx.xx.xxx` is the IP address of the link setup peer). Then, locate the debug log based on the run log location, and **troubleshoot the errors in the peer debug log**.

```text
The run log and debug log correspond as follows:
├── debug
│   ├── device-1
│   │   ├── device-2849436_20260722090447711.log
│   └── plog
│       ├── plog-2849436_20260722090443026.log
├── run
│   ├── device-1
│   │   ├── device-2849436_20260722090449878.log
│   └── plog
│       ├── plog-2849436_20260722090443227.log
```

### Confirming Peer Behavior to Troubleshoot Inter-Card Behavior Inconsistency

Based on the peer debug log, locate the issue step by step according to the following process:

![link_create_debug_method](figures/link_create_debug_method.png)

#### Troubleshooting Point 1: No Abnormal Logs on the Peer

If the peer has no debug logs, or the debug logs contain no ERROR information, the peer has not delivered the communication operator and therefore has not initiated a link setup request. This scenario is not an HCCL issue. **You need to check from the service side whether the communication operator delivery behavior is consistent on both ends.**

#### Troubleshooting Point 2: The Peer Reports Other Types of Errors

If the peer's debug log first shows other errors (rather than a parameter plane link setup timeout), the link setup timeout is usually only a subsequent symptom. This scenario is not an HCCL issue. **Prioritize analyzing the root cause of the peer's first error, and then continue locating the link setup issue.**

#### Troubleshooting Point 3: The Peer Also Experiences a Link Setup Timeout, but With a Different Object

As shown in the example, the local end `192.0.3.198` times out when establishing a link with the peer `192.0.2.199`, while the peer times out when establishing a link with `192.0.2.196`. In this case, you need to continue recursively tracing the behavior of the peer that establishes a link with `192.0.2.196`. **This error is not the first occurrence, and you need to continue troubleshooting by following the troubleshooting process.**
This is the most typical **cascade propagation** phenomenon of link setup timeout.

```text
# Link setup timeout log on the local end
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.646 [hccl_socket_manager.cc:623] [18744]   _________________________LINK_ERROR_INFO___________________________
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.650 [hccl_socket_manager.cc:624] [18744]   |  comm error, device[1]
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.653 [hccl_socket_manager.cc:626] [18744]   |  dest_ip(user_rank)  |   dest_port   |  src_ip(user_rank)   |   src_port   |   MyRole   |   Status   |    TlsStatus   |
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.655 [hccl_socket_manager.cc:628] [18744]   |----------------------|---------------|----------------------|--------------|------------|------------|----------------|
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.706 [hccl_socket_manager.cc:583] [18744]   |  192.0.2.199(0)   |  16666  |   192.0.3.198(1)   |  3234403008  |  client  | time out |   DISABLE  | LinkInfo

# Link setup timeout log on the peer
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.646 [hccl_socket_manager.cc:623] [18744]   _________________________LINK_ERROR_INFO___________________________
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.650 [hccl_socket_manager.cc:624] [18744]   |  comm error, device[1]
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.653 [hccl_socket_manager.cc:626] [18744]   |  dest_ip(user_rank)  |   dest_port   |  src_ip(user_rank)   |   src_port   |   MyRole   |   Status   |    TlsStatus   |
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.655 [hccl_socket_manager.cc:628] [18744]   |----------------------|---------------|----------------------|--------------|------------|------------|----------------|
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.706 [hccl_socket_manager.cc:583] [18744]   |  192.0.2.196(3)   |  16666  |   192.0.2.199(0)   |  3234403008  |  client  | time out |   DISABLE  | LinkInfo
```

#### Troubleshooting Point 4: Both Sides Time Out Waiting for Each Other

If both the local end and the peer end are waiting for each other to establish a connection, `192.0.2.199` and `192.0.3.198` time out when establishing a link to each other. **First, check whether the error reporting time on both ends exceeds the link setup wait time.**
If the time gap the error reporting times on the two ends already exceeds the configured link setup wait time (120 seconds by default), it usually indicates that service execution is out of sync and one end has not entered the communication operator for a long time. **The service side must troubleshoot the root cause of the communication operator delivery timeout on both ends.**
The link setup wait time can be configured through the environment variable `HCCL_CONNECT_TIMEOUT`, which defaults to 120 seconds. You can run `grep -r "HCCL_CONNECT_TIMEOUT" run/plog/` to confirm the current configuration.

```text
# Link setup timeout log on the local end
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.646 [hccl_socket_manager.cc:623] [18744]   _________________________LINK_ERROR_INFO___________________________
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.650 [hccl_socket_manager.cc:624] [18744]   |  comm error, device[1]
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.653 [hccl_socket_manager.cc:626] [18744]   |  dest_ip(user_rank)  |   dest_port   |  src_ip(user_rank)   |   src_port   |   MyRole   |   Status   |    TlsStatus   |
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.655 [hccl_socket_manager.cc:628] [18744]   |----------------------|---------------|----------------------|--------------|------------|------------|----------------|
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.706 [hccl_socket_manager.cc:583] [18744]   |  192.0.2.199(0)   |  16666  |   192.0.3.198(1)   |  3234403008  |  client  | time out |   DISABLE  | LinkInfo

# Link setup timeout log on the peer end
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.646 [hccl_socket_manager.cc:623] [18744]   _________________________LINK_ERROR_INFO___________________________
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.650 [hccl_socket_manager.cc:624] [18744]   |  comm error, device[1]
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.653 [hccl_socket_manager.cc:626] [18744]   |  dest_ip(user_rank)  |   dest_port   |  src_ip(user_rank)   |   src_port   |   MyRole   |   Status   |    TlsStatus   |
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.655 [hccl_socket_manager.cc:628] [18744]   |----------------------|---------------|----------------------|--------------|------------|------------|----------------|
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.706 [hccl_socket_manager.cc:583] [18744]   |  192.0.3.198(1)   |  16666  |   192.0.2.199(0)   |  3234403008  |  server | time out |   DISABLE  | LinkInfo
```

#### Troubleshooting Point 5: Both Sides Wait for Each Other Within the Timeout Period

If both sides enter link setup almost simultaneously but fail to establish a Socket connection, further troubleshooting of the network and configuration is required.

1. Check whether the TLS configurations are consistent.

   If the TLS configurations are inconsistent, the Socket handshake verification fails, and both sides report a link setup timeout. You can confirm the TLS switch settings on both ends by using the following method:

   - Run `grep -r "TLS SWITCH" log/run/device-*` in the node's log to obtain the TLS status:

     ```text
     run/device-0/device-2849330_20251024153927364.log:[INFO] HCCP(2988,hccp_service.bin):2025-10-24-15:39:26.133.826 [rs_ssl.c:1529]tid:2988,rs_ssl_init(1529) : TLS SWITCH (1)
     run/device-1/device-2849331_20251024153928174.log:[INFO] HCCP(30877,hccp_service.bin):2025-10-24-15:39:25.142.466 [rs_ssl.c:1529]tid:30877,rs_ssl_init(1529) : TLS SWITCH (0)
     ```

   - Check the `TLS SWITCH` field (0: disabled, 1: enabled) to ensure that the TLS configurations of both communication sides are consistent. For details, see [TLS Configuration](https://www.hiascend.com/document/detail/en/canncommercial/latest/API/hcclug/hcclug_000045.html).

2. Check the network connectivity of the devices.

   - First determine whether the src_ip in LINK_ERROR_INFO is a NIC or VNIC IP. Run the following command on the node that reports the error to query the IP address of each NIC:

     ```text
     for n in {0..15}; do hccn_tool -i $n -ip -g ;done   # Parameter -ip queries the NIC IP address.
     for n in {0..15}; do hccn_tool -i $n -vnic -g ;done   # Parameter -vnic queries the VNIC IP address.
     ```

     Match the reported IP address and its index in the queried IP list. For example, in the following queried IP list, `192.168.2.199` is a VNIC IP with index 0 (starting from 0).

     ```text
     $for n in {0..15}; do hccn_tool -i $n -ip -g ;done
     vnic link status: UP
     vnic ipaddr: 192.168.2.199
     vnic netmask: 255.255.255.0
     vnic link status: UP
     vnic ipaddr: 192.168.2.198
     vnic netmask: 255.255.255.0
     vnic link status: UP
     vnic ipaddr: 192.168.2.197
     vnic netmask: 255.255.255.0
     vnic link status: UP
     vnic ipaddr: 192.168.2.196
     vnic netmask: 255.255.255.0
     vnic link status: UP
     vnic ipaddr: 192.168.2.195
     vnic netmask: 255.255.255.0
     vnic link status: UP
     vnic ipaddr: 192.168.2.194
     vnic netmask: 255.255.255.0
     vnic link status: UP
     vnic ipaddr: 192.168.2.193
     vnic netmask: 255.255.255.0
     vnic link status: UP
     vnic ipaddr: 192.168.2.192
     vnic netmask: 255.255.255.0
     ```

   - Use the hccn_tool command to ping the dest_ip in LINK_ERROR_INFO on the node. If the IP found in step 1 is a vnic IP, use hccs ping. If it is a nic IP, use roce ping, where `node` is the index from step 1.

     ```text
     hccn_tool -i {node} -ping -g address {dest_ip} #ROCE ping
     or
     hccn_tool -i {node} -hccs_ping -g address {dest_ip}  #HCCS ping
     ```

   If the two ranks cannot ping each other or a network port is down, contact the lab administrator to check the configuration of the corresponding NIC and switch.

3. Check the logical SuperPoD configuration.

   For the Atlas A3 SuperPoD scenario, check whether the logical SuperPoD is incorrectly configured. If different physical SuperPoDs are configured as the same logical SuperPoD, HCCL may incorrectly select the VNIC link for communication, eventually causing both ends to wait for each other and time out.

   The link type and physical SuperPoD information of both ends can be confirmed through the following logs: the link type is vnic, and the physical SuperPoD IDs of the two ends are different (0 and 1, respectively). However, because the same logical SuperPoD ID (logic_1) is configured, the vnic link is selected for communication, causing a timeout. This can be fixed by modifying or canceling the `HCCL_LOGIC_SUPERPOD_ID` configuration.

    Local end log:

    ```text
    debug/plog/plog-3003627_20260205184335411.log:14:[ERROR] HCCL(3003627,scatter_test):2026-02-05-18:44:26.379.547 [transport_manager.cc:885] [3003959][TransportManager][PrintErrorInfo]local rank information: nicType[VNIC_TYPE], logicSuperPodId[logic_1], phySuperPodId[0]. Note: Do not configure ranks belonging to different physical superpod ID info a single logical superpod ID
    ```

    Remote end log:

    ```text
    debug/plog/plog-3003628_20260205184354321.log:14:[ERROR] HCCL(3003628,scatter_test):2026-02-05-18:44:26.379.542 [transport_manager.cc:885] [3003959][TransportManager][PrintErrorInfo]local rank information: nicType[VNIC_TYPE], logicSuperPodId[logic_1], phySuperPodId[1]. Note: Do not configure ranks belonging to different physical superpod ID info a single logical superpod ID
    ```
