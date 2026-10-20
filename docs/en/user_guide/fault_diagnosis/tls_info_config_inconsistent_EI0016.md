# TLS Information Configuration Inconsistency (EI0016)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:55:42.361Z pushedAt=2026-10-14T03:07:40.387Z -->

## Symptoms

The CANN log contains the keyword "All ranks are consistent.", as shown below:

```text
[ERROR] HCCL(94774,all_reduce_test):2025-10-27-11:51:32.570.490 [topoinfo_exchange_agent.cc:831] [94774][InitGroupStage][RanktableCheck] Value Disable for config "tls" is invalid. Expected Value:"All ranks are consistent. Current status : rankList for enabled tls:[10.78.106.107/0]; rankList for disabled tls:[10.78.106.107/0]; rankList for query failure tls:".;
```

## Possible Causes

During communicator creation, after the server node receives information about all ranks in the communicator, it verifies whether the TLS configurations of all ranks in the communicator are consistent. If any configuration inconsistency is detected, the verification fails immediately and the process exits. A list of nodes with Disable or Enable is printed, while the nodes not listed have the opposite TLS configuration.

This verification function is supported only in Ascend HDK 25.2.0 or later and in scenarios where the communicator is initialized through root information negotiation.
<!-- npu="950" id1 -->
The Ascend 950PR&950DT products do not support this feature.
<!-- end id1 -->

## Solution

1. Query the TLS status switch on each server involved in collective communication.

    Run the following command on the server to obtain the TLS switch status.

    ```bash
    hccn_tool -i <device_id> -tls -g
    ```

    <device_id\> is the logical ID of the device. You can also use the following for loop to query the TLS information of all devices at once.

    ```bash
    for i in `seq 0 7`; do hccn_tool -i $i -tls -g; done    # 0 and 7 are the start and end values of the Device IDs to be queried.
    ```

    The printed information is as follows:

    ```text
    dev_id:0, tls switch[0](0:disable, 1:enable), tls alarm time threshold[60]days
    dev_id:0, [pub cert] info:
             issuer[/C=CN/ST=GD/O=HUAWEI/OU=2012/CN=2_1thCA]
             start_time[Wed Feb 19 03:19:21 2020 GMT]
             end_time[Sat Feb 16 03:19:21 2030 GMT]
    dev_id:0, [ca1 cert] info:
             issuer[/C=CN/ST=GD/L=SZ/O=HUAWEI/CN=1thCA]
             start_time[Wed Feb 19 03:19:07 2020 GMT]
             end_time[Sat Feb 16 03:19:07 2030 GMT]
    dev_id:0, [ca2 cert] info:
             issuer[/C=CN/ST=GD/L=SZ/O=HUAWEI/CN=1thCA]
             start_time[Wed Feb 19 03:19:10 2020 GMT]
             end_time[Sat Feb 16 03:19:10 2030 GMT]
    dev_id:1, tls switch[0](0:disable, 1:enable), tls alarm time threshold[60]days
    dev_id:1, [pub cert] info:
             issuer[/C=CN/ST=GD/O=HUAWEI/OU=2012/CN=2_1thCA]
             start_time[Wed Feb 19 03:19:21 2020 GMT]
             end_time[Sat Feb 16 03:19:21 2030 GMT]
    dev_id:1, [ca1 cert] info:
             issuer[/C=CN/ST=GD/L=SZ/O=HUAWEI/CN=1thCA]
             start_time[Wed Feb 19 03:19:07 2020 GMT]
             end_time[Sat Feb 16 03:19:07 2030 GMT]
    dev_id:1, [ca2 cert] info:
             issuer[/C=CN/ST=GD/L=SZ/O=HUAWEI/CN=1thCA]
             start_time[Wed Feb 19 03:19:10 2020 GMT]
             end_time[Sat Feb 16 03:19:10 2030 GMT]
    ... ...
    ```

    In the output, `tls switch[0]` indicates that the TLS status is disabled, and `switch[1]` indicates that the TLS status is enabled.

2. Determine whether the TLS status switches of all devices on each server are consistent.
    - If they are inconsistent, it is recommended to uniformly set the TLS status to enabled. If the TLS switch is disabled, information may be subject to eavesdropping, tampering, and impersonation during collective communication.

        Use the following command to modify the TLS status switch:

        ```bash
        hccn_tool -i <device_id> -tls -s enable 1
        ```

        `enable` is the status switch. Setting it to `1` indicates enabled, and setting it to `0` indicates disabled.

    - If they are consistent and the status is enabled, proceed to step 3 to determine whether the TLS certificate information of each node is consistent.

3. Check whether the TLS certificate information of each device on all servers is consistent.

    You can determine whether the TLS certificate information of each device is consistent based on the information obtained in step 1. If it is inconsistent, you can replace the certificate suite by running the following command.

    ```bash
    hccn_tool -i 0 -tls -s path /root pri pri.pem pub pub.pem ca1 ca1.pem ca2 ca2.pem crl xxx.crl
    ```

    `-i` specifies the device ID. `-s path` specifies the storage path for the certificate, private key, and certificate revocation list. `pri` specifies the private key file name. `pub` specifies the device certificate file name. `ca1`, `ca2`, and `crl` specify the root certificate, secondary root certificate, and certificate revocation list file names, respectively.

    For more usage and parameter descriptions of the hccn_tool, see the corresponding version of *[HCCN Tool Interface Reference](https://support.huawei.com/enterprise/en/ascend-computing/ascend-hdk-pid-252764743?category=developer-documents&subcategory=interface-reference)*.
