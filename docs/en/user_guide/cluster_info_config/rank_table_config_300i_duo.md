# Rank Table Configuration Resource Information (Atlas Inference Products)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-09T03:20:31.758Z pushedAt=2026-11-09T07:44:32.346Z nl=2 -->

For Atlas inference products, only the single-server scenario is supported. The following uses four NPUs in one AI server as an example. A sample rank table file configuration is as follows:

> [!NOTE]
> The rank table file is in JSON format. This example uses one server and assigns consecutive Device IDs and rank IDs within the server. The IP addresses are for illustration only. Replace them with the actual NIC configuration. For networking restrictions, see [System Constraints and Restrictions](../constraints.md#atlas-inference-products).

```json
{
    "status": "completed",
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "node_0",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "192.168.1.8",
                    "rank_id": "0"
                },
                {
                    "device_id": "1",
                    "device_ip": "192.168.1.9",
                    "rank_id": "1"
                },
                {
                    "device_id": "2",
                    "device_ip": "192.168.1.10",
                    "rank_id": "2"
                },
                {
                    "device_id": "3",
                    "device_ip": "192.168.1.11",
                    "rank_id": "3"
                }
            ]
        }
    ]
}
```

The rank table configuration file is described as follows:

| Level-1 Configuration Item | Level-2 Configuration Item | Level-3 Configuration Item | Configuration Instructions |
| --- | --- | --- | --- |
| status |  |  | Mandatory.<br>Availability flag of the rank table.<br>  - completed: indicates that the rank table is available.<br>  - initializing: indicates that the rank table is unavailable. |
| version |  |  | Mandatory.<br>Rank table template version.<br>Set this parameter to 1.0. |
| server_count |  |  | Mandatory.<br>Number of AI servers participating in collective communication. |
| server_list |  |  | Mandatory.<br>List of AI servers participating in collective communication. |
|  | server_id |  | Mandatory.<br>Identifier of the AI server, string type, with a length less than or equal to 64. Ensure that it is globally unique.<br>Configuration example: node_0. |
|  | device |  | Mandatory.<br>List of devices in the AI server. |
|  |  | device_id | Mandatory.<br>Physical ID of the AI processor, that is, the serial number of the device on the AI server.<br>You can obtain the physical ID of the AI processor by running the "ls /dev/davinci*" command.<br>For example, if /dev/davinci0 is displayed, the physical ID of the AI processor is 0.<br>Value range: \[0, actual number of devices - 1].<br>Note: The "device_id" configuration item takes precedence over the environment variable "ASCEND_DEVICE_ID". |
|  |  | device_ip | Mandatory.<br>IP address of the integrated NIC of the AI processor. It must be globally unique and in the standard IPv4 or IPv6 format.<br>You can run the "cat /etc/hccn.conf" command on the current AI server to obtain the NIC IP address. For example:<br>address_0=xx.xx.xx.xx<br>netmask_0=xx.xx.xx.xx<br>netdetect_0=xx.xx.xx.xx<br>The queried address_xx is the NIC IP address. The number after "address" is the physical ID of the AI processor, that is, device_id, and the IP address after it is the NIC IP address of the device that you need to fill in. |
|  |  | rank_id | Required.<br>Unique rank identifier. Set it to an integer starting from 0. It must be globally unique. Value range: \[0, total number of devices - 1].<br>- It is recommended that rank_id be sorted by the physical connection order of devices, that is, devices that are physically closer are arranged together. Otherwise, performance may be affected.<br>&nbsp;&nbsp;  For example, if device_ip is set in ascending order of physical connections, it is recommended that rank_id also be set in ascending order.<br>- Cross configuration of rank_id across different AI servers is not supported.<br> &nbsp;&nbsp; Positive example: The rank_id set in server 1 is {0,1,2,3}, and the rank_id set in server 2 is {4,5,6,7}.<br> &nbsp;&nbsp; Negative example: The rank_id set in server 1 is {0,1,2,7}, and the rank_id set in server 2 is {3,4,5,6}. |
