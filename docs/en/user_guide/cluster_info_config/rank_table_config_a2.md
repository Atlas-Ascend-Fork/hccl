# Rank Table Configuration Resource Information (Atlas A2 Products)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:46:53.503Z pushedAt=2026-10-13T03:40:24.025Z -->

For Atlas A2 products, taking two AI servers with two devices in each AI server as an example, the rank table file configuration example is as follows:

> [!NOTE]Note
> The rank table file is in JSON format. The comments in the JSON file examples shown in this section are provided only for ease of understanding. In actual use, delete the comments from the JSON file.

```json
{
    "status":"completed",  // Rank table availability flag. "completed" indicates that the rank table is available.
    "version":"1.0",       // Rank table template version. Set this parameter to 1.0.
    "server_count":"2",    // Number of AI servers participating in training. In this example, there are two AI servers.
    "server_list":
    [
        {
            "server_id":"node_0",  // AI server ID, of the String type. Ensure that it is globally unique.
            "device":[             // List of devices in the AI server.
                {
                    "device_id":"0",            // Physical ID of the processor.
                    "device_ip":"192.168.1.8",  // Actual NIC IP address of the processor.
                    "device_port":"16667",      // NIC listening port of the processor.
                    "rank_id":"0"               // Rank identifier. Configure it starting from 0 and ensure that it is globally unique.
                },
                {
                    "device_id":"1",
                    "device_ip":"192.168.1.9", 
                    "device_port":"16667",
                    "rank_id":"1"
                }
            ]
        },
        {
            "server_id":"node_1",
            "device":[
                {
                    "device_id":"0",
                    "device_ip":"192.168.2.8",
                    "device_port":"16667",
                    "rank_id":"2"
                },
                {
                    "device_id":"1",
                    "device_ip":"192.168.2.9", 
                    "device_port":"16667",
                    "rank_id":"3"
                }
            ]
        }
    ]
}
```

The rank table configuration file is described as follows:

| Level-1 Configuration Item | Level-2 Configuration Item | Level-3 Configuration Item | Configuration Instructions |
| --- | --- | --- | --- |
| status |  |  | Mandatory.<br>rank table availability flag.<br>  - completed: indicates that the rank table is available.<br>  - initializing: indicates that the rank table is unavailable. |
| version |  |  | Mandatory.<br>Rank table template version.<br>Set this parameter to 1.0. |
| server_count |  |  | Mandatory.<br>Number of AI servers participating in collective communication. |
| server_list |  |  | Mandatory.<br>List of AI servers participating in collective communication. |
|  | server_id |  | Mandatory.<br>AI server identifier, string type, with a length less than or equal to 64. Ensure that it is globally unique.<br>Configuration example: node_0. |
|  | device |  | Mandatory.<br>List of devices in the AI server. |
|  |  | device_id | Mandatory.<br>Physical ID of the AI processor, that is, the serial number of the device on the AI server.<br>You can obtain the physical ID of the AI processor by running the "ls /dev/davinci*" command.<br>For example, if /dev/davinci0 is displayed, the physical ID of the AI processor is 0.<br>Value range: \[0, actual number of devices - 1].<br>Note: The "device_id" configuration item has a higher priority than the environment variable "ASCEND_DEVICE_ID". |
|  |  | device_ip | Optional.<br>IP address of the integrated NIC of the AI processor, globally unique, required to be in standard IPv4 or IPv6 format.<br>Note that:<br>  - In the multi-machine scenario, device_ip must be configured.<br>  - In the single-machine scenario, device_ip can be left unconfigured.<br>You can run the cat /etc/hccn.conf command on the current AI server to obtain the NIC IP, for example:<br>address_0=xx.xx.xx.xx<br>netmask_0=xx.xx.xx.xx<br>netdetect_0=xx.xx.xx.xx<br>The queried address_xx is the NIC IP. The number after address is the physical ID of the AI processor, that is, device_id, and the IP address after it is the NIC IP corresponding to the device that the user needs to fill in. |
|  |  | device_port | Optional.<br>Communication port of the device NIC. The value range is \[1,65535]. Ensure that the specified port is not occupied by other processes. Note that \[1,1023] are system-reserved ports and should be avoided.<br>In a single-card multi-process scenario (that is, multiple service processes share one NPU), you are advised to configure this field and set different port numbers for different service processes. Otherwise, the service may fail to run due to port conflicts. |
|  |  | rank_id | Mandatory.<br>Unique rank identifier. Configure it as an integer starting from 0, and ensure that it is globally unique. Value range: \[0, total number of devices - 1].<br>- It is recommended that rank_id be sorted according to the physical connection order of devices, that is, devices that are physically closer should be arranged together. Otherwise, performance may be affected.<br>&nbsp;&nbsp;  For example, if device_ip is set in ascending order of physical connections, it is recommended that rank_id also be set in ascending order.<br>- rank_id values in different AI servers do not support cross configuration.<br> &nbsp;&nbsp; Positive example: the rank_id set in server 1 is {0,1,2,3}, and the rank_id set in server 2 is {4,5,6,7}.<br> &nbsp;&nbsp; Negative example: the rank_id set in server 1 is {0,1,2,7}, and the rank_id set in server 2 is {3,4,5,6}. |
