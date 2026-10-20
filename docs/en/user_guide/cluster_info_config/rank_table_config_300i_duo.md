# Rank Table Configuration Resource Information (Atlas 300I Duo Inference Card)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:46:39.889Z pushedAt=2026-10-13T02:19:42.011Z -->

For the Atlas 300I Duo inference card, taking two AI servers with two devices in each AI server as an example, the rank table file configuration example is as follows:

> [!NOTE]Note
> The rank table file is in JSON format. The comments in the JSON file examples shown in this section are provided only for ease of understanding. In actual use, delete the comments from the JSON file.

```json
{
    "status":"completed",   // Rank table availability flag; completed indicates available
    "version":"1.0",        // Rank table template version; set to 1.0
    "server_count":"2",     // Number of AI servers participating in training; in this example, there are two AI servers
    "server_list":
    [
        {
            "server_id":"node_0",  // AI server identifier, string type; ensure it is globally unique
            "device":[             // List of devices in the AI server
                {
                    "device_id":"0",   // Physical ID of the processor
                    "device_ip":"192.168.1.8",   // Actual NIC IP address of the processor
                    "rank_id":"0"                // Rank identifier, configured starting from 0; ensure it is globally unique
                },
                {
                    "device_id":"1",
                    "device_ip":"192.168.1.9", 
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
                    "rank_id":"2"
                },
                {
                    "device_id":"1",
                    "device_ip":"192.168.2.9", 
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
|  |  | device_ip | Mandatory.<br>IP address of the integrated NIC of the AI processor. It must be globally unique and in the standard IPv4 or IPv6 format.<br>You can run the "cat /etc/hccn.conf" command on the current AI server to obtain the NIC IP address. For example:<br>address_0=xx.xx.xx.xx<br>netmask_0=xx.xx.xx.xx<br>netdetect_0=xx.xx.xx.xx<br>The queried address_xx is the NIC IP address. The number after "address" is the physical ID of the AI processor, that is, device_id, and the IP address after it is the NIC IP address of the device that you need to fill in. |
|  |  | rank_id | Mandatory.<br>Unique rank identifier. Set it to an integer starting from 0, and ensure that it is globally unique. Value range: \[0, total number of devices - 1].<br>- It is recommended that rank_id be sorted by the physical connection order of devices, that is, devices that are physically closer are arranged together. Otherwise, performance may be affected.<br>&nbsp;&nbsp;  For example, if device_ip is set in ascending order of physical connections, it is recommended that rank_id also be set in ascending order.<br>- rank_id values in different AI servers do not support cross configuration.<br> &nbsp;&nbsp; Positive example: the rank_id set in server 1 is {0,1,2,3}, and the rank_id set in server 2 is {4,5,6,7}.<br> &nbsp;&nbsp; Negative example: the rank_id set in server 1 is {0,1,2,7}, and the rank_id set in server 2 is {3,4,5,6}. |
