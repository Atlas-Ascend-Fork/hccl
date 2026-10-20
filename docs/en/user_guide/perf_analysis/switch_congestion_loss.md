# Switch Traffic Congestion, Backpressure, or Packet Loss and Retransmission

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T04:09:32.548Z pushedAt=2026-10-15T08:46:37.290Z -->

> [!NOTE]
> The use cases and handling methods described in this section apply only to the following products:
>
> - Atlas A3 products
> - Atlas A2 products

If a notify wait task lasting about 4 seconds is observed in the profile data, it typically indicates a network configuration issue that has caused packet loss and retransmission. You can locate the issue by checking the `roce_new_pkt_rty_num` field in the statistics using the hccn_tool.

If the value of this field increases during task execution, it indicates that packet loss and retransmission have occurred on the network. In this case, further troubleshoot the switch configuration.

Run the following command to view the statistics:

```bash
hccn_tool -i {DeviceId} -stat -g
```
