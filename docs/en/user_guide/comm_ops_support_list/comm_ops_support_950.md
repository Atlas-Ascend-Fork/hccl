# Ascend 950PR&950DT Products

This section provides the communication operator support status for the Ascend 950PR&950DT products.

> [!NOTE]Note
> 
> - The communication operator support status is presented below by expansion mode.
> - In the tables in this section, "√" indicates supported, "×" indicates not supported, and "NA" indicates not applicable.

## AICPU_TS (AI CPU)

<table><thead align="left"><tr><th><p>Operator</p>
</th>
<th><p>Network Running Mode</p>
</th>
<th><p>Deterministic Computation</p>
</th>
<th><p>Intra-box Communication</p>
</th>
<th><p>Intra-POD Communication</p>
</th>
<th><p>Intra-SuperPoD Communication</p>
</th>
<th><p>Inter-SuperPoD Communication</p>
</th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="3"><p>AllGather</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="3"><p>AllGatherV</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>Reduce</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="3"><p>Scatter</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="3"><p>Send</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="3"><p>Recv</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="3"><p>BatchSendRecv</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
</tbody>
</table>

## AIV

<table><thead align="left"><tr><th><p>Operator</p>
</th>
<th><p>Network Running Mode</p>
</th>
<th><p>Deterministic Computation</p>
</th>
<th><p>Intra-box Communication</p>
</th>
<th><p>Intra-POD Communication</p>
</th>
<th><p>Intra-SuperPoD Communication</p>
</th>
<th><p>Inter-SuperPoD Communication</p>
</th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AllGather</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AllGatherV</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>Reduce</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>Scatter</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>Send</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>Recv</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>BatchSendRecv</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
</tbody>
</table>

## CCU MS Mode

<table><thead align="left"><tr><th><p>Operator</p>
</th>
<th><p>Network Running Mode</p>
</th>
<th><p>Deterministic Computation</p>
</th>
<th><p>Intra-Box Communication (Ascend 950PR)</p>
</th>
<th><p>Intra-Box Communication (Ascend 950DT)</p>
</th>
<th><p>Intra-POD Communication</p>
</th>
<th><p>Intra-SuperPoD Communication</p>
</th>
<th><p>Inter-SuperPoD Communication</p>
</th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AllGather</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AllGatherV</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>Reduce</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>Scatter</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>Send</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>Recv</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>BatchSendRecv</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
</tbody>
</table>

## CCU SCHED Mode

<table><thead align="left"><tr><th><p>Operator</p>
</th>
<th><p>Network Running Mode</p>
</th>
<th><p>Deterministic Computation</p>
</th>
<th><p>Intra-box Communication</p>
</th>
<th><p>Intra-POD Communication</p>
</th>
<th><p>Intra-SuperPoD Communication</p>
</th>
<th><p>Inter-SuperPoD Communication</p>
</th>
</tr>
</thead>
<tbody><tr><td rowspan="3"><p>Broadcast</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AllGather</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AllGatherV</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>Reduce</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AllReduce</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>Scatter</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>ReduceScatter</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>ReduceScatterV</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AlltoAll</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AlltoAllV</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>AlltoAllVC</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>Send</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>Recv</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td rowspan="3"><p>BatchSendRecv</p>
</td>
<td><p>Single-Operator Mode</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Mode (Ascend IR)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
<tr><td><p>Graph Capture Mode (ACL graph)</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
<td><p>×</p>
</td>
</tr>
</tbody>
</table>
