# Typical Multi-Node Communicator Initialization Failure (EI0015)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-09T03:29:05.851Z pushedAt=2026-11-12T08:36:28.184Z nl=2 -->

## Symptoms

The following figure shows a communicator creation failure case in a three-node scenario:

![Communicator Creation Failure Case in a Three-Node Scenario](figures/3node_comm_create_fail.png)

This symptom is a typical error log of a communicator creation negotiation timeout in a three-node, 24-card scenario, where node 0 is the root node of the communicator. The error symptoms of each node are analyzed as follows:

- **Node 0**: Node 0 is the root node. The error message indicates that the server thread timed out while waiting for all ranks in the communicator to connect. The successfully connected ranks can be obtained from the error message, and the unconnected ranks (rank16 to rank23) can be deduced by reverse inference.
- **Node 1**: This node successfully created a socket connection with the root node. After waiting for the root node timeout, it received the unconnected rank information broadcast by the root node. The unconnected ranks, rank16 to rank23, can be directly obtained from the error log.
- **Node 2**: The error log of Node 2 indicates a socket timeout when establishing a connection with the server node. The root cause is a Host-side network configuration error between Node 2 and the root node, which prevented the connection. The problem was resolved after the configuration was corrected.

## Troubleshooting Approach

As can be seen from this typical scenario, when a link setup timeout occurs during communicator creation in the cluster, the unconnected rank -- that is, the root node that reported the error -- can be quickly identified from the error log, regardless of whether the log originates from the server node or a node that has been successfully connected. At this point, it is sufficient to focus on troubleshooting the failure cause of the unconnected rank. For example, a common cause of connection timeout is that the [HCCL_SOCKET_IFNAME](../hccl_env/HCCL_SOCKET_IFNAME.md) environment variable is not configured, resulting in the use of an unreachable Host network interface card.
