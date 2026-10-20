# Typical Multi-Node Scenario Communicator Initialization Failure (EI0015)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:55:37.271Z pushedAt=2026-10-14T06:59:44.925Z -->

## Symptoms

The following figure shows a communicator creation failure case in a three-node scenario:

![Communicator Creation Failure Case in a Three-Node Scenario](figures/3node_comm_create_fail.png)

This symptom is a typical error log of a communicator creation negotiation timeout in a three-node, 24-card scenario, where node 0 is the root node of the communicator. The error symptoms of each node are analyzed as follows:

- **Node 0**: Node 0 is the root node. The error message indicates that the server thread timed out while waiting for all ranks in the communicator to connect. The successfully connected ranks can be obtained from the error message, and the unconnected ranks (rank16 to rank23) can be deduced by reverse inference.
- **Node 1**: This node successfully created a socket connection with the root node.  After the wait for the root node timed out, it received the unconnected rank information broadcast by the root node. The unconnected ranks, rank16 to rank23, can be directly obtained from the error log.
- **Node 2**: The error log of Node 2 indicates a socket timeout when establishing a connection with the server node. The root cause is a Host-side network configuration error between Node 2 and the root node, which prevented the connection. The problem was resolved after the configuration was corrected.

## Troubleshooting Approach

As can be seen from this typical scenario, when the cluster encounters a timeout during communicator creation and connection setup, the unconnected ranks can be quickly identified from the error log, whether on the server node or on nodes that have already connected successfully. In this case, only the failure causes of the unconnected ranks need to be focused on. For example, a common cause of connection timeout is that the [HCCL_SOCKET_IFNAME](../hccl_env/HCCL_SOCKET_IFNAME.md) environment variable is not configured, which causes an unreachable host NIC to be used.
