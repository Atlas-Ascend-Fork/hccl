# Service Flow and Troubleshooting Approach

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:52:00.229Z pushedAt=2026-10-13T08:53:36.499Z -->

## Service Flow

During cluster information negotiation (the scenario where a communicator is created based on root node information), HCCL establishes socket connections between the server node and each rank node to exchange local information, thereby obtaining the cluster information of the entire communicator and completing communicator initialization.

![Cluster Information Negotiation Service Flow](figures/cluster_info_nego_flow.png)

1. The server node calls the HcclGetRootInfo API to start a listening thread.
    1. A server node is selected from the cluster, typically the rank0 node within the communicator. This node calls the HcclGetRootInfo API.
    2. The IP address and port information of the host network interface card are obtained to generate the rootInfo value. The host network interface card can be specified through the HCCL_SOCKET_IFNAME environment variable, and the port can be specified through the HCCL_IF_BASE_PORT and HCCL_HOST_SOCKET_PORT_RANGE environment variables.
    3. The server node performs binding and listening based on the IP address and port, starts a background thread to wait for connections from all agents in the communicator, and directly returns rootInfo to complete the interface call.

2. Each rank node calls the HcclCommInitRootInfo interface to establish a connection to the server node.
    1. The upper-layer service or framework broadcasts rootInfo to each rank in the communicator. Each rank node calls the HcclCommInitRootInfo interface with rootInfo as an input parameter.
    2. A socket connection is established with the server through the host network interface card, and the rank's own rankInfo is sent to the server. After the transmission is complete, the rank enters the receiving state and waits for the server to return the complete cluster information. The process of establishing the socket connection and waiting for the cluster information to be returned must be completed within a specified timeout period, which can be controlled through the HCCL_CONNECT_TIMEOUT environment variable.

3. The server node collects the complete cluster information and sends it to each rank.
    1. After collecting the rankInfo of all ranks, the background thread on the server node generates the complete cluster information and sends it to each agent thread of the ranks. In this way, each rank obtains the full rank information of the entire communicator.
    2. Meanwhile, the server must complete waiting for socket connections from all ranks within a certain timeout. The timeout can be controlled by the HCCL_CONNECT_TIMEOUT environment variable.

## Cause Analysis

Since the communicator creation method based on root node information requires socket connections to be established between rank nodes, the host-side network interface cards are used, and all ranks within the communicator must execute synchronously within the timeout period. Therefore, network connectivity issues with the host NICs or the failure of some ranks to correctly establish socket connections are common causes of negotiation failures. The primary task in fault location is to identify the abnormal rank.

Common causes of cluster information negotiation phase failures and key logs:

- If port binding fails on the server node, run the following command to check whether a port binding failure has occurred. For details, see [Server Node Port Binding Failure (EI0019)](./server_node_port_bind_fail_EI0019.md#server-node-port-binding-failure-ei0019).

    ```bash
    grep -r "socket type\[2\].*Please check the port status and whether the port is being used by other process"
    ```
    
- If a socket connection between a rank node and the server node times out, run the following command for a quick query. For details, see [Communicator Initialization Topo Probe Timeout (EI0015)](./topo_detect_timeout_EI0015.md#communicator-initialization-topo-detection-timeout-ei0015).

    ```bash
    grep -r "topo exchange server get socket timeout"
    grep -r "topo exchange agent get socket timeout"
    ```

## Key Log Information

- During communicator creation, HCCL records key log information when the communicator creation interface is invoked. The log records are stored in the plog file under the run directory of the CANN log. Whether a process has invoked the corresponding communicator creation interface can be determined based on the corresponding logs. For detailed log information, see [HCCL-Related Log Description](./debug_thinking.md#hccl-related-log-description).
- During communicator negotiation, if any rank fails to establish a socket connection with the root node, the root node prints the information of the connected ranks before it exits due to timeout, and broadcasts the information of the missing ranks to the ranks that have already established connections. You can use this information to locate the unconnected rank and further identify the root cause. For detailed log information, see [Communicator Initialization Topo Probe Timeout (EI0015)](./topo_detect_timeout_EI0015.md#communicator-initialization-topo-detection-timeout-ei0015).
