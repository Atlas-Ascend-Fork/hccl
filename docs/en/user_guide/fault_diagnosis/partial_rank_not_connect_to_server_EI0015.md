# Some Ranks Fail to Connect to the Server Node (EI0015)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:53:44.751Z pushedAt=2026-10-13T11:48:05.792Z -->

## Symptoms

The CANN log contains the keyword "topo exchange server get socket timeout!" or "Failed to connect agent".

- Symptom on the server node:

    ```text
    [ERROR] HCCL(1041081,all_reduce_test):2025-11-21-01:20:01.624.966 [topoinfo_exchange_server.cc:314] [1041362][InitGroupStage][RanktableDetect]topo exchange server get socket timeout! timeout[120 s]
    [ERROR] HCCL(1041081,all_reduce_test):2025-11-21-01:20:01.625.103 [topoinfo_exchange_server.cc:501] [1041362][InitGroupStage][DisplayConnectionedRank]total connected num is [4],line num is [1]
    [ERROR] HCCL(1041081,all_reduce_test):2025-11-21-01:20:01.625.112 [topoinfo_exchange_server.cc:503] [1041362][InitGroupStage][DisplayConnectionedRank]need connect rankNum is [8]
    [ERROR] HCCL(1041081,all_reduce_test):2025-11-21-01:20:01.625.145 [topoinfo_exchange_server.cc:517] [1041362][InitGroupStage][DisplayConnectionedRank]connected rankinfo[LINE 0]: [0000000000000000],[0000000000000002],[0000000000000004],[0000000000000006];
    ```

- Symptom on the agent node:

    ```text
    [ERROR] HCCL(1041085,all_reduce_test):2025-11-21-01:20:01.630.122 [topoinfo_exchange_base.cc:145] [1041085][InitGroupStage][RanktableDetect] TopoDetect ERROR occur !!! fault_type[1], fault_info["Failed to connect agent[1,3,5,7,]"]
    [ERROR] HCCL(1041085,all_reduce_test):2025-11-21-01:20:01.630.557 [topoinfo_exchange_agent.cc:552] [1041085][InitGroupStage][RanktableDetect]rank num[8]is different with rank list size[4] in total topo rank info.
    ```

## Possible Causes

After the server node calls the `HcclGetRootinfo` interface, a background thread is started to wait for all ranks to connect until the timeout expires. Therefore, if not all ranks in the communicator successfully connect to the server thread within the timeout period, the server thread waits until timeout and reports an error. In addition, after the timeout error occurs, the server thread prints the list of currently connected ranks. Based on this information, the ranks that failed to connect can be identified, and the cause of the connection failure for those ranks can be further investigated.

## Solution

The following figure shows the troubleshooting approach for partial rank disconnection:

![Troubleshooting approach for partial rank disconnection](figures/rank_disconnect_debug.png)

1. Identify the disconnected ranks from the error message.
    - **Server node**: When the server node times out, it prints the information of connected ranks. Since the rankId sequence within the communicator is [0 ~ rankSize-1], the disconnected ranks can be derived from the connected rank information. For example, in the log case above, the connection of rank9 is missing, so further investigation into why rank9 failed to connect is required.
    - **Agent node**: For agents that have successfully connected, after the server node times out, they receive the cluster unconnected rank error message disseminated by the server node. Therefore, the unconnected rank can be directly identified based on "Failed to connect agent" in the error message, and the cause of the connection failure for that rank can be further investigated. For example, the log case above indicates that the connection of rank9 is missing, so the cause of rank9's failure to connect needs to be further investigated.

2. Confirm whether the unconnected rank has issued the communicator creation interface.

    Since HCCL records default logs in the run directory of CANN logs during communicator creation, you can filter the **entire cluster's logs** to check whether the corresponding rank has issued the communicator creation. For example, run the following command to filter whether rank9 has a Communicator creation issuance record. If there are multiple communicators, multiple log lines may appear. You can confirm whether they belong to the same communicator based on the "identifier" communicator name in the logs:

    ```bash
    grep -r "Entry-HcclCommInitRootInfoInner" | grep "rank\[9\]"
    ```

    - If the corresponding rank has issued the communicator creation interface, obtain the node and process ID information of the missing rank based on the search results, and then further investigate the related error log information based on the CANN logs of the missing rank.
    - If no communicator creation interface issuance log is found for the corresponding rank in the cluster logs, investigate the cause of the rank not issuing the corresponding Communicator creation interface from the service side.
