# Fault Diagnosis

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:54:37.259Z pushedAt=2026-10-14T02:54:34.999Z -->

- [Troubleshooting Approach](debug_thinking.md)
- [Communicator Initialization Stage](comm_domain_init_stage.md)
  - [Overall Process](comm_domain_init_flow.md)
  - [Environment Variable Configuration Error (EI0001)](_dump_env_config_error_EI0001.md)
    - [Troubleshooting Approach](env_config_error_EI0001_troubleshooting.md)
    - [HCCL_RDMA_SL Configuration Error (EI0001)](hccl_rdma_sl_config_error_EI0001.md)
    - [HCCL_SOCKET_IFNAME Configuration Error (EI0001)](hccl_socket_ifname_config_error_EI0001.md)
  - [Rank Table File Loading Failure](_dump_rank_table_load_fail.md)
    - [Troubleshooting Approach](rank_table_load_fail_troubleshooting.md)
    - [Rank Table File Read Failure (EI0004)](rank_table_file_read_fail_EI0004.md)
    <!-- npu="A3" id3 -->
    - [Rank Table Field Configuration Error (EI0004)](rank_table_field_config_error_EI0004.md)
    <!-- end id3 -->
    - [device_ip Field Validation Failure in Rank Table File (EI0014)](rank_table_device_ip_field_validation_fail_EI0014.md)
  - [Cluster Information Negotiation](_dump_cluster_info_nego.md)
    - [Business Process and Troubleshooting Approach](cluster_info_nego_troubleshooting.md)
    - [Server Node Port Binding Failure (EI0019)](server_node_port_bind_fail_EI0019.md)
    - [Communicator Initialization Topology Detection Timeout (EI0015)](topo_detect_timeout_EI0015.md)
    - [Communicator Initialization Failure in Typical Multi-Node Scenarios (EI0015)](typical_multi_node_comm_init_fail_EI0015.md)
    - [Some Ranks Cannot Connect to the Server (EI0015)](partial_rank_not_connect_to_server_EI0015.md)
    - [Socket Connection Timeout Between Rank and Server (EI0015)](rank_server_socket_timeout_EI0015.md)
  - [Cluster Information Verification Failure](_dump_cluster_info_verify_fail.md)
    - [Troubleshooting Approach](cluster_info_verify_fail_troubleshooting.md)
    - [Inconsistent IP Family Verification (EI0001)](ip_family_check_inconsistent_EI0001.md)
    - [Inconsistent TLS Information Configuration (EI0016)](tls_info_config_inconsistent_EI0016.md)
    <!-- npu="A3" id2 -->
    - [Duplicate superDeviceId (EI0014)](super_device_id_duplicate_EI0014.md)
    <!-- end id2 -->
- [Parameter Plane Link Setup Stage](_dump_param_link_stage.md)
  - [Troubleshooting Approach for Link Setup Failure](param_link_stage_troubleshooting.md)
  - [Parameter Plane Port Binding Failure (EI0019)](param_port_bind_fail_EI0019.md)
  - [QP Memory Resource Application (EI0011)](qp_mem_resource_apply_EI0011.md)
  - [Link Setup Timeout (EI0006)](link_timeout_EI0006.md)
  - [Parameter Consistency Check (EI0005)](param_consistency_check_EI0005.md)
- [Task Dispatch and Execution Stage](_dump_task_exec_stage.md)
  - [Troubleshooting Approach](task_exec_stage_troubleshooting.md)
  - [Cluster Heartbeat Mechanism](_dump_cluster_heartbeat_mechanism.md)
    - [Troubleshooting Approach](cluster_heartbeat_mechanism_troubleshooting.md)
    - [Process Hang or Peer Heartbeat Loss](process_hang_or_peer_heartbeat_loss_example.md)
  - [task_exception Mechanism](_dump_task_exception_mechanism.md)
    - [Troubleshooting Approach](task_exception_mechanism_troubleshooting.md)
    - [notify_wait Timeout (EI0002)](notify_wait_timeout_EI0002.md)
    - [SDMA_ERROR (EI0012)](sdma_error_EI0012.md)
  - [ERROR_CQE Report (EI0013)](error_cqe_report_EI0013.md)
  <!-- npu="910b" id1 -->
  - [AIV Communication Operator Execution Failure](aiv_comm_op_exec_fail.md)
  <!-- end id1 -->
