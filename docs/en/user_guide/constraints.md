# System Constraints and Limitations

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:56:49.558Z pushedAt=2026-10-14T07:59:11.340Z -->

## General Constraints

- Collective communication is not supported in Ascend virtualization instance scenarios.
  > An Ascend virtualization instance refers to a vNPU (virtual NPU instance) created by partitioning the NPU configured on a physical machine or VM through resource virtualization and mounting it to the target environment. The virtualization management approach enables unified allocation and reclamation of resources with different specifications, meeting the operational requirements of multiple users repeatedly applying for and releasing resources.
- When the ACL Graph feature is used, if a captured Graph instance contains communication operators, all Ranks in the communicator must execute the same Graph instance generated from the same capture session during the execution phase. Pairing Graph instances generated from different capture sessions to execute communication operations is not supported. Even if the internal operator sequences of these Graph instances are identical, communication operator execution exceptions will occur.
- In single-machine multi-container scenarios, it is recommended to configure the same IP address for all containers to avoid performance degradation compared with bare-metal environments in performance comparison tests.

<!-- npu="950" id3 -->
## Ascend 950PR&950DT Products

- In scenarios where Ascend computing modules are adapted to third-party x86 servers, the operating system kernel version must be later than 5.18. If this requirement is not met, the haveged entropy supplement tool must be manually installed and its service started.
- Single-card multi-process scenarios:
    Because CCU does not support multi-process usage, in single-card multi-process scenarios, ensure that the creation of the first communicator on different NPUs belongs to the same group of service processes, so that the CCU is occupied by the services of the same group of processes.<br>If the expansion mode of the communication operator is AICPU_TS (default value), the single-card process concurrency count is recommended not to exceed 6.<br>If the expansion mode of the communication operator is AIV, single-card multi-process concurrent execution is not recommended, and serial execution is recommended among multiple processes.<br>Follow the recommended configuration above; otherwise, there is a risk of task deadlock. The expansion mode of the communication operator can be set through the environment variable "HCCL_OP_EXPANSION_MODE".
- In graph mode (Ascend IR) or graph capture (aclgraph) scenarios, when the communication algorithm uses the default AI CPU mode, the number of concurrent graphs on a single card must not exceed 6. Otherwise, communication may be blocked due to AI CPU cores being fully occupied.
- A new CCU scheduling mode is added, but it is limited by the CCU hardware resource specifications. It is recommended to use it flexibly based on actual service scenarios.  
    CCU applicable scenarios: For communication operators within 64P (AlltoAllV operator within 16P), you can consider enabling CCU mode to improve communication performance. CCU resources cannot be reused across communicators. In multi-communicator scenarios, evaluate whether the CCU hardware resources are sufficient.  
    CCU configuration methods: HCCL collective communication uses AI CPU mode by default. To enable CCU, the following two configuration methods are provided:  
        1. When CCU scheduling mode is enabled through an environment variable, HCCL falls back to AI CPU mode if resources are insufficient. For configuration details, see [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md).  
        2. Configure CCU enablement at the communicator granularity to manage CCU communicators independently and plan CCU resource usage as needed. For configuration details, see [HcclCommInitClusterInfoConfig](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/comm_mgr_c/HcclCommInitClusterInfoConfig.md).  
    CCU hardware resource specifications:

    | CCU Mode | Resource Category | Unit | Hardware Resource Specification |
    | --- | --- | --- | --- |
    | Scheduling mode | Channel | units | 120 |
    | Scheduling mode | Loop | units | 68 |
    | Scheduling mode | CCU Buffer | KB | 2048 |
<!-- end id3 --> 

<!-- npu="A3" id4 -->
## Atlas A3 Products

- If your driver firmware is version 25.0.RC1 or later, single-card multi-process service scenarios are supported, meaning that multiple service processes can share one NPU simultaneously. Note that multi-process execution may have a certain impact on resource overhead and communication performance. If too many processes run on the same NPU, service execution may fail due to insufficient resources. If your driver firmware does not meet the version requirement, single-process execution is used.

    If the expansion mode of the communication operator is AI_CPU (default value), it is recommended that the number of concurrent processes on a single card not exceed 6. If the expansion mode of the communication operator is AIV, concurrent execution of multiple processes on a single card is not recommended, and it is advised that multiple processes be executed serially. Follow the configuration recommendations above; otherwise, there is a risk of task deadlock. The expansion mode of the communication operator can be set through the environment variable "HCCL_OP_EXPANSION_MODE".

- It is recommended that the number of servers in each SuperPoD be consistent, and the number of AI processors in each server be consistent. Inconsistency may cause performance degradation.
- In graph mode (Ascend IR) or graph capture (aclgraph) scenarios, when the communication algorithm uses the default AI CPU mode, the number of concurrent graphs on a single card must not exceed 6. Otherwise, communication may be blocked due to AI CPU cores being fully occupied.
<!-- end id4 -->

<!-- npu="910b" id5 -->
## Atlas A2 Products

- If your driver firmware is version 25.0.RC1 or later, single-card multi-process service scenarios are supported, meaning that multiple service processes can share one NPU simultaneously. Note that multi-process execution may have a certain impact on resource overhead and communication performance. If too many processes run on the same NPU, service execution may fail due to insufficient resources. If your driver firmware does not meet the version requirement, single-process execution is used.

    If the expansion mode of the communication operator is HOST (default value), the concurrency count of single-card processes should not exceed 8. If the expansion mode of the communication operator is AIV, concurrent execution of single-card multi-process is not recommended, and serial execution among multiple processes is advised. Follow the configuration recommendations above; otherwise, there is a risk of task deadlock. The operator expansion mode for the Atlas A2 products can be set through the environment variable "HCCL_OP_EXPANSION_MODE".

- In a single-server scenario, there is no limit on the number of AI processors participating in collective communication. In a server cluster scenario, the number of AI processors participating in collective communication must be (1 to 8) * n, where n is the number of servers participating in training. It is recommended that the number of AI processors participating in collective communication in each server be consistent; otherwise, performance degradation may occur.
<!-- end id5 -->

<!-- npu="910" id1 -->
## Atlas Training Products

- Single-card multi-process service scenarios are not supported, meaning that multiple service processes cannot share a single NPU simultaneously.
- In a single-server scenario, the number of AI processors that actually participate in collective communication can only be 1/2/4/8, and cards 0-3 and cards 4-7 each form a network group. When using 2 or 4 cards for training, creating a device cluster across network groups is not supported. In a server cluster scenario, the number of AI processors participating in collective communication can only be 1\*n, 2\*n, 4\*n, 8\*n (where n is the number of servers participating in training), and cluster performance is optimal when n is a power of 2. It is recommended that users prioritize this method for cluster networking. <!-- end id1 -->

<!-- npu="310p" id2 -->
## Atlas 300I Duo Inference Card

- Single-card multi-process service scenarios are not supported, meaning that multiple service processes cannot share a single NPU simultaneously.
- Only single-server scenarios are supported. For details about the maximum number of NPUs supported by each collective communication operation, see the [specific API](../api_ref/comm_op_interface/README.md).
- Communicator initialization must be performed before any other operations involving device memory allocation. Otherwise, initialization may fail due to insufficient P2P memory.
<!-- end id2 -->
