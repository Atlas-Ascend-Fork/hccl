# HCCL_ALGO

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-06T09:46:06.335Z pushedAt=2026-11-16T06:28:10.135Z nl=2 -->

## Function

Configures the inter-server communication algorithm and inter-SuperPoD communication algorithm for collective communication. It supports two configurations: globally configuring the algorithm type and configuring the algorithm type by operator.

> [!NOTE]
>
> - HCCL selects an algorithm adaptively. By default, it selects a proper algorithm based on product form, data volume, and number of servers. Generally, you do not need to specify it manually. If you specify an inter-server or inter-SuperPoD communication algorithm through this environment variable, adaptive algorithm selection no longer takes effect.
> - For certain communication operators, when a specific type of AI processor is used and the data volume is small, HCCL selects a communication algorithm adaptively, which is not controlled by this environment variable.

<!-- npu="950" id2 -->
## Configuration Description (Ascend 950PR&950DT Products)

### Global Configuration

The global configuration is the configuration without the operator OpType specified, and it applies to all operators.

```bash
export HCCL_ALGO="<ExecutorType>{level0=<algoType>,level1=<algoType>};not(<ExecutorType>{level3=<algoType>})"
```

- `<ExecutorType>{level0=<algoType>,level1=<algoType>}`: Specifies the algorithm executor and defines the algorithms used at different levels under this executor.<a id="executor_type_define"></a>
  - `<ExecutorType>` is the type of the algorithm executor and supports the following configurations:
    - Sole (sole): Single-algorithm executor.
    - Sequence (sequence): Sequential executor that executes a group of algorithms serially. Currently, only level-2 or level-3 algorithms are supported.
    - Parallel (parallel): Two-dimensional parallel executor that executes different algorithms in parallel in two dimensions at the same time.
    - PipeLine (pipeline): Pipeline parallel executor.
    - Concur (concur): Concurrent executor that executes different algorithms in parallel at the same level at the same time.
    - StrictOrdered (strictordered): Strict ordered executor.

    If `<ExecutorType>` is not configured, the Sole single-algorithm executor is used by default.

  - `level*=<algoType>`: Specifies the algorithm used at each level. Multiple level configurations use commas (,) as the separator.<a id="algo_define"></a>

    - `level*`: Algorithm level to configure, for example, level0 and level1.
    - `<algoType>`: Type of the communication algorithm, which supports the following configurations:

      - Mesh (mesh): a full-mesh algorithm. Each rank establishes a direct connection with all other ranks and sends/receives directly (completed in one step), achieving optimal latency. It is suitable for intra-node full-mesh topologies with direct connections or cross-node low-latency use cases with a small data volume.
      - NHR (nhr): a non-uniform hierarchical ring algorithm. It completes in log2(N) steps through recursive halving + doubling, and introduces no extra communication volume even when the scale is not an integer power of 2. It is suitable for inter-server/inter-SuperPoD use cases with a large data volume under multi-layer topologies.
      - Mesh2Die (mesh2die): a mesh algorithm with ports distributed across dies. It groups mesh communication tasks by local die, and the two die groups concurrently perform direct send/receive and then aggregate results. It is applicable to topologies where full-mesh links are distributed across two dies.
      - MeshOneShot (meshoneshot): a OneShot mesh algorithm. Each rank directly writes the complete input to the other N-1 peers, then locally reduces N copies of data. It completes in one step with optimal latency, but the traffic is (N-1) multiplied by the full data volume. It is only suitable for AllReduce use cases with a small data volume.
      - MeshTwoShot (meshtwoshot): a TwoShot mesh algorithm. It splits AllReduce into two steps: ReduceScatter + AllGather (first split into N slices for distribution and reduction, then full collection). The communication volume is about 2/N of OneShot. It is suitable for use cases with a large communication data volume, and is applicable to AllReduce, Reduce, and Broadcast operators.
      - MeshConcur (meshconcur): a CLOS-axis detour algorithm. It combines the level0 direct mesh links and the level1 CLOS uplinks, splits each peer's traffic according to the bandwidth ratio between the two sides, and routes part of the data "upward" as a detour. The algorithm uses both full-mesh and CLOS links.
      - MeshConcurrent (meshconcurrent): a mesh/CLOS dual-path concurrency algorithm. Within the same group of ranks, it jointly uses mesh direct links and CLOS links, divides each peer's send and receive data into two parts according to a preset bandwidth ratio, and processes them concurrently. It is applicable to AllToAll/AlltoAllV operators.
      - MeshChunk (meshchunk): a chunk-enabled mesh algorithm. It splits data into multiple chunks (N-1 chunks and (N-1) steps under ReduceScatter) and exchanges them with peers step by step, fusing send/receive with reduction. It is applicable to ReduceScatter over full-mesh direct links.
      - MeshChunkTwoShot (meshchunktwoshot): a TwoShot variant of the MeshChunk algorithm. AllReduce follows a two-step ReduceScatter + AllGather flow, with mesh chunk used in the ReduceScatter stage.
      - MeshSingleChannel (meshsinglechannel): a single-channel mesh algorithm. It uses the same algorithm as the default mesh, but establishes only one channel per pair of ranks and does not split data across multiple channels. It is suitable for AllToAll operators that need to reduce channel/stream synchronization overhead or are constrained by link resources.
      - Pairwise (pairwise): a pairwise communication algorithm. In a multi-node, multi-device use case, each node (with multiple devices) acts as a group. In each round, a target group is paired according to a symmetric pairing matrix to avoid one-to-many network traffic. Each device on the local node establishes channels and concurrently sends/receives data with the peers in the target group on a pairwise basis. This algorithm is currently supported only by AllToAll/AlltoAllV/AlltoAllVC (AICPU). Requirements: level0 must be MESH_1D, 8 devices per node, and rankSize (a multiple of 16) ≥ 16.
      - MeshMultiLink (meshmultilink): a multi-link mesh algorithm for topologies where each pair of ranks has multiple links. Links are partitioned based on the local die, and data is split and sent concurrently over the links of the two dies according to the number of ports contained in each link. It is applicable to AllToAll/AllToAllV operators whose ports are distributed across two dies.
      - NHRMultiLink (nhrmultilink): a multi-link NHR algorithm. Each link may contain multiple physical ports, and in each step the data to exchange is sliced across links according to the port count and sent concurrently. It is applicable to topologies where each pair of ranks has multiple links.
      - MeshMultiJetty (meshmultijetty): a multi-jetty mesh algorithm. It establishes multiple channels for each peer (one jetty per channel), slices data by jetty, and aggregates and sends it concurrently over multiple ports. It can be used for AllToAll/AlltoAllV operators where each pair of ranks has multiple physical ports.
      - NHRMultiJetty (nhrmultijetty): a multi-Jetty NHR algorithm. It establishes multiple channels per peer (one jetty per channel), slices data by jetty, and aggregates and sends it concurrently over multiple ports. It is suitable for NHR communication where each pair of ranks has multiple physical ports.
      - NHRAicpuReduce (nhraicpureduce): an NHR algorithm variant that first collects data through NHR-style log2(N)-step halving exchanges, then uses AICPU scalar reduction to ensure correctness. It is applicable to data types not supported by standard NHR/mesh templates (e.g., INT64/UINT64/FP64 and PROD reduction).
      - HDAicpuReduce (hdaicpureduce): an HD recursive halving algorithm for ReduceScatter (AICPU engine). It performs log2(N) steps of XOR-paired exchanges, with in-step reduction completed on the DMA transfer. It requires the number of ranks in the communicator to be a power of 2, and is considered in algorithm selection only when explicitly configured (the default selection remains unchanged).
      - HD (hd): a cross-node (level1) HD recursive halving algorithm (AICPU engine). Level0 intra-node communication uses mesh, while level1 performs log2(N) steps of halving exchanges with reduction. It requires the rank count at this level to be a power of 2, and is considered in algorithm selection only when explicitly configured (the default selection remains unchanged).

    - If `level` is not specified, `<algoType>` is matched by algorithm level from low to high by default. For example, {Mesh, NHR} indicates that the mesh algorithm is used at level0 and the NHR algorithm is used at level1.

- `not(a1)`: negation, indicating that a certain algorithm is not supported and all others are selectable. Negation can be applied to `<algoType>`, and also to `executor{a1,a2}`.

Configuration example:

```bash
# All operators at all levels prefer the NHR algorithm, using the default algorithm executor Sole.
export HCCL_ALGO="nhr"
# No operator uses the sole(nhr) algorithm.
export HCCL_ALGO="not(nhr)"
# Algorithms execute sequentially. level0 executes the mesh algorithm, and level1 executes the NHR algorithm.
export HCCL_ALGO="sequence{mesh,nhr}"
# Algorithms execute sequentially. level0 can execute algorithms other than the NHR algorithm, and level1 executes the NHR algorithm.
export HCCL_ALGO="sequence{level0=not(nhr),level1=nhr}"
# The parallel{mesh,nhr} algorithm is not supported (that is, level0 uses mesh, level1 uses NHR, and level0 and level1 execute in parallel). Other options can be selected.
export HCCL_ALGO="not(parallel{mesh,nhr})"
```

### Configuration by Operator Type

Specify the operator type OpType. The configured algorithm takes effect only for the specified operator.

```bash
export HCCL_ALGO="<OpType>:<ExecutorType>{level0=<algoType>,level1=<algoType>,...};<OpType>:<ExecutorType>{...}"
```

- `<OpType>`: Type of the communication operator. It supports the following options:
  - AllGather (allgather)
  - AllGatherV (allgatherv)
  - AllReduce (allreduce)
  - AllToAll (alltoall)
  - AllToAllV (alltoallv)
  - AllToAllVC (alltoallvc)
  - Broadcast (broadcast)
  - Reduce (reduce)
  - ReduceScatter (reducescatter)
  - ReduceScatterV (reducescatterv)
  - Scatter (scatter)
- `<ExecutorType>`: Type of the algorithm executor. The supported configurations are the same as those in [ExecutorType in the global configuration](#executor_type_define).
- `level*=<algoType>`: Algorithm used at each level. Configurations of multiple levels are separated by commas (,). The configuration method is the same as that in [algoType in the global configuration](#algo_define).
- Configurations of multiple operators are separated by semicolons (;).

Configuration example:

```bash
# The AllReduce operator uses the Sequence executor to execute algorithms sequentially. level0 executes the mesh algorithm, and level1 executes the NHR algorithm.
export HCCL_ALGO="allreduce:sequence{mesh,nhr}"
# The AllGather operator is prohibited from using the Parallel executor with level0 executing the mesh algorithm and level1 executing the NHR algorithm.
export HCCL_ALGO="allgather:not(parallel{mesh,nhr})"
```

### Configuration Effectiveness Semantics

For Ascend 950PR&950DT products, the HCCL_ALGO configuration only provides algorithm priorities:

- Positively specified algorithms take effect with priority. When multiple positive configurations coexist, the optimal algorithm is selected within the specified algorithm set based on communication cost.
- When multiple configurations exist at the same algorithm level, the later configuration overrides the earlier one.

### Behaviors Under Configuration Errors

Configuration errors do not cause communication failures. In the worst case, the system falls back to adaptive algorithm selection. The error behaviors and their corresponding code print keywords (all at WARNING log level) are as follows:

- Configuration format error: The entire setting does not take effect. The system prints a warning message, does not block the communication process, and falls back to adaptive algorithm selection.
  - Print keyword: `parse algo config failed, the config has not taken effect`
- The algorithm combination configured by operator type does not match any registered algorithm: The clause does not take effect. The system prints a warning message and falls back to adaptive algorithm selection.
  - Print keyword: when a single clause does not match, print `HCCL_ALGO clause for opType=<operator type> matched no candidate algorithm`; when the entire configuration does not match, print `HCCL_ALGO config matched no candidate algorithm under current engine candidates`.
- The specified algorithm does not match the current communication use case (for example, hard constraints such as topology or data type are not met): The algorithm is excluded from algorithm selection. The system prints a warning message and falls back to adaptive algorithm selection.
  - Print keyword: when the topology constraints are not satisfied, print `HCCL_ALGO configured algName=<algorithm name> skipped by topo filter, fallback to auto selection.`; when operator constraints such as data type are not satisfied, print `HCCL_ALGO configured algName=<algorithm name> filtered: <filter reason>, fallback to auto selection.`
- The negative configuration excludes all available algorithms for the current use case: The system prints a warning message, ignores the negative configuration, and adaptively selects from all candidate algorithms to ensure normal communication execution.
  - Print keyword: `all survivors are hcclAlgo-negated, fallback to full candidates.`

If no algorithm is available in the current use case after fallback (algorithm selection fails completely), the system prints an ERROR-level message. Major log keywords are as follows:

- `No support alg for current topoType/opInfo/externalConfig, please check.`
- `Failed to select algorithm with opExecuteConfig=<engine>.`
- `algName=<algorithm name> filtered by <engine|attrs|topo|executor|op>: <filter reason>`
<!-- end id2 -->

<!-- npu="A3,910b,910" id3 -->
## Configuration Description (Atlas A3 Products/Atlas A2 Products/Atlas Training Products)

### Global Configuration

```bash
export HCCL_ALGO="level0:NA;level1:<algo>;level2:<algo>"
```

- level0 represents the intra-server communication algorithm. Currently, only `NA` is supported.
- level1 represents the inter-server communication algorithm, which supports the following values:
  - ring: a ring-based communication algorithm with many communication steps (linear complexity) and relatively high latency, but simple communication relationships and less impact from network congestion. It is suitable for use cases where the number of servers within the communicator is small, the communication data volume is small, the network has significant congestion, and the pipeline algorithm is not applicable.
  - H-D_R: Recursive Halving-Doubling (RHD) algorithm with few communication steps (logarithmic complexity) and relatively low latency, but it introduces additional communication overhead when the number of nodes is not an integer power of 2. It is suitable for use cases where the number of servers within the communicator is an integer power of 2 and the pipeline algorithm is not applicable, or where the number of servers is not an integer power of 2 but the communication data volume is small.
  - NHR: Nonuniform Hierarchical Ring algorithm with few communication steps (logarithmic complexity) and relatively low latency. It is suitable for use cases where the number of servers within the communicator is large and the pipeline algorithm is not applicable.
  - NHR_V1: The earlier version of the NHR algorithm, which involves fewer communication steps (square root complexity) and relatively low latency. It is ideal when the number of servers in the communicator is not an integer power of 2 and the pipeline algorithm cannot apply. The theoretical performance of this algorithm is lower than that of the new NHR algorithm. This configuration option will be gradually deprecated. You are advised to use the NHR algorithm.
  - NB: Nonuniform Bruck algorithm. It has few communication steps (logarithmic complexity) and relatively low latency. It is suitable for use cases where the number of servers in the communicator is large and the pipeline algorithm is not applicable.
  - AHC: Asymmetric Hierarchical Concatenate algorithm. It is suitable for use cases where NPU distribution within the communicator has multiple levels, with symmetric or asymmetric NPU distribution across levels (i.e., asymmetric device count). The benefit is more tangible when bandwidth convergence exists between levels within the communicator.
    Note: When level1 (inter-server communication algorithm) is set to `AHC`, level2 (inter-SuperPoD communication algorithm) automatically adopts the AHC algorithm without additional configuration. Even if other algorithms are set for level2, these settings will not take effect.
  - pipeline: a pipeline parallel algorithm. It can concurrently use intra-server and inter-server links. It is suitable for use cases with large communication data volumes where each machine in the communicator contains multiple devices.
  - pairwise: a pairwise communication algorithm, used only for AlltoAll, AlltoAllV, and AlltoAllVC operators. It involves a large number of communication steps (linear complexity) and relatively high latency, and requires additional memory allocation with the memory size proportional to the data volume. However, it is suitable for use cases with large communication data volumes where one-to-many network patterns need to be avoided.

  When level1 is not set:
    <!-- npu="A3" id4 -->
  - For Atlas A3 products, the algorithm is automatically selected based on the product form, number of nodes, and data volume.
    <!-- end id4 -->
    <!-- npu="910b" id5 -->
  - For Atlas A2 products, the algorithm is automatically selected based on the product form, number of nodes, and data volume.
    <!-- end id5 -->
    <!-- npu="910" id6 -->
  - For Atlas training products, when the number of servers in the communicator is not an integer power of 2, the ring algorithm is used by default; in other cases, the H-D_R algorithm is used by default.
    <!-- end id6 -->
- level2 represents the inter-SuperPoD communication algorithm, which supports the following values:
  - ring: a ring-structure-based communication algorithm with many communication steps (linear complexity) and relatively high latency, but simple communication relationships and less susceptibility to network congestion. Suitable for use cases where there are few SuperPoDs within the communicator and the number is not an integer power of 2.
  - H-D_R: Recursive Halving-Doubling (RHD) algorithm with few communication steps (logarithmic complexity) and relatively low latency, but introduces additional communication at non-integer-power-of-2 node scales. Suitable for use cases where the number of SuperPoDs within the communicator is an integer power of 2, or where the number of SuperPoDs is not an integer power of 2 but the communication data volume is small.
  - NHR: Nonuniform Hierarchical Ring algorithm with few communication steps (logarithmic complexity) and relatively low latency. Suitable for use cases where there are many SuperPoDs within the communicator.
  - NB: Nonuniform Bruck algorithm, featuring fewer communication steps (logarithmic complexity) and relatively low latency. Suitable for use cases with a large number of SuperPoDs within the communicator.
  - pipeline: a pipeline parallel algorithm that can concurrently use intra-SuperPoD and inter-SuperPoD links. Suitable for use cases with large communication data volume where each SuperPoD within the communicator contains multiple devices.

  When level2 is not set, the ring algorithm is used if the number of SuperPoDs within the communicator is less than 8 and is not an integer power of 2; otherwise, the H-D_R algorithm is used.

  The level2 configuration currently applies only to Atlas A3 products and supports only use cases where the communication operator expansion mode is AI_CPU.

  The communication operator expansion mode can be configured through the environment variable [HCCL_OP_EXPANSION_MODE](HCCL_OP_EXPANSION_MODE.md).

  For the inter-server communication algorithms and inter-SuperPoD communication algorithms supported by different products, see [Inter-Server Communication Algorithm Support List](inter_server_algo_support.md) and [Inter-SuperPoD Communication Algorithm Support List](inter_superpod_algo_support.md).

Configuration example:

```bash
export HCCL_ALGO="level0:NA;level1:NHR"
```

### Configuration by Operator Type

```bash
export HCCL_ALGO="<op0>=level0:NA;level1:<algo0>;level2:<algo1>/<op1>=level0:NA;level1:<algo3>;level2:<algo4>"
```

- <op\> specifies the communication operator type. The following settings are allowed:
  - allgather: corresponds to the communication operators AllGather and AllGatherV.
  - reducescatter: corresponds to the communication operators ReduceScatter and ReduceScatterV.
  - allreduce: corresponds to the communication operator AllReduce.
  - broadcast: corresponds to the communication operator Broadcast.
  - reduce: corresponds to the communication operator Reduce.
  - scatter: corresponds to the communication operator Scatter.
  - alltoall: Corresponds to the communication operators AlltoAll, AlltoAllV, and AlltoAllVC.
- <algo\> specifies the communication algorithm used by the communication operator. The supported configurations are the same as the level1 value and level2 values in the global configuration method. Ensure that the specified communication algorithm is of a type supported by the communication operator. For the algorithms supported by each operator, see [Inter-Server Communication Algorithm Support List](inter_server_algo_support.md) and [Inter-SuperPoD Communication Algorithm Support List](inter_superpod_algo_support.md). Communication operators for which no algorithm is specified will automatically select a communication algorithm based on the product form, number of nodes, and data volume.
- Use `/` to separate configurations of multiple operators.

Configuration example:

```bash
# The AllReduce operator uses the Ring algorithm, the AllGather operator uses the RHD algorithm, and other operators automatically select the communication algorithm based on the product form, number of nodes, and data volume.
export HCCL_ALGO="allreduce=level0:NA;level1:ring/allgather=level0:NA;level1:H-D_R"
```
<!-- end id3 -->

## Constraints

- If you call the HCCL C API to initialize a communicator with specific configurations and specify a communication algorithm through the `hcclAlgo` parameter of `HcclCommConfig`, the communicator-level configurations take precedence.
- If the input HCCL_ALGO configuration string is incorrectly formatted, the configuration does not take effect. The system prints a warning message, but the subsequent execution flow is not blocked.
<!-- npu="910b" id7 -->
- For Atlas A2 products, do not configure the HCCL_ALGO environment variable in ordered use cases that require strict deterministic computation.
<!-- end id7 -->

## Applicable Products

<!-- npu="950" id10 -->
- Ascend 950PR&950DT products: Supported
<!-- end id10 -->
<!-- npu="A3" id9 -->
- Atlas A3 products: Supported
<!-- end id9 -->
<!-- npu="910b" id8 -->
- Atlas A2 products: Supported
<!-- end id8 -->
<!-- npu="910" id1 -->
- Atlas training products: Supported
<!-- end id1 -->
<!-- npu="310p" id11 -->
- Atlas inference products: Not supported
<!-- end id11 -->
