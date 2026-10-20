# Troubleshooting Approach for Link Setup Failures

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:53:37.514Z pushedAt=2026-10-13T11:44:14.000Z -->

When a communication operator is invoked, HCCL creates socket connections over the Parameter Plane network based on the TCP protocol to exchange information such as addresses according to service requirements. If a fault prevents some ranks from invoking the expected communication operator, they will be unable to initiate link setup requests. Alternatively, if network connectivity issues or behavior consistency problems prevent ranks from responding to each other's link setup requests, socket connection timeout errors will occur on other ranks.

Due to the design of HCCL algorithms and the order in which operators are invoked, link setup timeouts exhibit cascading propagation among ranks. Therefore, when a link setup timeout is detected, the fault point must be located first.

During the parameter plane link setup stage, HCCL provides the mechanisms described below to facilitate rapid problem locating.

## Root Node Location Mechanism for Link Setup

Considering the cascading propagation of link setup issues, for example, rank0 times out while waiting for link setup with rank1, and rank1 times out while waiting for link setup with rank2. If the link setup between rank1 and rank2 fails due to network or other reasons, rank0 will eventually report a link setup timeout error with rank1 as well, but the root cause lies between rank1 and rank2. Therefore, locating the root node of a link setup failure in a cluster is relatively difficult and tedious. HCCL initiates a fault detection link immediately after a service link setup failure. The main implementation principle is as follows:

The schematic diagram of root node location for link setup failure is shown below:

![Schematic Diagram of Root Node Location for Link Setup Failure](figures/link_fail_root_debug.png)

1. After a link setup failure, each rank starts a server side that listens to and can respond to the fault detection links of all ranks.
2. Initiate a fault probe link connection request to the remote end that cannot respond to its own service link setup request.
3. If the remote end cannot respond to its own probe link setup request, the link to the remote end or the service process on the remote end is considered faulty, and a probe failure event is generated. This event is then propagated to other links that have been successfully established on the server side.
4. If the remote end has established a probe link, receive the probe failure event sent by the peer and forward it.

In this way, if a link setup failure is caused by any single point of failure, you can quickly locate the node where the fault occurs through logs and proceed with further fault locating. For the detailed locating process, see [Link Setup Timeout (EI0006)](./link_timeout_EI0006.md#link-setup-timeout-ei0006).

If no event is detected after probing, the issue is most likely a behavioral consistency problem. That is, each rank has entered the link setup stage and responds to fault probe requests from other ranks, but the links experience mutual wait timeout due to inconsistent communication operators invoked by each rank. This is generally caused by a cluster behavioral consistency issue. Check factors such as scripts, environment, versions, and datasets. If the behavior of communication operators needs to be referenced, the operator behavior can be inferred from the tag information corresponding to the keyword "Alloc transports failed" in the link setup failure error log. For example, traverse the tag information of each rank. If, within a 16-rank communicator, 15 ranks are performing allgather while one rank is performing AllReduce, focus on analyzing the differences in the invocation logic of the two operators.

For link setup timeout scenarios, it is possible to quickly determine whether it is a full-scale link setup timeout. If it is not a full-scale link setup timeout, priority should be given to troubleshooting nodes that have not reported link setup timeout errors. The reference command is as follows:

```bash
for i in *;do cd $i;pwd;grep -rnc "connection fail" | grep -v ":0" | wc -l; cd ..;done
```

## Consistency Check Mechanism

After HCCL successfully creates a socket connection with the peer, it exchanges information such as operator input parameters and the CANN version and verifies them against the local information. If any inconsistency is detected during the verification, an error is reported in the CANN log and the console log, and an error code is returned. For the detailed fault locating process, see [Parameter Consistency Check (EI0005)](./param_consistency_check_EI0005.md#parameter-consistency-check-ei0005).

In single-operator mode, to ensure performance, HCCL triggers link setup only when an operator of a new type or algorithm is called for the first time in each communicator. Since the consistency check is performed only after the link setup succeeds, this feature cannot intercept all dispatch inconsistency issues.

## Error Stage Analysis

HCCL has the following common error stage scenarios during the Communication Operator Parameter Plane link setup stage:

  <!-- npu="A3,910b" id1 -->
- If device NIC port binding fails, run the following command to check whether any port binding failure exists. For detailed information, see [Parameter Plane Port Binding Failure (EI0019)](./param_port_bind_fail_EI0019.md#parameter-plane-port-binding-failure-ei0019).

    ```bash
    grep -rE "socket type\[(0|1)\].*Please check the port status and whether the port is being used by other process"
    ```
    
    This operation applies only to the following products:
      <!-- npu="A3" id2 -->
    - Atlas A3 products
      <!-- end id2 -->
      <!-- npu="910b" id3 -->
    - Atlas A2 products
      <!-- end id3 -->
  <!-- end id1 -->

- If the parameter plane socket link setup times out, run the following command to check whether any parameter plane link setup failure exists. For detailed information, see [Link Setup Timeout (EI0006)](./link_timeout_EI0006.md#link-setup-timeout-ei0006).

    ```bash
    grep -r "wait socket establish timeout"
    ```

- If the communication operator consistency check fails, run the following command to check whether any consistency check failure exists. For detailed information, see [Parameter Consistency Check \(EI0005\)](./param_consistency_check_EI0005.md#parameter-consistency-check-ei0005).

    ```bash
    grep -r "CMD information .* check fail"
    ```
