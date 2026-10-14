# Collective Communication - AllGatherV

## Sample Description

This sample demonstrates how to use the `HcclAllGatherV()` API to perform the AllGatherV operation. It covers the following functions:

- Call `aclrtGetDeviceCount()` to detect devices and query the number of available devices.
- Call `HcclGetRootInfo()` and use `rank 0` as the root rank to generate the rootinfo identifier.

  > The rootinfo identifier contains the device IP address and device ID. This information must be broadcast to all ranks in the cluster to initialize the communicator.

- In each thread, call `HcclCommInitRootInfo()` to initialize the communicator based on the rootinfo identifier.
- Call `HcclAllGatherV()` to concatenate the input data of all ranks in the communicator in the rank ID order, send the result to all ranks, and display the result.

Unlike AllGather, AllGatherV allows different ranks in the communicator to configure different data sizes. In this sample, the send data size of rank i is set to `i + 1`. `recvCounts` and `recvDispls` describe the length and offset of each rank's data in `recvBuf`, demonstrating the V (Variable) semantics.

## Directory Structure

```
├── main.cc       # Sample source file
├── Makefile      # Compilation and build configuration file
└── allgatherv    # Compiled executable file
```

## Environment Preparation

### Environment Requirements

This sample supports the following products in a single-server N-card configuration (N >= 2):

- Ascend 950PR&950DT products
- Atlas A3 products
- Atlas A2 training products
- Atlas inference products

> Note: The AllGatherV operator does not support Atlas training products, so this sample does not support them either. For details, see [HcclAllGatherV](../../../docs/en/api_ref/comm_op_interface/HcclAllGatherV.md).

### Setting Environment Variables

```bash
# Set CANN environment variables. The following uses the root user default installation path as an example.
source /usr/local/Ascend/cann/set_env.sh
```

## Compiling and Running the Sample

Run the following commands in the sample code directory:

```bash
make
make test
```

> Note: You can set the `HCCL_OP_EXPANSION_MODE` environment variable to configure the expansion mode of communication operators. For the supported ranges for different product models, see the usage instructions for this environment variable in the [Environment Variable List](https://hiascend.com/document/redirect/CannCommunityEnvRef).
>
> ```bash
> # Set the communication operator expansion mode to the AI CPU communication engine
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## Sample Output

Rank i sends `i + 1` elements, and the data content is initialized to the rank ID of that rank. After the AllGatherV operation, the result on each rank is the concatenation of data from all ranks. Therefore, the output length equals the sum of all ranks' data sizes, and the content is rank 0 repeated once, rank 1 repeated twice, and so on.

The following uses 8 cards as an example:

```text
Found 8 NPU device(s) available
rankId: 0, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
rankId: 1, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
rankId: 2, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
rankId: 3, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
rankId: 4, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
rankId: 5, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
rankId: 6, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
rankId: 7, output: [ 0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 ]
```
