# Collective Communication - ReduceScatterV

## Sample Description

This sample demonstrates how to use the `HcclReduceScatterV()` API to perform the ReduceScatterV operation. It covers the following functions:

- Call `aclrtGetDeviceCount()` to detect devices and query the number of available devices.
- Call `HcclGetRootInfo()` and use `rank 0` as the root rank to generate the rootinfo identifier.

  > The rootinfo identifier contains the device IP address and device ID. This information must be broadcast to all ranks in the cluster to initialize the communicator.

- In each thread, call `HcclCommInitRootInfo()` to initialize the communicator based on the rootinfo identifier.
- Call `HcclReduceScatterV()` to reduce the input data of all ranks at the same index, scatter the result to each rank by index, and display the result.

Unlike ReduceScatter, ReduceScatterV allows different data blocks with different indices on the same rank to have different sizes. In this sample, the size of the data block with index i is set to `i + 1`. `sendCounts` and `sendDispls` describe the length and offset of each data block, demonstrating the V (Variable) semantics. Note that `sendCounts` and `sendDispls` must be identical on all ranks.

## Directory Structure

```
├── main.cc            # Sample source file
├── Makefile           # Compilation and build configuration file
└── reduce_scatterv    # Compiled executable file
```

## Environment Preparation

### Environment Requirements

This sample supports the following products in a single-server N-card configuration (N >= 2):

- Ascend 950PR&950DT products
- Atlas A3 products
- Atlas A2 training products
- Atlas inference products

> Note: The ReduceScatterV operator does not support Atlas training products, so this sample does not support them either. For details, see [HcclReduceScatterV](../../../docs/en/api_ref/comm_op_interface/HcclReduceScatterV.md).

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

The input data of each rank is initialized to the rank ID of that rank, and the size of the data block with index i is `i + 1`. After the ReduceScatterV operation, the part with index i is the sum of the data at the corresponding position from all ranks, and rank i receives the result segment with index i and length `i + 1`.

The following uses 8 cards as an example. The sum of all rank IDs is `0 + 1 + ... + 7 = 28`, and rank i receives `i + 1` elements. Therefore, the output of each rank is as follows:

```text
Found 8 NPU device(s) available
rankId: 0, output: [ 28 ]
rankId: 1, output: [ 28 28 ]
rankId: 2, output: [ 28 28 28 ]
rankId: 3, output: [ 28 28 28 28 ]
rankId: 4, output: [ 28 28 28 28 28 ]
rankId: 5, output: [ 28 28 28 28 28 28 ]
rankId: 6, output: [ 28 28 28 28 28 28 28 ]
rankId: 7, output: [ 28 28 28 28 28 28 28 28 ]
```
