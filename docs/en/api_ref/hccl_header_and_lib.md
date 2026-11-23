# HCCL Header Files and Library Files

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-06T09:45:06.855Z pushedAt=2026-11-16T06:28:10.114Z nl=2 -->

Built on Ascend AI Processors, Huawei Collective Communication Library (HCCL) delivers high-performance collective and point-to-point communication in single-node and multi-node environments. As one of the core components of CANN, HCCL is decoupled from the communication base library HCOMM through dlsym dynamic loading. HCCL and HCOMM are compiled and versioned independently. For the external header files and library files of HCOMM, see [HCOMM Header Files and Library Files](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/hcomm_header_and_lib.md).

This section describes the header files and library files of HCCL APIs.

## API Categories

HCCL APIs are classified into the following categories by function:

**Table 1** API categories

| API Category | Description |
| --- | --- |
| Collective communication operators | A total of 11 collective communication APIs: AllReduce, Broadcast, AllGather, AllGatherV, ReduceScatter, ReduceScatterV, Scatter, Reduce, AlltoAll, AlltoAllV, and AlltoAllVC. |
| Point-to-point communication operators | A total of 3 point-to-point communication APIs: Send, Recv, and BatchSendRecv. |
| MC2 operators | MC2 (Kernel Fusion Custom) operator framework APIs, including HcclKfc* parameter objects and HcclCreateOpResCtx. |

## Header Files and Library Files Required for Calling APIs

After installing the firmware, driver, and CANN software package, you can reference the HCCL API header files and library files when building and running your app.

The HCCL API header files are in the `${INSTALL_DIR}/include/hccl/` directory, and the library files are in `${INSTALL_DIR}/lib64/`. Replace ${INSTALL_DIR} with the actual CANN installation path. For example, if you install the software as user root, the default installation path is /usr/local/Ascend/cann.

> [!CAUTION]
> When compiling an HCCL API program, link the library files according to the dependencies of the included header files. Linking unnecessary `.so` files may cause version functionality exceptions or compatibility issues during subsequent version upgrades.

Include the required files based on the HCCL APIs you actually use. The purpose of each header file is described in the following table.

**Table 2** Header files

| API-Defining Header File | Purpose | Library File |
| --- | --- | --- |
| hccl/hccl.h | Defines the collective communication and point-to-point communication operator APIs. | libhccl.so |
| hccl/hccl_mc2.h | Defines the MC2 operator framework APIs, including HcclKfc* parameter object allocation/setting and HcclCreateOpResCtx communication resource context creation. | libhccl.so |

HCCL is released as the `cann-hccl_<version>_linux-<arch>.run` installation package, which contains `libhccl.so`, external header files, and `aicpu_hccl.tar.gz` (the HCCL AI CPU operator package). In static builds, `libhccl_static.a` is also generated.

- For the source code build and installation procedure, see [Build Guide](../build/build.md).
- For details about the prototype, parameters, data types, and constraints of communication operator APIs, see [Communication Operator APIs](./comm_op_interface/README.md).
