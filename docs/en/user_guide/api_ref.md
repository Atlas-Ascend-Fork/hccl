# API Reference

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-09T03:29:20.548Z pushedAt=2026-11-12T08:45:32.492Z nl=2 -->

HCCL provides communicator management APIs and communication operator APIs, supporting framework developers in implementing distributed capabilities.

- [Communicator Management Interface](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/README.md): Provides APIs for creating, destroying, and handling exceptions of communicators.
  
  Communicator management APIs support both C and Python. The C APIs are used to implement framework adaptation in single operator mode and enable distributed capabilities. The Python APIs are used to implement framework adaptation in graph mode and are currently used only for distributed optimization of TensorFlow networks on NPUs.

- [Communication Operator Interface](../api_ref/comm_op_interface/README.md): Provides two major categories of APIs: collective communication and point-to-point communication.
