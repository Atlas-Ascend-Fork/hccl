# API Reference

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:56:01.784Z pushedAt=2026-10-14T07:08:09.742Z -->

HCCL provides communicator management APIs and communication operator APIs, supporting framework developers in implementing distributed capabilities.

- [Communicator Management Interface](https://gitcode.com/cann/hcomm/blob/9.2.0/docs/en/api_ref/README.md): provides interfaces for creating, destroying, and handling exceptions of communicators.
  
  Communicator management APIs support both C and Python. The C APIs are used to implement framework adaptation in single operator mode and enable distributed capabilities. The Python APIs are used to implement framework adaptation in graph mode and are currently used only for distributed optimization of TensorFlow networks on NPUs.

- [Communication Operator Interface](../api_ref/comm_op_interface/README.md): provides two major categories of interfaces: collective communication and point-to-point communication.
