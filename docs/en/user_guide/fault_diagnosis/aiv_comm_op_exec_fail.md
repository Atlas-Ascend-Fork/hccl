# AIV Communication Operator Execution Failure

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:51:44.006Z pushedAt=2026-10-13T08:49:23.615Z -->

## Symptoms

For Atlas A2 products, after AIV mode is enabled via export HCCL_OP_EXPANSION_MODE="AIV", HCCL implements the orchestration and execution of HCCL communication operators in kernel execution mode in certain scenarios. In such cases, if a communication operator execution exception occurs, the following critical log line is printed: "fault kernel_name=aiv_all_reduce_***", indicating that the HCCL AIV operator execution has failed:

```text
[ERROR] RUNTIME(699131,python):2025-04-24-21:54:17.707.236 [davinci_kernel_task.cc:1268]699131 PrintErrorInfoForDavinciTask:[INIT][DEFAULT]Aicore kernel execute failed, device_id=0, stream_id=2, report_stream_id=2, task_id=55873, flip_num=2073, fault kernel_name=aiv_all_reduce_***, fault kernel info ext=aiv_all_reduce_910b_bfloat16_t, program id=42, hash=9645272693770703471.
```

In addition, the same task exception information described above is also printed. The root cause of the task failure can still be analyzed using the troubleshooting approach in [Notify Wait Timeout (EI0002)](./notify_wait_timeout_EI0002.md#operator-execution-timeout-ei0002), such as whether a full timeout has occurred or whether a node in the cluster experienced an exception first.
