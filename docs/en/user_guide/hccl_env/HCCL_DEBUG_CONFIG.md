# HCCL_DEBUG_CONFIG

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T03:51:59.584Z pushedAt=2026-10-15T08:46:37.169Z -->

## Function

Enabling this environment variable will include detailed running information of specific HCCL submodules in the run log (the log in the `$HOME/ascend/log/run` directory). Currently, the following configuration items are supported: `ALG` or `alg` (algorithm orchestration module), `TASK` or `task` (task orchestration module), and `RESOURCE` or `resource` (resource management module, including resource application and release operations).

You can set this environment variable in the following two ways:

- Forward configuration: Configure one or more modules, separated by commas. TASK (or task), ALG (or alg), and RESOURCE (or resource) are case-insensitive.

    ```bash
    # Record the runtime information of the task module in the run logs.
    export HCCL_DEBUG_CONFIG="TASK" 
    # Record the runtime information of the alg, task, and resource modules in the run logs.
    export HCCL_DEBUG_CONFIG="alg,task,resource" 
    ```

- Backward configuration: Add `^` before the first module name, indicating that the detailed running information of all other modules except the configured submodules is recorded in the run log.

    ```bash
    # Record the runtime information of all modules except the task module in the run logs (that is, the runtime information of the alg and resource modules is recorded).
    export HCCL_DEBUG_CONFIG="^task"
    # Record the runtime information of all modules except the task and alg modules in the run logs (that is, the runtime information of the resource module is recorded).
    export HCCL_DEBUG_CONFIG="^task,alg"
    ```

**Note**

- Extra spaces will nullify the environment variable setting. For example, in `export HCCL_DEBUG_CONFIG="alg, task "`, there are extra spaces before and after `task`, making this config invalid.
- The TASK module takes effect when it is enabled in either HCCL_DEBUG_CONFIG or HCOMM_DEBUG_CONFIG. For details, see [HCOMM_DEBUG_CONFIG](./HCOMM_DEBUG_CONFIG.md).

**Recommendation**: When TASK module logs print communication operator call information, set `HCCL_ENTRY_LOG_ENABLE=1` to print the operator call logs in real time, so as to distinguish TASK logs of different operator intervals. For details, see [HCCL_ENTRY_LOG_ENABLE](./HCCL_ENTRY_LOG_ENABLE.md).

## Configuration Example

```bash
export HCCL_DEBUG_CONFIG="ALG,TASK,RESOURCE" 
```

## Constraints

None

## Applicable Products

<!-- npu="950" id3 -->
- Ascend 950PR&950DT products: Supported
<!-- end id3 -->
<!-- npu="A3" id1 -->
- Atlas A3 products: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2 products: Supported
<!-- end id2 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
