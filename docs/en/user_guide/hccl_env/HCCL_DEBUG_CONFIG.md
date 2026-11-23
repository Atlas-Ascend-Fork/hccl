# HCCL_DEBUG_CONFIG

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-06T09:46:15.372Z pushedAt=2026-11-16T06:28:10.141Z nl=2 -->

## Function

Enabling this environment variable will include detailed running information of specific HCCL submodules in the run log (the log in the `$HOME/ascend/log/run` directory). Currently, the following configuration items are supported: `ALG` or `alg` (algorithm orchestration module), `TASK` or `task` (task orchestration module), and `RESOURCE` or `resource` (resource management module, including resource application and release operations).

You can set this environment variable in the following two ways:

- Forward configuration: Configure one or more modules, separated by commas. TASK (or task), ALG (or alg), and RESOURCE (or resource) are case-insensitive.

    ```bash
    # Record the running information of the task module in the running log.
    export HCCL_DEBUG_CONFIG="TASK" 
    # Record the running information of the alg, task, and resource modules in the running log.
    export HCCL_DEBUG_CONFIG="alg,task,resource" 
    ```

- Backward configuration: Add `^` before the first module name, indicating that the detailed running information of all other modules except the configured submodules is recorded in the run log.

    ```bash
    # Record the running information of all modules except the task module in the running log (that is, record the running information of the alg and resource modules).
    export HCCL_DEBUG_CONFIG="^task"
    # Record the running information of all modules except the task and alg modules in the running log (that is, record the running information of the resource module).
    export HCCL_DEBUG_CONFIG="^task,alg"
    ```

**Note**

- Extra spaces will nullify the environment variable setting. For example, in `export HCCL_DEBUG_CONFIG="alg, task "`, there are extra spaces before and after `task`, making this config invalid.
- The TASK module takes effect when it is enabled in either HCCL_DEBUG_CONFIG or HCOMM_DEBUG_CONFIG. For details, see [HCOMM_DEBUG_CONFIG](./HCOMM_DEBUG_CONFIG.md).

**Recommendation**: When the TASK module log prints communication operator call information, set `HCCL_ENTRY_LOG_ENABLE=1` to print the call logs of communication operators in real time, so that TASK logs of different operator intervals can be distinguished. For details, see [HCCL_ENTRY_LOG_ENABLE](./HCCL_ENTRY_LOG_ENABLE.md).

## Configuration Example

```bash
export HCCL_DEBUG_CONFIG="ALG,TASK,RESOURCE" 
```

## Constraints

None.

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
