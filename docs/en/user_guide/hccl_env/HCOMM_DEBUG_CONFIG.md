# HCOMM_DEBUG_CONFIG

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T04:05:06.775Z pushedAt=2026-10-15T08:46:37.246Z -->

## Function

After this environment variable is enabled, the run logs (in the `$HOME/ascend/log/run` directory) contain detailed running information about specific HCOMM submodules. Currently, the following items are supported: TASK or task (task orchestration module), DATA_OP or data_op (data-plane API module), RESOURCE or resource (resource management module, including resource application and release operations), and CHANNEL or channel (channel module).

You can set this environment variable in the following two ways:

- Forward configuration: Configure one or more modules, separated by commas (,). TASK (or task), DATA_OP (or data_op), RESOURCE (or resource), and CHANNEL (or channel) are case-insensitive.

    ```bash
    # The run log records the runtime information of the task module.
    export HCOMM_DEBUG_CONFIG="TASK" 
    # The run log records the runtime information of the task, data_op, resource, and channel modules.
    export HCOMM_DEBUG_CONFIG="task,data_op,resource,channel" 
    ```

- Reverse configuration: Add `^` before the first module name. In this case, the run log records detailed runtime information about modules other than the configured submodules.

    ```bash
    # The run log records the runtime information of all modules except the data_op module (that is, the runtime information of the task, resource, and channel modules).
    export HCOMM_DEBUG_CONFIG="^data_op"
    # The run log records the runtime information of all modules except the task and data_op modules (that is, the runtime information of the resource and channel modules).
    export HCOMM_DEBUG_CONFIG="^task,data_op"
    ```

**Note**

- Extra spaces will nullify this environment variable. For example, in `export HCOMM_DEBUG_CONFIG="task, data_op "`, extra spaces exist before and after `data_op`, making the setting invalid.
- The TASK module takes effect when it is enabled in either HCOMM_DEBUG_CONFIG or HCCL_DEBUG_CONFIG. For details, see [HCCL_DEBUG_CONFIG](./HCCL_DEBUG_CONFIG.md).

**Recommendation**: When the TASK module log prints communication operator call information, set `HCCL_ENTRY_LOG_ENABLE=1` to print the call logs of communication operators in real time, so that TASK logs of different operator intervals can be distinguished. For details, see [HCCL_ENTRY_LOG_ENABLE](./HCCL_ENTRY_LOG_ENABLE.md).

## Configuration Example

```bash
export HCOMM_DEBUG_CONFIG="TASK,DATA_OP,RESOURCE,CHANNEL" 
```

## Constraints

None

## Applicable Products

<!-- npu="950" id3 -->
- Ascend 950PR&950DT products: Supported
<!-- end id3 -->
<!-- npu="A3" id1 -->
- Atlas A3 products: Not supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2 products: Not supported
<!-- end id2 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
