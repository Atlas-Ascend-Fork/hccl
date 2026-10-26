# HCCL_ALG_TUNER_PLUGIN

## Function

Specifies the dynamic library of the HCCL Tuner plugin. This plugin can adjust the cost values of candidate algorithms based on the communicator, operator type, data type, and data volume for the algorithm selector to select an algorithm.

- The value is the path of the dynamic library of the plugin. An absolute path is recommended.
- If this environment variable is not set, left empty, or set to `none`, the plugin is not loaded and the original algorithm selection logic applies.
- The dynamic library must export the `hcclTunerPlugin_v1` function table, and the `init` and `getCollInfo` function pointers cannot be empty.
- HCCL reads this variable when it attempts to load the plugin for the first time in a process. Configure this variable before starting the service process. If the first loading fails, the process does not retry loading.
- If the dynamic library cannot be loaded or the necessary symbols fail to be validated, HCCL prints a warning message and adopts the original algorithm selection logic.

For details about how to develop and use the plugin (including how to read the plugin configuration file and the JSON configuration format requirements), see [Tuner Plugin Example](../../../../examples/07_tuner_plugin/README.md).

> [!NOTE]
> This is an incubating feature, providing experimental capabilities and not open by default. API compatibility is not guaranteed before this feature becomes stable.

## Example

```bash
export HCCL_ALG_TUNER_PLUGIN=/path/to/hccl_tuner_example.so
```

To disable the plugin, run the following command before starting the service process:

```bash
export HCCL_ALG_TUNER_PLUGIN=none
```

## Constraints

The plugin runs in the service process. You are advised to load only trusted dynamic libraries that are compatible with the current HCCL plugin API.

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
