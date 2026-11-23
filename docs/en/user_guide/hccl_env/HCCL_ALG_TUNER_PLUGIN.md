# HCCL_ALG_TUNER_PLUGIN

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-06T09:45:53.317Z pushedAt=2026-11-16T06:28:10.133Z nl=2 -->

## Function

Specifies the HCCL Tuner dynamic library. The HCCL Tuner plugin can adjust the cost values of candidate algorithms based on the communicator, operator type, data type, and data volume, for the algorithm selector to select an algorithm.

- The value is the path to the plugin dynamic library. An absolute path is recommended.
- If the variable is not set, is empty, or is set to `none`, the plugin is not loaded and the original algorithm selection logic is used.
- The dynamic library must export the `hcclTunerPlugin_v1` function table, and the `init` and `getCollInfo` function pointers must not be null.
- HCCL reads this variable at its first attempt to load the plugin in a process. Configure this variable before starting a service process. After the first load fails, the process does not retry loading.
- If the dynamic library cannot be loaded or required symbol verification fails, HCCL prints a warning and uses the original algorithm selection logic.

For plugin development and instructions (including how to read the plugin configuration file and the JSON configuration format requirements), see [Tuner Plugin Example](../../../../examples/07_tuner_plugin/README.md).

> [!NOTE]
> Incubating feature: This is an experimental capability that is not enabled by default. API compatibility is not guaranteed before it becomes stable.

## Configuration Example

```bash
export HCCL_ALG_TUNER_PLUGIN=/path/to/hccl_tuner_example.so
```

To disable the plugin, run the following command before starting the service process:

```bash
export HCCL_ALG_TUNER_PLUGIN=none
```

## Constraints

The plugin runs in the service process. Load only dynamic libraries from trusted sources that are compatible with the current HCCL plugin API.

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
