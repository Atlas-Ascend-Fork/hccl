# Troubleshooting Approach

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:52:21.372Z pushedAt=2026-10-13T09:32:43.517Z -->

The appearance of the EI0001 fault code in service logs indicates an HCCL environment variable configuration error. Generally, the ERROR MESSAGE in the printed logs and the CANN logs display the name of the environment variable with the configuration error, the cause of the error, and the valid configuration range. For any questions, refer to [Environment Variable Reference](../hccl_env/README.md).

For Ascend 950PR&950DT products, you can search for `[InitGroupStage][EnvConfig]` in the plog, and then locate the issue based on the environment variable name, configured value, and error cause. For example, when `HCCL_ENTRY_LOG_ENABLE=-1` is set, the log body contains the following information, indicating that this environment variable supports only 0 or 1:

```text
[InitGroupStage][EnvConfig] ... Env config "HCCL_ENTRY_LOG_ENABLE" value "-1" is invalid. ... Should be 0 or 1
```

The log body may differ slightly across versions. If the complete example above is not found, you can search by combining `[InitGroupStage][EnvConfig]` with the environment variable name, and correct the configuration based on the actual value and the allowed range in the error message.
