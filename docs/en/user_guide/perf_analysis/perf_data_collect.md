# Profile Data Collection

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T04:09:18.879Z pushedAt=2026-10-15T08:46:37.284Z -->

Collective communication is a global collaborative behavior within a communicator. It is often difficult to analyze the performance issues of collective communication using the profile data of only one rank. Therefore, you need to collect the profile data of all ranks to accurately identify the performance bottleneck of collective communication. Currently, profile data can be collected in the following two ways:

- Method 1: Refer to *[Performance Tuning Tool User Guide](https://hiascend.com/document/redirect/CannCommunityToolProfiling)* to collect profile data.
- Method 2: Refer to [*HCCL Performance Tester User Guide*](https://gitcode.com/cann/oam-tools/blob/9.2.0/docs/zh/hccl_test/README.md) and use HCCL Test to collect profile data and test the performance.

  Follow the steps below to run HCCL Test for collecting profile data:

    ```bash
    # 1 indicates that profiling is enabled, and 0 indicates that profiling is disabled. The default value is 0. If enabled, profile data is collected when HCCL Test is executed.
    export HCCL_TEST_PROFILING=1
    # Specify the path for storing profile data. The default value is /var/log/npu/profiling.
    export HCCL_TEST_PROFILING_PATH=/home/profiling
    ```

    If HCCL_TEST_PROFILING is enabled, profile data is generated in the directory specified by `HCCL_TEST_PROFILING_PATH` after the HCCL Test tool completes execution. For profile data parsing, see the "Using the msprof Command to Parse, Query, and Export the Profile Data" section in *[Performance Tuning Tool User Guide](https://hiascend.com/document/redirect/CannCommunityToolProfiling)*.
