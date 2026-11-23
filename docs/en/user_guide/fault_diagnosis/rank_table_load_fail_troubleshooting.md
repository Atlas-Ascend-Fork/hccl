# Troubleshooting Approach

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-09T03:28:10.473Z pushedAt=2026-11-11T09:05:14.990Z nl=2 -->

The method of creating a communicator based on a rank table requires loading the rank table file. If the file path does not exist, permissions are insufficient, or the file format or configuration is incorrect, HCCL fails to load and returns an error.

The following sections describe some common cases of rank table file validation failure errors. If no matching case is found, troubleshooting can be performed based on the actual error message.
