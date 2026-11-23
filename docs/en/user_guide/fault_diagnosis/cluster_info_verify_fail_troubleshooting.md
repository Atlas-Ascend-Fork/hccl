# Troubleshooting Approach

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-09T03:25:44.986Z pushedAt=2026-11-11T02:58:24.525Z nl=2 -->

HCCL verifies the rank table file or the rank table information collected through negotiation. If the verification fails, HCCL reports an error and exits. Locate the fault based on the actual error message.

Possible causes include: rank table file verification failure, content mismatch with the hardware configuration, inconsistent TLS configuration, or duplicate superDeviceId.

The following sections present common cases of cluster information verification failure errors. If no matching case is found, locate and troubleshoot based on the actual error message.
