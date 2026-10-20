# Troubleshooting Approach

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:52:01.068Z pushedAt=2026-10-13T08:53:43.692Z -->

HCCL verifies the rank table file or the rank table information collected through negotiation. If the verification fails, HCCL reports an error and exits. Locate the fault based on the actual error message.

Possible causes include: rank table file verification failure, content mismatch with the hardware configuration, inconsistent TLS configuration, or duplicate superDeviceId.

The following sections present common cases of cluster information verification failure errors. If no matching case is found, locate and troubleshoot based on the actual error message.
