# HCCL_WHITELIST_FILE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-13T04:03:37.868Z pushedAt=2026-10-15T08:46:37.242Z -->

## Function

When communication trustlist verification is enabled through HCCL_WHITELIST_DISABLE, use this environment variable to specify the path to the HCCL communication trustlist configuration file. Only IP addresses in the communication trustlist are allowed to perform collective communication.

The format of the HCCL communication trustlist configuration file is:

```text
{ "host_ip": ["ip1", "ip2"], "device_ip": ["ip1", "ip2"] } 
```

Where:

- `device_ip` is a reserved field and is not supported in the current version.
- The IP address format is dotted decimal notation.

> [!NOTE]
> The trustlist IP must be specified as a valid IP used for cluster communication.

## Configuration Example

```bash
export HCCL_WHITELIST_FILE=/home/test/whitelist
```

## Constraints

None

## Applicable Products

<!-- npu="950" id3 -->
- Ascend 950PR&950DT products: Supported
<!-- end id3 -->
<!-- npu="A3" id4 -->
- Atlas A3 products: Supported
<!-- end id4 -->
<!-- npu="910b" id5 -->
- Atlas A2 products: Supported
<!-- end id5 -->
<!-- npu="910" id1 -->
- Atlas training products: Supported
<!-- end id1 -->
<!-- npu="310p" id2 -->
- Atlas inference products: Supported
<!-- end id2 -->
