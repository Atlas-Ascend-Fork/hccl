# Environment Preparation

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T10:56:46.482Z pushedAt=2026-10-14T07:19:25.724Z -->

## Installing Driver Firmware and CANN Software Package

The use of the HCCL collective communication library and the development of communication operators depend on the driver firmware and CANN software package. For detailed installation steps, see *[CANN Software Installation](https://hiascend.com/document/redirect/CannCommunityInstSoftware)*.

> [!NOTE]Note
> If only app development and compilation are performed without involving execution, the driver firmware package is not required.

## Setting Environment Variables

Before compiling and running programs, the CANN software environment variables need to be set.

```bash
source /usr/local/Ascend/cann/set_env.sh
```

`/usr/local/Ascend` is the default installation path of CANN software for the root user. If the software is installed by a regular user or to a custom path, replace it accordingly.
