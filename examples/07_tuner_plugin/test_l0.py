#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L0 纯函数单测（本地 Python3，无 NPU）。

覆盖 perf_test.py / optimize_config.py / tuner_common.py 的核心纯函数：
算法串解析、命令构造、MPI 环境探测、stdout 解析、CSV 结构、参数白名单、
preflight 自检、去重 / 分组 / 选优 / 区间合并 / 补缝 / 白名单校验 / conf 结构。

运行：python3 test_l0.py  （或 python3 -m unittest test_l0 -v）
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import perf_test  # noqa: E402
import optimize_config  # noqa: E402
import tuner_common  # noqa: E402


class TunerCommonTest(unittest.TestCase):
    """C 一致性：公共表单一来源，两脚本引用同一对象；plugin 白名单 ⊆ 采集权威域。"""

    def test_two_scripts_share_common_tables(self):
        # 同一来源（非副本）
        self.assertIs(perf_test.CSV_FIELDS, tuner_common.CSV_FIELDS)
        self.assertIs(perf_test.OP_TYPES, tuner_common.OP_TYPES)
        self.assertIs(perf_test.DEVICE_ENGINES, tuner_common.DEVICE_ENGINES)
        self.assertIs(perf_test.ENGINE_ALL, tuner_common.ENGINE_ALL)
        self.assertIs(perf_test.EXECUTOR_TYPES, tuner_common.EXECUTOR_TYPES)
        self.assertIs(perf_test.TEMPLATE_TYPES, tuner_common.TEMPLATE_TYPES)
        self.assertIs(optimize_config.CSV_FIELDS, tuner_common.CSV_FIELDS)
        self.assertIs(optimize_config.PLUGIN_ENGINES, tuner_common.PLUGIN_ENGINES)
        self.assertIs(optimize_config.PLUGIN_EXECUTORS, tuner_common.PLUGIN_EXECUTORS)
        self.assertIs(optimize_config.PLUGIN_OP_TYPES, tuner_common.PLUGIN_OP_TYPES)
        self.assertIs(optimize_config.PLUGIN_TEMPLATES, tuner_common.PLUGIN_TEMPLATES)
        self.assertIs(optimize_config.DEVICE_ENGINES, tuner_common.DEVICE_ENGINES)

    def test_csv_fields_structure(self):
        self.assertEqual(len(tuner_common.CSV_FIELDS), 17)
        # 拓扑 5 字段 = CSV_FIELDS 的 ranks.* 后段
        self.assertEqual(tuner_common.CSV_FIELDS[-9:-4], tuner_common.RANKS_FIELDS)

    def test_plugin_whitelist_subset_of_collect_domain(self):
        # plugin 白名单是采集权威域的子集
        self.assertTrue(tuner_common.PLUGIN_EXECUTORS <= set(tuner_common.EXECUTOR_TYPES))
        self.assertTrue(tuner_common.PLUGIN_TEMPLATES <= set(tuner_common.TEMPLATE_TYPES))
        self.assertEqual(tuner_common.PLUGIN_OP_TYPES, set(tuner_common.OP_TYPES))
        self.assertEqual(tuner_common.PLUGIN_ENGINES, set(tuner_common.ENGINE_ALL))
        # 具体差集：executor 差 strictordered；template 白名单 = 权威域全量
        # （plugin.cpp 不做 template 枚举校验，13 种均对应已注册算法）
        self.assertEqual(set(tuner_common.EXECUTOR_TYPES) - tuner_common.PLUGIN_EXECUTORS,
                         {"strictordered"})
        self.assertEqual(tuner_common.PLUGIN_TEMPLATES, set(tuner_common.TEMPLATE_TYPES))
        # 权威枚举规模
        self.assertEqual(len(tuner_common.EXECUTOR_TYPES), 6)
        self.assertEqual(len(tuner_common.TEMPLATE_TYPES), 13)
        self.assertEqual(len(tuner_common.PLUGIN_EXECUTORS), 5)
        self.assertEqual(len(tuner_common.PLUGIN_TEMPLATES), 13)
        self.assertEqual(len(tuner_common.ENGINE_ALL), 5)

    def test_op_types_derived_from_pascal_map(self):
        # OP_TYPES 由 _PASCAL_TO_OP 派生，锁定顺序（探测轮次按此序下发）
        self.assertEqual(
            tuple(tuner_common.OP_TYPES),
            ("allreduce", "allgather", "broadcast", "reduce", "reduce_scatter", "scatter", "alltoall", "alltoallv"))

    def test_whitelist_import_guard(self):
        # 白名单塞入词表外名字时，import 期立即报错（防拼写漂移）
        with open(tuner_common.__file__, encoding="utf-8") as handle:
            source = handle.read()
        marker = '    "mesh", "mesh2die", "meshoneshot", "meshtwoshot", "meshconcur", "meshmultilink",'
        self.assertIn(marker, source)
        patched = source.replace(marker, '    "mesh", "mesh2die", "badtpl",', 1)
        namespace = {"__name__": "tuner_common_patched"}
        with self.assertRaises(ValueError) as ctx:
            exec(compile(patched, "<patched-tuner-common>", "exec"), namespace)
        self.assertIn("TEMPLATE_TYPES", str(ctx.exception))
        self.assertIn("badtpl", str(ctx.exception))

    def test_timestamp_helpers(self):
        # now_timestamp 产出 17 位合法紧凑格式，且能被 parse_timestamp 原样解析
        ts = tuner_common.now_timestamp()
        self.assertEqual(len(ts), 17)
        self.assertRegex(ts, r"20\d{15}")
        self.assertEqual(tuner_common.parse_timestamp(ts), int(ts))
        # 非法输入（旧 epoch 格式/乱改/缺列）一律 -1，去重时视作最旧
        self.assertEqual(tuner_common.parse_timestamp("1000000000000"), -1)
        self.assertEqual(tuner_common.parse_timestamp("99999999999999999"), -1)
        self.assertEqual(tuner_common.parse_timestamp("2026-10-12 09:00:00"), -1)
        self.assertEqual(tuner_common.parse_timestamp(None), -1)
        self.assertEqual(tuner_common.parse_timestamp(""), -1)


# kwargs 关键字名不能带括号/斜杠（如 alg_bandwidth(GB/s)），调用点用旧简名，
# 由 make_row 归一成 CSV 真实列名（带单位）。
ROW_KEY_ALIAS = {
    "alg_bandwidth": "alg_bandwidth(GB/s)",
    "alg_latency": "alg_latency(us)",
}


def make_row(**overrides):
    """构造一条标准采集行（默认 allreduce / fp32 / aicpu / sole{mesh} / 8 rank 单机拓扑）。"""
    row = {
        "op_type": "allreduce", "size_bytes": "8192", "data_type": "fp32", "reduce_type": "sum",
        "engine": "aicpu", "algorithm.executor_type": "sole", "algorithm.template_type": "mesh",
        "HCCL_BUFFSIZE": "2048", "ranks.ranks": "8", "ranks.npus_per_server": "8",
        "ranks.servers": "1", "ranks.pods": "1", "ranks.super_pods": "1",
        "check_result": "success", "alg_bandwidth(GB/s)": "10.0", "alg_latency(us)": "100.0",
        "timestamp": "20261012090000000",
    }
    row.update((ROW_KEY_ALIAS.get(key, key), value) for key, value in overrides.items())
    return row


class ProfParseBytesTest(unittest.TestCase):
    """P01 parse_bytes 后缀解析。"""

    def test_parse_bytes(self):
        self.assertEqual(perf_test.parse_bytes("1K"), 1024)
        self.assertEqual(perf_test.parse_bytes("4M"), 4194304)
        self.assertEqual(perf_test.parse_bytes("2G"), 2147483648)
        self.assertEqual(perf_test.parse_bytes("512"), 512)
        self.assertEqual(perf_test.parse_bytes("1.5K"), 1536)
        self.assertEqual(perf_test.parse_bytes("1k"), 1024)
        with self.assertRaises(ValueError):
            perf_test.parse_bytes("abc")
        with self.assertRaises(ValueError):
            perf_test.parse_bytes("1T")


class ProfParseAlgosTest(unittest.TestCase):
    """parse_algos 算法串解析。"""

    def test_full_form(self):
        self.assertEqual(tuner_common.parse_algos("sole{mesh}"), [("sole", ["mesh"])])

    def test_shorthand_defaults_sole(self):
        self.assertEqual(tuner_common.parse_algos("mesh"), [("sole", ["mesh"])])

    def test_multi_level_comma_sequence(self):
        self.assertEqual(tuner_common.parse_algos("parallel{mesh,nhr}"), [("parallel", ["mesh", "nhr"])])

    def test_multi_token_and_empty_segment(self):
        algos = tuner_common.parse_algos("sequence{mesh,nhr,nhr};sole{mesh2die};;")
        self.assertEqual(algos, [("sequence", ["mesh", "nhr", "nhr"]), ("sole", ["mesh2die"])])

    def test_empty_string(self):
        self.assertEqual(tuner_common.parse_algos(""), [])

    def test_invalid_executor_or_template(self):
        with self.assertRaises(ValueError):
            tuner_common.parse_algos("badexec{mesh}")
        with self.assertRaises(ValueError):
            tuner_common.parse_algos("sole{badtpl}")

    def test_level_syntax_rejected(self):
        with self.assertRaises(ValueError):
            tuner_common.parse_algos("sequence{level1=nhr,level0=mesh}")


class ProfAlgoToEnvTest(unittest.TestCase):
    """P03 algo_to_env 往返。"""

    def test_roundtrip(self):
        self.assertEqual(tuner_common.algo_to_env("sole", ["mesh"]), "sole{mesh}")
        self.assertEqual(tuner_common.algo_to_env("parallel", ["mesh", "nhr"]), "parallel{mesh,nhr}")
        # 往返：parse_algos(algo_to_env(x)) == x
        for algos in [("sole", ["mesh"]), ("parallel", ["mesh", "nhr"])]:
            parsed = tuner_common.parse_algos(tuner_common.algo_to_env(*algos))
            self.assertEqual(parsed, [algos])


class ProfBuildAlgosTest(unittest.TestCase):
    """P04 build_algos 双入口。"""

    def test_algos_arg_passthrough(self):
        self.assertEqual(perf_test.build_algos("sole{mesh};parallel{mesh,nhr}", None, None),
                         [("sole", ["mesh"]), ("parallel", ["mesh", "nhr"])])

    def test_executors_templates_cartesian(self):
        algos = perf_test.build_algos(None, "sole,parallel", "mesh,nhr")
        self.assertEqual(algos, [("sole", ["mesh"]), ("sole", ["nhr"]),
                                 ("parallel", ["mesh"]), ("parallel", ["nhr"])])

    def test_default_full_enum(self):
        # 三参数均未指定 → None 哨兵（采集计划直接展开注册表全量组合，含多级）
        self.assertIsNone(perf_test.build_algos(None, None, None))


class ProfEstimateTopoLevelsRemovedTest(unittest.TestCase):
    """回归：静态层数估算与黑名单已由 probe 名单取代，确认不再存在。"""

    def test_static_tables_removed(self):
        for gone in ("OP_VALID_ALGOS", "_TOPO_LEVEL_CONSTRAINTS", "ALGO_BLACKLIST_KEYWORDS",
                     "is_blacklisted", "estimate_topo_levels"):
            self.assertFalse(hasattr(tuner_common, gone), gone)
            self.assertFalse(hasattr(perf_test, gone), gone)


class ProfBuildMpirunCmdTest(unittest.TestCase):
    """P05 build_mpirun_cmd 命令构造与拓扑派生。"""

    def test_cmd_and_topo(self):
        with tempfile.TemporaryDirectory() as tmp:
            hostfile = os.path.join(tmp, "hosts")
            with open(hostfile, "w", encoding="utf-8") as handle:
                handle.write("10.10.130.22:8\n10.10.130.21:8\n")
            cmd = perf_test.build_mpirun_cmd(16, hostfile, ["./bin/all_reduce_test", "-p", "8"],
                                             env_keys=["HCCL_ALGO"], flavor="mpich")
            self.assertIn("mpirun", cmd[0])
            self.assertIn("-n", cmd)
            self.assertEqual(cmd[cmd.index("-n") + 1], "16")
            self.assertIn("-f", cmd)  # MPICH hostfile 旗标
            self.assertEqual(cmd[cmd.index("-f") + 1], hostfile)
            self.assertIn("./bin/all_reduce_test", cmd)
            topo = perf_test.derive_topo(16, 8, hostfile, 1, 1)
            self.assertEqual(topo["ranks.ranks"], 16)
            self.assertEqual(topo["ranks.npus_per_server"], 8)
            self.assertEqual(topo["ranks.servers"], 2)

    def test_openmpi_hostfile_and_env(self):
        # 官方清单有值才透传、env_keys 追加去重、root 豁免与系统限制放开
        env = {"ASCEND_HOME_PATH": "/usr/local/Ascend", "LD_LIBRARY_PATH": "/usr/local/Ascend/lib64",
               "HCCL_SOCKET_IFNAME": "eth0", "HCCL_ALGO": ""}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(perf_test, "derive_mpi_prefix",
                                   return_value="/usr/local/openmpi"):
                with mock.patch.object(os, "getuid", return_value=0, create=True):
                    cmd = perf_test.build_mpirun_cmd(8, "hostfile", ["./bin/reduce_test"],
                                                     env_keys=["HCCL_ALGO", "LD_LIBRARY_PATH"],
                                                     flavor="openmpi")
        self.assertEqual(cmd[cmd.index("--prefix") + 1], "/usr/local/openmpi")
        keys = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-x"]
        # HCCL_ALGO 值为空被剔除，LD_LIBRARY_PATH 与清单去重
        self.assertEqual(keys, ["ASCEND_HOME_PATH", "LD_LIBRARY_PATH", "HCCL_SOCKET_IFNAME"])
        self.assertIn("--allow-run-as-root", cmd)
        self.assertEqual(cmd[cmd.index("--mca") + 1], "opal_set_max_sys_limits")

    def test_openmpi_non_root_no_flag(self):
        # 非 root 不加 --allow-run-as-root
        with mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/lib"}, clear=True):
            with mock.patch.object(perf_test, "derive_mpi_prefix", return_value=None):
                with mock.patch.object(os, "getuid", return_value=599, create=True):
                    cmd = perf_test.build_mpirun_cmd(8, None, ["./bin/all_reduce_test"],
                                                     flavor="openmpi")
        self.assertNotIn("--allow-run-as-root", cmd)
        self.assertEqual([cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-x"],
                         ["LD_LIBRARY_PATH"])

    def test_derive_mpi_prefix(self):
        # 真实路径剥 <prefix>/bin/mpirun*；默认安装路径与找不到 mpirun 返回 None
        cases = [
            ("/opt/openmpi/bin/mpirun", "/opt/openmpi"),
            ("/usr/local/openmpi/bin/mpirun", "/usr/local/openmpi"),
            ("/usr/lib64/openmpi/bin/mpirun.openmpi", "/usr/lib64/openmpi"),
            ("/usr/bin/mpirun", None),
            ("/usr/local/bin/mpirun", None),
        ]
        for path, expect in cases:
            with mock.patch.object(perf_test.shutil, "which", return_value=path):
                with mock.patch.object(perf_test.os.path, "realpath", return_value=path):
                    self.assertEqual(perf_test.derive_mpi_prefix(), expect)
        with mock.patch.object(perf_test.shutil, "which", return_value=None):
            self.assertIsNone(perf_test.derive_mpi_prefix())

    def test_single_machine_no_hostfile(self):
        cmd = perf_test.build_mpirun_cmd(8, None, ["./bin/all_reduce_test"], flavor="mpich")
        self.assertNotIn("-f", cmd)
        self.assertNotIn("--prefix", cmd)  # 自动补齐参数仅 OpenMPI 分支
        self.assertNotIn("--mca", cmd)
        self.assertEqual(perf_test.derive_topo(8, 8, None, 1, 1)["ranks.servers"], 1)

    def test_npus_derived_when_omitted(self):
        # --npus 缺省：单机推导 np_total，双机按 hostfile 行数均分
        self.assertEqual(perf_test.derive_topo(8, None, None, 0, 0)["ranks.npus_per_server"], 8)
        with tempfile.TemporaryDirectory() as tmp:
            hostfile = os.path.join(tmp, "hostfile")
            with open(hostfile, "w", encoding="utf-8") as handle:
                handle.write("10.0.0.1:8\n10.0.0.2:8\n")
            topo = perf_test.derive_topo(16, None, hostfile, 0, 0)
        self.assertEqual(topo["ranks.npus_per_server"], 8)
        self.assertEqual(topo["ranks.servers"], 2)

    def test_npus_derivation_indivisible_rejected(self):
        # 总卡数无法均分到服务器 → 要求显式指定 --npus
        with tempfile.TemporaryDirectory() as tmp:
            hostfile = os.path.join(tmp, "hostfile")
            with open(hostfile, "w", encoding="utf-8") as handle:
                handle.write("10.0.0.1:8\n10.0.0.2:8\n10.0.0.3:8\n")
            with self.assertRaises(ValueError):
                perf_test.derive_topo(16, None, hostfile, 0, 0)

    def test_heterogeneous_hostfile_rejected(self):
        # 各机 NPU 数不一致（异构）→ 启动即报错，MPICH 与 OpenMPI 语法同校验
        with tempfile.TemporaryDirectory() as tmp:
            hostfile = os.path.join(tmp, "hostfile")
            with open(hostfile, "w", encoding="utf-8") as handle:
                handle.write("# comment\n10.0.0.1:8\n\n10.0.0.2:4\n")
            with self.assertRaises(ValueError):
                perf_test.derive_topo(12, None, hostfile, 0, 0)
            with open(hostfile, "w", encoding="utf-8") as handle:
                handle.write("10.0.0.1 slots=8\n10.0.0.2 slots=4\n")
            with self.assertRaises(ValueError):
                perf_test.derive_topo(12, None, hostfile, 0, 0)
            # 纯 IP 行 / 裸 IPv6 地址无 NPU 数可校验 → 放行不误判
            with open(hostfile, "w", encoding="utf-8") as handle:
                handle.write("10.0.0.1\n10.0.0.2  # no count\nfe80::1\n")
            self.assertEqual(perf_test.derive_topo(16, 8, hostfile, 0, 0)["ranks.servers"], 3)
            # MPICH 均质 + 行内注释 → 正常通过
            with open(hostfile, "w", encoding="utf-8") as handle:
                handle.write("10.0.0.1:8 # node1\n10.0.0.2:8\n")
            topo = perf_test.derive_topo(16, None, hostfile, 0, 0)
        self.assertEqual(topo["ranks.npus_per_server"], 8)


class ProfDetectMpiEnvTest(unittest.TestCase):
    """detect_mpi_env / check_mpi_env MPI 环境探测。"""

    def test_no_mpi_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(perf_test.detect_mpi_env(), (0, 1))

    def test_ompi_env(self):
        env = {"OMPI_COMM_WORLD_RANK": "2", "OMPI_COMM_WORLD_SIZE": "8"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(perf_test.detect_mpi_env(), (2, 8))
            self.assertEqual(perf_test.detect_mpi_flavor(), "openmpi")

    def test_flavor_from_mpi_path(self):
        # 发行版 alternatives / 安装目录 / 源码编译符号链接均暴露实现名
        cases = [
            ("/usr/bin/mpirun.openmpi", "openmpi"),
            ("/usr/lib64/openmpi/bin/orterun", "openmpi"),
            ("/usr/local/bin/orterun", "openmpi"),
            ("/usr/bin/mpiexec.hydra", "mpich"),
            ("/usr/bin/mpirun.mpich", "mpich"),
            ("/usr/bin/mpirun", "mpich"),
        ]
        for path, expect in cases:
            self.assertEqual(perf_test.flavor_from_mpi_path(path), expect)

    def test_flavor_from_mpirun_lookup(self):
        # OMPI 变量缺失时读 mpirun 真实路径判定；找不到 mpirun 按 mpich 兜底
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(perf_test.shutil, "which",
                                   return_value="/usr/lib64/openmpi/bin/mpirun"):
                self.assertEqual(perf_test.detect_mpi_flavor(), "openmpi")
            with mock.patch.object(perf_test.shutil, "which",
                                   return_value="/usr/bin/mpiexec.hydra"):
                self.assertEqual(perf_test.detect_mpi_flavor(), "mpich")
            with mock.patch.object(perf_test.shutil, "which", return_value=None):
                self.assertEqual(perf_test.detect_mpi_flavor(), "mpich")

    def test_default_flavor_mpich(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(perf_test.shutil, "which", return_value=None):
                self.assertEqual(perf_test.detect_mpi_flavor(), "mpich")

    def test_check_mpi_env_outside_session(self):
        # 会话外：返回 flavor，不报错
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(perf_test.shutil, "which", return_value=None):
                self.assertEqual(perf_test.check_mpi_env(), "mpich")
        env = {"OMPI_COMM_WORLD_RANK": "0", "OMPI_COMM_WORLD_SIZE": "1"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(perf_test.check_mpi_env(), "openmpi")

    def test_check_mpi_env_nested_rejected(self):
        # 已在 MPI 会话内（size>1）：拦截嵌套运行
        env = {"OMPI_COMM_WORLD_RANK": "2", "OMPI_COMM_WORLD_SIZE": "8"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(perf_test.PreflightError):
                perf_test.check_mpi_env()
        env = {"PMI_RANK": "3", "PMI_SIZE": "16"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(perf_test.PreflightError):
                perf_test.check_mpi_env()


class ProfParseStdoutTest(unittest.TestCase):
    """P07 parse_stdout 解析 hccl_test 表格。"""

    STDOUT = (
        "the minbytes is 8192, maxbytes is 67108864, iters is 20, warmup_iters is 5\n"
        "data_size(Bytes): |   avg_time(us): | alg_bandwidth(GB/s): | check_result:\n"
        "8192              |     764.55    |       0.00998        | success\n"
        "16384             |     858.80    |       0.01777        | success\n"
        "\n"
        "32768             |     901.10    |       0.03387        | failed\n"
        "some warning line without pipes\n"
        "65536             |     900.00    |       0.06782        | NULL\n"
    )

    def test_parse_rows(self):
        rows = perf_test.parse_stdout(self.STDOUT)
        self.assertEqual(len(rows), 4)
        first = rows[0]
        self.assertEqual(first["size_bytes"], 8192)
        self.assertEqual(first["alg_latency(us)"], 764.55)
        self.assertEqual(first["alg_bandwidth(GB/s)"], 0.00998)
        self.assertEqual(first["check_result"], "success")
        self.assertEqual(rows[2]["check_result"], "failed")
        self.assertEqual(rows[3]["check_result"], "NULL")
        self.assertEqual(rows[3]["size_bytes"], 65536)

    def test_column_order_tolerant(self):
        # 列序变化（check_result 提前）仍按列名匹配
        stdout = (
            "check_result: | data_size(Bytes): | avg_time(us): | alg_bandwidth(GB/s):\n"
            "success      | 1024             | 10.5          | 0.5\n"
        )
        rows = perf_test.parse_stdout(stdout)
        self.assertEqual(rows[0], {"size_bytes": 1024, "alg_latency(us)": 10.5,
                                   "alg_bandwidth(GB/s)": 0.5, "check_result": "success"})

    def test_real_binary_header_aveg_time(self):
        # 真实二进制表头是 "aveg_time(us):"（源码拼写笔误，非文档的 avg_time）
        stdout = (
            "the minbytes is 8192, maxbytes is 67108864, iters is 20, warmup_iters is 5\n"
            "data_size(Bytes): |   aveg_time(us): | alg_bandwidth(GB/s): | check_result:\n"
            "8192              |     764.55    |       0.00998        | success\n"
        )
        rows = perf_test.parse_stdout(stdout)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["alg_latency(us)"], 764.55)


class ProfParseHeaderInfoTest(unittest.TestCase):
    """parse_header_info：校验 -w/-n 是否下发生效（解析 hccl_test 首行回显）。"""

    def test_first_line_echo(self):
        stdout = ("the minbytes is 8192, maxbytes is 67108864, iters is 20, warmup_iters is 10\n"
                  "data_size(Bytes): | avg_time(us): | alg_bandwidth(GB/s): | check_result:\n")
        self.assertEqual(perf_test.parse_header_info(stdout), (20, 10))

    def test_real_binary_aveg_echo(self):
        # 真实回显顺序：iters 在前，warmup_iters 在后
        stdout = "the minbytes is 8, maxbytes is 67108864, iters is 20, warmup_iters is 5\n"
        self.assertEqual(perf_test.parse_header_info(stdout), (20, 5))

    def test_no_echo_returns_none(self):
        self.assertIsNone(perf_test.parse_header_info("no summary line here"))
        self.assertIsNone(perf_test.parse_header_info(""))

    def test_searches_all_lines(self):
        # 回显不在首行（前面有 mpirun banner）也能找到
        stdout = "mpirun banner...\n\nthe minbytes is 8, iters is 7, warmup_iters is 3\ntable..."
        self.assertEqual(perf_test.parse_header_info(stdout), (7, 3))


class ProfParseRunOutputTest(unittest.TestCase):
    """parse_run_output：stdout 优先、空时回退 stderr（表格可能落到 stderr）。"""
    def test_stdout_preferred(self):
        stdout = ("data_size(Bytes): | avg_time(us): | alg_bandwidth(GB/s): | check_result:\n"
                  "8192              | 10.5          | 0.5                | success\n")
        stderr = "some mpirun noise"
        rows = perf_test.parse_run_output(stdout, stderr)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["alg_latency(us)"], 10.5)

    def test_fallback_to_stderr_when_stdout_empty(self):
        stdout = "warmup and no table here"
        stderr = ("data_size(Bytes): | avg_time(us): | alg_bandwidth(GB/s): | check_result:\n"
                  "8192              | 10.5          | 0.5                | success\n")
        rows = perf_test.parse_run_output(stdout, stderr)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["size_bytes"], 8192)

    def test_both_empty_no_rows(self):
        self.assertEqual(perf_test.parse_run_output("", "noise only"), [])


class ProfDumpRawOutputTest(unittest.TestCase):
    """dump_raw_output：解析失败轮取证 dump。"""

    def test_dump_appends_with_tag_and_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "out.csv")
            path1 = perf_test.dump_raw_output(csv_path, "op=allreduce engine=aicpu", "raw-out", "raw-err")
            path2 = perf_test.dump_raw_output(csv_path, "op=broadcast engine=aicpu", "out2", "err2")
            self.assertEqual(path1, path2)
            self.assertEqual(path1, csv_path + ".raw")
            with open(path1, encoding="utf-8") as raw:
                content = raw.read()
            self.assertIn("===== op=allreduce engine=aicpu =====", content)
            self.assertIn("----- stdout -----\nraw-out", content)
            self.assertIn("----- stderr -----\nraw-err", content)
            self.assertIn("===== op=broadcast engine=aicpu =====", content)
            # 追加写：两轮内容共存
            self.assertEqual(content.count("====="), 4)


class ProfCsvStructureTest(unittest.TestCase):
    """P08 CSV 结构与追加写。"""

    def test_header_and_append(self):
        import csv as csv_mod
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            with perf_test.CsvWriter(path) as writer:
                writer.write_row(make_row())
                writer.write_row(make_row(size_bytes="16384"))
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv_mod.reader(handle)
                rows = list(reader)
            self.assertEqual(rows[0], list(perf_test.CSV_FIELDS))
            self.assertEqual(len(rows[0]), 17)
            self.assertIn("timestamp", rows[0])
            self.assertIn("check_result", rows[0])
            self.assertIn("reduce_type", rows[0])
            self.assertIn("HCCL_BUFFSIZE", rows[0])
            self.assertEqual(len(rows), 3)  # header + 2 行，字段数与 header 对齐
            self.assertTrue(all(len(r) == 17 for r in rows))
            # 二次 run 追加同一文件：表头不重写，旧行保留、新行续写
            with perf_test.CsvWriter(path) as writer:
                writer.write_row(make_row(size_bytes="32768"))
            with open(path, newline="", encoding="utf-8") as handle:
                rows2 = list(csv_mod.reader(handle))
            self.assertEqual(len(rows2), 4)
            self.assertEqual(rows2[0], list(perf_test.CSV_FIELDS))  # 表头仍只有一份
            self.assertEqual(rows2[1][1], "8192")   # 旧行保留
            self.assertEqual(rows2[3][1], "32768")  # 新行在尾部

    def test_append_rejects_mismatched_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                handle.write("op_type,size_bytes\n")  # 异构表头
            with self.assertRaises(ValueError):
                perf_test.CsvWriter(path)
            # 构造失败后不得追加任何数据
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "op_type,size_bytes\n")

    def test_append_completes_missing_trailing_newline(self):
        # 末字节非换行（上一轮被强杀等）→ 追加前补 \n，新行不与残行粘连
        import csv as csv_mod
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            with perf_test.CsvWriter(path) as writer:
                writer.write_row(make_row())
            with open(path, "r+b") as handle:  # 砍掉末尾换行，模拟半行落盘
                handle.truncate(os.path.getsize(path) - 1)
            with perf_test.CsvWriter(path) as writer:
                writer.write_row(make_row(size_bytes="32768"))
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv_mod.reader(handle))
            # header + 残缺旧行 + 新行；新行独立成行不粘连（残行少一列，被 reader 原样收）
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0], list(perf_test.CSV_FIELDS))
            self.assertTrue(rows[2][1] == "32768" and len(rows[2]) == 17)


class ProfValidateArgsTest(unittest.TestCase):
    """P09 参数白名单校验。"""

    def test_v_series_op_rejected(self):
        with self.assertRaises(ValueError):
            perf_test.validate_args(self._args(op_types="allreduce,allgatherv"))

    def test_fp8_collectable(self):
        op_types, engines, data_types, _ = perf_test.validate_args(
            self._args(data_types="fp32,fp8e5m2"))
        self.assertEqual(data_types, ["fp32", "fp8e5m2"])

    def test_engines_all_expands_supported(self):
        # all 展开为全部支持引擎（probe 分引擎探测后全量放开）
        _, engines, _, _ = perf_test.validate_args(self._args(engines="all"))
        self.assertEqual(engines, list(perf_test.PROF_ENGINES))

    def test_unknown_engine_rejected(self):
        # 白名单外引擎（含误写 aiv_only）被 CLI 拦截
        for bad in ("aiv_only", "host", "aicpu,bad"):
            with self.assertRaises(ValueError):
                perf_test.validate_args(self._args(engines=bad))

    def test_engines_accepted(self):
        # 白名单内引擎单传或混合均通过（ccu 系、aiv/dpu 各验一组）
        for value, expected in (("ccums,ccusched", ["ccums", "ccusched"]),
                                ("aiv,dpu", ["aiv", "dpu"])):
            _, engines, _, _ = perf_test.validate_args(self._args(engines=value))
            self.assertEqual(engines, expected)

    def test_reduce_ops_multi_value(self):
        _, _, _, reduce_ops = perf_test.validate_args(self._args(reduce_ops="sum,prod"))
        self.assertEqual(reduce_ops, ["sum", "prod"])
        with self.assertRaises(ValueError):
            perf_test.validate_args(self._args(reduce_ops="sum,badop"))

    def _args(self, **overrides):
        base = {"op_types": "allreduce", "engines": "aicpu", "data_types": "fp32",
                "reduce_ops": "sum", "minbytes": None, "maxbytes": None,
                "stepbytes": None, "stepfactor": None, "hostfile": None}
        base.update(overrides)
        return mock.Mock(**base)


class ProfEngineMapTest(unittest.TestCase):
    """P10 engine→-a 映射。"""

    def test_mapping(self):
        self.assertEqual(perf_test.engine_to_accelerator("aicpu"), "aicpu_ts")
        self.assertEqual(perf_test.engine_to_accelerator("aiv"), "aiv_only")
        self.assertEqual(perf_test.engine_to_accelerator("ccums"), "ccu_ms")
        self.assertEqual(perf_test.engine_to_accelerator("ccusched"), "ccu_sched")
        self.assertIsNone(perf_test.engine_to_accelerator("dpu"))  # dpu 不走 -a
        with self.assertRaises(ValueError):
            perf_test.engine_to_accelerator("default")

    def test_test_args_accel_flag(self):
        profile = perf_test.TestProfile(npus=8, minbytes="1K", maxbytes="1M", stepbytes=None,
                                        stepfactor=2, warmup_iters=10, iters=20)
        args = perf_test.build_test_args(profile, "allreduce", "fp32", "sum",
                                         accel=perf_test.engine_to_accelerator("aiv"))
        self.assertEqual(args[args.index("-a") + 1], "aiv_only")
        dpu_args = perf_test.build_test_args(profile, "allreduce", "fp32", "sum",
                                             accel=perf_test.engine_to_accelerator("dpu"))
        self.assertNotIn("-a", dpu_args)
        # probe 轮按引擎分轮下发 -a（单引擎圈定候选，dpu 轮除外）；-w/-n 来自 profile，
        # probe 轮由 run_probe 以 PROBE_WARMUP/PROBE_ITERS 构造
        probe_profile = perf_test.TestProfile(npus=8, minbytes="1K", maxbytes="1M", stepbytes=None,
                                              stepfactor=2, warmup_iters=perf_test.PROBE_WARMUP,
                                              iters=perf_test.PROBE_ITERS)
        probe_args = perf_test.build_test_args(probe_profile, "allreduce", "fp32", "sum",
                                               accel=perf_test.engine_to_accelerator("aiv"))
        self.assertEqual(probe_args[probe_args.index("-a") + 1], "aiv_only")
        self.assertEqual(probe_args[probe_args.index("-w") + 1], "0")
        self.assertEqual(probe_args[probe_args.index("-n") + 1], "1")


class ProfSizeAndReduceTest(unittest.TestCase):
    """size 区间参数与 reduce_type。"""

    def test_interval_maps_to_b_e_f(self):
        profile = perf_test.TestProfile(npus=8, minbytes="1K", maxbytes="1M", stepbytes=None,
                                        stepfactor=2, warmup_iters=10, iters=20)
        args = perf_test.build_test_args(profile, "allreduce", "fp32", "sum", accel="aicpu_ts")
        self.assertEqual(args[args.index("-b") + 1], "1K")
        self.assertEqual(args[args.index("-e") + 1], "1M")
        self.assertEqual(args[args.index("-f") + 1], "2")
        self.assertNotIn("-i", args)

    def test_stepbytes_maps_to_i(self):
        profile = perf_test.TestProfile(npus=8, minbytes="1K", maxbytes="1M", stepbytes=1024,
                                        stepfactor=None, warmup_iters=10, iters=20)
        args = perf_test.build_test_args(profile, "allreduce", "fp32", "sum", accel="aicpu_ts")
        self.assertEqual(args[args.index("-i") + 1], "1024")
        self.assertNotIn("-f", args)

    def test_step_args_mutually_exclusive(self):
        argv = ["--op-types", "allreduce", "--np-total", "8",
                "--minbytes", "1K", "--maxbytes", "1M",
                "--stepbytes", "1024", "--stepfactor", "2"]
        args = perf_test.parse_args(argv)
        with self.assertRaises(ValueError):
            perf_test.validate_args(args)

    def test_reduce_type_field(self):
        # reduce 类算子：传 -o 且 CSV reduce_type=reduce_op
        profile = perf_test.TestProfile(npus=None, minbytes=None, maxbytes=None, stepbytes=None,
                                        stepfactor=None, warmup_iters=10, iters=20)
        args = perf_test.build_test_args(profile, "reduce", "fp32", "max", accel="aicpu_ts")
        self.assertEqual(args[args.index("-o") + 1], "max")
        self.assertEqual(perf_test.reduce_type_for_op("reduce", "max"), "max")
        self.assertEqual(perf_test.reduce_type_for_op("allreduce", "sum"), "sum")
        # 非 reduce 类：不下发 -o，CSV reduce_type=NA
        args = perf_test.build_test_args(profile, "broadcast", "fp32", "sum", accel="aicpu_ts")
        self.assertNotIn("-o", args)
        self.assertEqual(perf_test.reduce_type_for_op("broadcast", "sum"), "NA")


class ProfMergeProbeHitsTest(unittest.TestCase):
    """P13b probe 命中汇总：allgather 档位换算、下限过滤、去重与通配档位。"""

    def test_allgather_rescale_and_floor(self):
        # allgather 锚点 dataSize 是每卡输入（10M/4ranks=2.5M），×4 回请求口径后过 10M 下限
        hits = [(2621440, "DpuAllGatherSoloMesh", 1.0), (1024, "DpuAllGatherSoloMesh", 1.0)]
        merged = perf_test.merge_probe_hits(hits, "allgather", rank_size=4, min_bytes=10485760)
        self.assertEqual(merged, [(10485760, "DpuAllGatherSoloMesh", 1.0)])

    def test_non_rescaled_op_uses_raw_size(self):
        # reduce_scatter 等算子 HCCL inputSize 已是总量口径，不换算
        hits = [(10485760, "DpuReduceScatterSoloMesh", 1.0), (2048, "DpuReduceScatterSoloMesh", 1.0)]
        merged = perf_test.merge_probe_hits(hits, "reduce_scatter", rank_size=4, min_bytes=10485760)
        self.assertEqual(merged, [(10485760, "DpuReduceScatterSoloMesh", 1.0)])

    def test_dedup_keeps_first(self):
        # 多 rank 重复行按 (size, reg_name) 去重，保留首个 cost
        hits = [(8388608, "AivAllGatherSoloMesh", 3.0), (8388608, "AivAllGatherSoloMesh", 5.0)]
        merged = perf_test.merge_probe_hits(hits, "allreduce", rank_size=8, min_bytes=8388608)
        self.assertEqual(merged, [(8388608, "AivAllGatherSoloMesh", 3.0)])

    def test_zero_size_wildcard_retained(self):
        # 无锚点 DFX 行的 size=0 通配档位保留，不受下限过滤影响
        hits = [(0, "DpuAllGatherSoloMesh", 1.0), (512, "DpuAllGatherSoloMesh", 1.0)]
        merged = perf_test.merge_probe_hits(hits, "allgather", rank_size=4, min_bytes=10485760)
        self.assertEqual(merged, [(0, "DpuAllGatherSoloMesh", 1.0)])


class ProfFormatDurationTest(unittest.TestCase):
    """P13 耗时格式化与 ETA 口径。"""

    def test_format_duration(self):
        self.assertEqual(perf_test.format_duration(12.34), "12.3s")
        self.assertEqual(perf_test.format_duration(59.9), "59.9s")
        self.assertEqual(perf_test.format_duration(63), "01m 03s")
        self.assertEqual(perf_test.format_duration(3661), "1h 01m 01s")
        self.assertEqual(perf_test.format_duration(0), "0.0s")
        self.assertEqual(perf_test.format_duration(-5), "0.0s")  # 负数钳 0


class ProfPreflightTest(unittest.TestCase):
    """preflight 环境自检。"""

    def test_python_version_too_old(self):
        with self.assertRaises(perf_test.PreflightError) as ctx:
            perf_test.check_python_version((3, 7, 5))
        self.assertIn("3.8", str(ctx.exception))

    def test_python_version_ok(self):
        self.assertTrue(perf_test.check_python_version((3, 8, 0)))
        self.assertTrue(perf_test.check_python_version((3, 12, 1)))

    def test_mpirun_missing(self):
        with mock.patch.object(perf_test.shutil, "which", return_value=None), \
                mock.patch.object(perf_test.sys, "version_info", (3, 9, 0)):
            with self.assertRaises(perf_test.PreflightError) as ctx:
                perf_test.preflight(["allreduce"], env={"ASCEND_HOME_PATH": "/usr/local/Ascend"})
        self.assertIn("MPICH", str(ctx.exception))
        self.assertIn("Open MPI", str(ctx.exception))

    def test_hccl_test_exe_missing(self):
        with mock.patch.object(perf_test.shutil, "which", return_value="/usr/bin/mpirun"), \
                mock.patch.object(perf_test.sys, "version_info", (3, 9, 0)), \
                mock.patch.object(perf_test.os.path, "isfile", return_value=False):
            with self.assertRaises(perf_test.PreflightError) as ctx:
                perf_test.preflight(["allreduce"], env={"ASCEND_HOME_PATH": "/usr/local/Ascend"})
        self.assertIn("all_reduce_test", str(ctx.exception))
        # 报错路径 = $ASCEND_HOME_PATH + 默认子目录 + exe
        self.assertIn(os.path.join("/usr/local/Ascend", "tools/hccl_test/bin", "all_reduce_test"),
                      str(ctx.exception))

    def test_ascend_home_path_not_set(self):
        with mock.patch.object(perf_test.shutil, "which", return_value="/usr/bin/mpirun"), \
                mock.patch.object(perf_test.sys, "version_info", (3, 9, 0)):
            with self.assertRaises(perf_test.PreflightError) as ctx:
                perf_test.preflight(["allreduce"], env={})
        self.assertIn("ASCEND_HOME_PATH", str(ctx.exception))
        self.assertIn("setenv.bash", str(ctx.exception))

    def test_all_satisfied(self):
        with mock.patch.object(perf_test.shutil, "which", return_value="/usr/bin/mpirun"), \
                mock.patch.object(perf_test.sys, "version_info", (3, 9, 0)), \
                mock.patch.object(perf_test.os.path, "isfile", return_value=True), \
                mock.patch.dict(os.environ, {}, clear=True):
            # 通过时返回 MPI flavor 供命令构造
            self.assertEqual(perf_test.preflight(
                ["allreduce", "alltoallv"], env={"ASCEND_HOME_PATH": "/usr/local/Ascend"}), "mpich")


class ProfExePathTest(unittest.TestCase):
    """P12b hccl_test 路径拼接（$ASCEND_HOME_PATH + 相对路径）。"""

    ENV = {"ASCEND_HOME_PATH": "/usr/local/Ascend"}

    def test_default_subdir_join(self):
        # 默认：$ASCEND_HOME_PATH/tools/hccl_test/bin/<exe>
        self.assertEqual(
            perf_test.exe_path_of("allreduce", None, env=self.ENV),
            os.path.join("/usr/local/Ascend", "tools/hccl_test/bin", "all_reduce_test"))

    def test_custom_subdir_join(self):
        self.assertEqual(
            perf_test.exe_path_of("reduce_scatter", "my/bin", env=self.ENV),
            os.path.join("/usr/local/Ascend", "my/bin", "reduce_scatter_test"))

    def test_absolute_bin_dir_passthrough(self):
        # 绝对路径直接使用，不拼 ASCEND_HOME_PATH
        path = perf_test.exe_path_of("allreduce", "/opt/custom/bin", env={})
        self.assertEqual(path, os.path.join("/opt/custom/bin", "all_reduce_test"))

    def test_missing_env_raises(self):
        with self.assertRaises(perf_test.PreflightError):
            perf_test.exe_path_of("allreduce", None, env={})
        # 空白串同未设置
        with self.assertRaises(perf_test.PreflightError):
            perf_test.exe_path_of("allreduce", None, env={"ASCEND_HOME_PATH": "  "})

    def test_env_whitespace_stripped(self):
        path = perf_test.exe_path_of("allgather", None, env={"ASCEND_HOME_PATH": " /usr/local/Ascend "})
        self.assertTrue(path.startswith(os.path.join("/usr/local/Ascend", "")))


def write_csv(path, rows):
    import csv as csv_mod
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv_mod.DictWriter(handle, fieldnames=list(perf_test.CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class OptDedupTest(unittest.TestCase):
    """O01 最新优先去重。"""

    def test_keep_latest(self):
        rows = [
            make_row(timestamp="20261012090000000", alg_bandwidth="1.0"),
            make_row(timestamp="20261012090000100", alg_bandwidth="2.0"),  # 最新，保留
            make_row(timestamp="20261012090000050", alg_bandwidth="3.0"),  # 旧，丢弃
        ]
        kept, dropped = optimize_config.dedup_latest(rows)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 2)
        self.assertEqual(kept[0]["timestamp"], "20261012090000100")
        self.assertEqual(kept[0]["alg_bandwidth(GB/s)"], "2.0")

    def test_invalid_timestamp_treated_as_oldest(self):
        # 非法 timestamp（被改/旧格式）视作最旧：与合法行竞争时恒被淘汰，并打 WARN
        rows = [
            make_row(timestamp="99999999999999999", alg_bandwidth="9.9"),
            make_row(timestamp="20261012090000100", alg_bandwidth="2.0"),
        ]
        with self.assertLogs(optimize_config.logger, level="WARNING") as logs:
            kept, dropped = optimize_config.dedup_latest(rows)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 1)
        self.assertEqual(kept[0]["alg_bandwidth(GB/s)"], "2.0")
        self.assertIn("invalid timestamp", logs.output[0])

    def test_different_key_not_deduped(self):
        rows = [make_row(), make_row(engine="aiv")]
        kept, dropped = optimize_config.dedup_latest(rows)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 0)


class OptGroupTest(unittest.TestCase):
    """O02 分组键正确性。"""

    def test_engine_not_group_key(self):
        rows = [make_row(engine="aicpu"), make_row(engine="aiv"), make_row(engine="dpu")]
        groups = optimize_config.group_rows(rows)
        self.assertEqual(len(groups), 1)  # engine 不同仍同组（比较维度）
        rows2 = [make_row(), make_row(data_type="fp16")]
        self.assertEqual(len(optimize_config.group_rows(rows2)), 2)
        rows3 = [make_row(), make_row(reduce_type="NA")]
        self.assertEqual(len(optimize_config.group_rows(rows3)), 2)
        rows4 = [make_row(), make_row(**{"ranks.ranks": "16"})]
        self.assertEqual(len(optimize_config.group_rows(rows4)), 2)


class OptSelectBestTest(unittest.TestCase):
    """O03 带宽选优 + latency tie-break。"""

    def test_bandwidth_max_wins(self):
        rows = [
            make_row(alg_bandwidth="1.0", alg_latency="50"),
            make_row(alg_bandwidth="3.0", alg_latency="200", engine="aiv"),
            make_row(alg_bandwidth="2.0", alg_latency="80", engine="ccums"),
        ]
        best = optimize_config.select_best_per_size(rows)
        self.assertEqual(best[8192]["engine"], "aiv")

    def test_latency_tie_break(self):
        rows = [
            make_row(alg_bandwidth="2.0", alg_latency="100"),
            make_row(alg_bandwidth="2.0", alg_latency="60", engine="aiv"),
        ]
        best = optimize_config.select_best_per_size(rows)
        self.assertEqual(best[8192]["engine"], "aiv")

    def test_per_size_winners_differ_by_size(self):
        # 每个 size 都有 3 个候选，且不同 size 的最优算法不同：
        # 选优必须逐 size 独立比较，而非全局取一个最优
        rows = [
            # size=8192：aiv 最优（5.0 > 3.0 > 1.0）
            make_row(size_bytes="8192", alg_bandwidth="1.0"),
            make_row(size_bytes="8192", alg_bandwidth="5.0", engine="aiv"),
            make_row(size_bytes="8192", alg_bandwidth="3.0", engine="ccums"),
            # size=16384：ccums 最优（4.0 > 2.0 > 1.5）
            make_row(size_bytes="16384", alg_bandwidth="2.0"),
            make_row(size_bytes="16384", alg_bandwidth="1.5", engine="aiv"),
            make_row(size_bytes="16384", alg_bandwidth="4.0", engine="ccums"),
        ]
        best = optimize_config.select_best_per_size(rows)
        self.assertEqual(len(best), 2)
        self.assertEqual(best[8192]["engine"], "aiv")
        self.assertEqual(best[16384]["engine"], "ccums")


class OptMergeIntervalsTest(unittest.TestCase):
    """O04 区间合并。"""

    def test_merge_same_algo_adjacent(self):
        points = {
            8192: make_row(size_bytes="8192"),
            16384: make_row(size_bytes="16384"),
            32768: make_row(size_bytes="32768", engine="aiv"),
        }
        intervals = optimize_config.merge_intervals(points)
        self.assertEqual(len(intervals), 2)
        self.assertEqual((intervals[0][0], intervals[0][1]), (8192, 16384))
        self.assertEqual((intervals[1][0], intervals[1][1]), (32768, 32768))

    def test_diff_algo_single_points(self):
        points = {
            8192: make_row(size_bytes="8192"),
            16384: make_row(size_bytes="16384", engine="aiv"),
        }
        intervals = optimize_config.merge_intervals(points)
        self.assertEqual([(lo, hi) for lo, hi, _ in intervals], [(8192, 8192), (16384, 16384)])


class OptFillGapsTest(unittest.TestCase):
    """O05 补缝（归右）。"""

    def test_gap_assigned_to_right(self):
        intervals = [
            (8192, 8192, make_row(size_bytes="8192")),                       # A: mesh
            (32768, 32768, make_row(size_bytes="32768", engine="aiv")),      # B: aiv
        ]
        filled = optimize_config.fill_gaps(intervals)
        self.assertEqual([(lo, hi) for lo, hi, _ in filled], [(8192, 8192), (8193, 32768)])
        # 右侧区间算法（B=aiv）覆盖整条缝
        self.assertEqual(filled[1][2]["engine"], "aiv")

    def test_no_gap_no_change(self):
        intervals = [(8192, 8192, make_row()), (8193, 9000, make_row(engine="aiv"))]
        filled = optimize_config.fill_gaps(intervals)
        self.assertEqual([(lo, hi) for lo, hi, _ in filled], [(8192, 8192), (8193, 9000)])

    def test_first_interval_not_extended_left(self):
        intervals = [(4096, 4096, make_row(size_bytes="4096"))]
        filled = optimize_config.fill_gaps(intervals)
        self.assertEqual(filled[0][0], 4096)  # 首区间不向 0 扩展


class OptWhitelistTest(unittest.TestCase):
    """O06 白名单硬校验。"""

    def test_strictordered_dropped_with_count(self):
        row = make_row(**{"algorithm.executor_type": "strictordered"})
        self.assertEqual(optimize_config.validate_winner(row), "executor")
        conf, _ = optimize_config.build_conf([row])
        self.assertEqual(conf["op_types"], {})  # 无 rule 产出

    def test_invalid_template_dtype_engine_op(self):
        # meshconcurrent / nhraicpureduce 已回白名单（plugin 不做 template 枚举校验），
        # 拼写错误的 template 仍须拒绝
        self.assertEqual(optimize_config.validate_winner(
            make_row(**{"algorithm.template_type": "meshconcurrent"})), None)
        self.assertEqual(optimize_config.validate_winner(
            make_row(**{"algorithm.template_type": "nhraicpureduce"})), None)
        self.assertEqual(optimize_config.validate_winner(
            make_row(**{"algorithm.template_type": "badtemplate"})), "template")
        self.assertEqual(optimize_config.validate_winner(
            make_row(engine="badengine")), "engine")
        # fp8 系已入脚本白名单（无别名）；插件 data_type 映射补齐前产出规则会被运行时剔除
        for dtype in ("hif8", "fp8e4m3", "fp8e5m2", "fp8e8m0"):
            self.assertIsNone(optimize_config.validate_winner(make_row(data_type=dtype)))
        self.assertEqual(optimize_config.validate_winner(
            make_row(op_type="allgatherv")), "op_type")
        self.assertIsNone(optimize_config.validate_winner(make_row()))

    def test_dropped_counted_in_build(self):
        # strictordered 带宽最高但白名单外 → 该点丢弃，不产 rule
        rows = [
            make_row(alg_bandwidth="5.0", **{"algorithm.executor_type": "strictordered"}),
            make_row(alg_bandwidth="1.0"),
        ]
        conf, total = optimize_config.build_conf(rows)
        # strictordered 胜者被丢弃后该 size 点无 rule；sole{mesh} 是唯一合法候选，
        # 但选优在先：strictordered 胜出 → 点丢弃 → 无 rule
        self.assertEqual(total, 0)


class OptMultiTemplateTest(unittest.TestCase):
    """多级模板放行 + conf 拼接串口径转换。"""

    def test_multi_template_tokens_validated_individually(self):
        # 多级逐 token 过白名单：全在白名单内放行，任一 token 白名单外拒绝
        self.assertIsNone(optimize_config.validate_winner(
            make_row(**{"algorithm.template_type": "meshconcur,nhr,nhr"})))
        self.assertEqual(optimize_config.validate_winner(
            make_row(**{"algorithm.template_type": "mesh,badtemplate"})), "template")

    def test_multi_template_concatenated_in_rule(self):
        # rule.template 写 plugin 拼接串口径（去逗号），CSV 行内保持语义串
        rows = [make_row(alg_bandwidth="9.0", **{"algorithm.executor_type": "sequence",
                                                "algorithm.template_type": "meshconcur,nhr,nhr"})]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(total, 1)
        rule = conf["op_types"]["allreduce"]["rules"][0]
        self.assertEqual(rule["template"], "meshconcurnhrnhr")
        self.assertEqual(rule["executor"], "sequence")

    def test_single_template_passthrough(self):
        # 单级语义串与拼接串同形，转换透传
        rows = [make_row(alg_bandwidth="9.0")]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(total, 1)
        rule = conf["op_types"]["allreduce"]["rules"][0]
        self.assertEqual(rule["template"], "mesh")


class OptEnginePassthroughTest(unittest.TestCase):
    """O08 engine 透传。"""

    def test_engine_written_as_is(self):
        for engine in ("aicpu", "aiv", "ccums", "ccusched"):
            conf, total = optimize_config.build_conf([make_row(engine=engine)])
            self.assertEqual(total, 1)
            self.assertEqual(conf["op_types"]["allreduce"]["rules"][0]["engine"], engine)

    def test_dpu_independent_rules(self):
        # dpu 与 device 并存：各成一组独立产 rule；device 排序在前
        rows = [
            make_row(engine="aicpu", alg_bandwidth="1.0"),
            make_row(engine="dpu", alg_bandwidth="5.0"),
        ]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(total, 2)
        rules = conf["op_types"]["allreduce"]["rules"]
        self.assertEqual([r["engine"] for r in rules], ["aicpu", "dpu"])

    def test_dpu_rule_forces_min_servers_2(self):
        # dpu rule 强制 min_servers>=2（hostdpu 拓扑必要条件）：单机采集的 dpu
        # 数据不会产死 rule；servers=1 时丢弃 max_servers=1 避免 min>max SchemaError
        rows = [make_row(engine="dpu", **{"ranks.servers": "1"})]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(total, 1)
        match = conf["op_types"]["allreduce"]["rules"][0]["match"]
        self.assertEqual(match["min_servers"], 2)
        self.assertNotIn("max_servers", match)

    def test_dpu_rule_keeps_multi_server_bounds(self):
        # 多 server 采集的 dpu 数据：min=max=servers 原样保留
        rows = [make_row(engine="dpu", **{"ranks.servers": "2"})]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(total, 1)
        match = conf["op_types"]["allreduce"]["rules"][0]["match"]
        self.assertEqual(match["min_servers"], 2)
        self.assertEqual(match["max_servers"], 2)


class OptFailedFilterTest(unittest.TestCase):
    """O09 failed 行过滤。"""

    def test_failed_dropped_null_kept(self):
        rows = [
            make_row(check_result="failed", alg_bandwidth="99.0"),
            make_row(check_result="NULL", alg_bandwidth="1.0", engine="aiv"),
            make_row(check_result="success", alg_bandwidth="2.0", engine="ccums"),
        ]
        kept, dropped = optimize_config.filter_failed(rows)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(kept), 2)
        # NULL 保留参与选优：胜者为 ccums（带宽 2.0）
        best = optimize_config.select_best_per_size(kept)
        self.assertEqual(best[8192]["engine"], "ccums")

    def test_noresult_dropped(self):
        rows = [
            make_row(check_result="noresult", alg_bandwidth=""),
            make_row(check_result="success", alg_bandwidth="2.0", engine="ccums"),
        ]
        kept, dropped = optimize_config.filter_failed(rows)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["check_result"], "success")

    def test_fallback_dropped(self):
        # 行级回退校验标记的 fallback 行与 failed 同等丢弃：
        # probe 名单不含该 (算法, 档位)，实际执行的是回退算法，测的不是目标算法
        rows = [
            make_row(check_result="fallback", alg_bandwidth="99.0"),
            make_row(check_result="FALLBACK", alg_bandwidth="98.0", engine="aiv"),
            make_row(check_result="success", alg_bandwidth="2.0", engine="ccums"),
        ]
        kept, dropped = optimize_config.filter_failed(rows)
        self.assertEqual(dropped, 2)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["engine"], "ccums")


class OptBytesNormalizeTest(unittest.TestCase):
    """字节口径归一化（allgather 系 ÷ ranks、scatter × ranks，其余透传）。"""

    def test_allgather_divided_by_ranks(self):
        # allgather：采集 size 是输出总量，归一化为每 rank 发送量（向上取整）
        rows = [make_row(op_type="allgather", size_bytes="81920", **{"ranks.ranks": "8"})]
        normalized = optimize_config.normalize_bytes(rows)
        self.assertEqual(normalized[0]["size_bytes"], "10240")

    def test_allgatherv_ceil_non_divisible(self):
        # allgatherv：非整除向上取整（ceil）
        rows = [make_row(op_type="allgatherv", size_bytes="8193", **{"ranks.ranks": "8"})]
        normalized = optimize_config.normalize_bytes(rows)
        self.assertEqual(normalized[0]["size_bytes"], "1025")

    def test_scatter_multiplied_by_ranks(self):
        # scatter：采集 size 是每 rank 收到量，插件 nBytes 是 root 总发出量，需 × ranks
        rows = [make_row(op_type="scatter", size_bytes="10240", **{"ranks.ranks": "8"})]
        normalized = optimize_config.normalize_bytes(rows)
        self.assertEqual(normalized[0]["size_bytes"], "81920")

    def test_aligned_ops_passthrough(self):
        # 其余 op（allreduce 等）采集口径与插件一致，原值透传
        rows = [make_row(op_type="allreduce", size_bytes="8192"),
                make_row(op_type="alltoall", size_bytes="81920")]
        normalized = optimize_config.normalize_bytes(rows)
        self.assertEqual(normalized[0]["size_bytes"], "8192")
        self.assertEqual(normalized[1]["size_bytes"], "81920")

    def test_degenerate_cases_passthrough(self):
        # ranks<=1 / size<=0 / 空值：不换算，原样透传（含字符串原样保留）
        rows = [
            make_row(op_type="allgather", size_bytes="8192", **{"ranks.ranks": "1"}),
            make_row(op_type="allgather", size_bytes="0"),
            make_row(op_type="allgather", size_bytes=""),
        ]
        normalized = optimize_config.normalize_bytes(rows)
        self.assertEqual(normalized[0]["size_bytes"], "8192")
        self.assertEqual(normalized[1]["size_bytes"], "0")
        self.assertEqual(normalized[2]["size_bytes"], "")

    def test_build_conf_end_to_end_allgather(self):
        # 端到端：conf 规则 min/max_bytes 落每 rank 口径（81920/8ranks → 10240）
        rows = [make_row(op_type="allgather", size_bytes="81920", **{"ranks.ranks": "8"})]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(total, 1)
        match = conf["op_types"]["allgather"]["rules"][0]["match"]
        self.assertEqual(match["min_bytes"], 10240)
        self.assertEqual(match["max_bytes"], 10240)


class OptConfStructureTest(unittest.TestCase):
    """O10 conf 结构合规。"""

    def test_conf_structure(self):
        # 端到端选优不变量：每个 [lo,hi] 区间的算法 = 落在该区间内各采集 size 点上的
        # 逐 size 最优合法算法（逐点比较 → 合并 → 补缝归右后的最终归属）
        rows = [
            # 8192：aiv 胜（5.0）
            make_row(size_bytes="8192", alg_bandwidth="1.0"),
            make_row(size_bytes="8192", alg_bandwidth="5.0", engine="aiv"),
            make_row(size_bytes="8192", alg_bandwidth="3.0", engine="ccums"),
            # 16384：aiv 胜（2.5）→ 与 8192 同算法，合并
            make_row(size_bytes="16384", alg_bandwidth="2.5", engine="aiv"),
            # 32768：ccums 胜（4.0）→ 单点区间 + 补缝归右
            make_row(size_bytes="32768", alg_bandwidth="4.0", engine="ccums"),
            make_row(size_bytes="32768", alg_bandwidth="0.5", engine="aiv"),
        ]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(conf["version"], 1)
        self.assertIn("allreduce", conf["op_types"])
        rules = conf["op_types"]["allreduce"]["rules"]
        self.assertEqual(total, len(rules))
        self.assertEqual(total, 2)
        match = rules[0]["match"]
        for field in ("min_ranks", "max_ranks", "min_npus_per_server", "max_npus_per_server",
                      "min_servers", "max_servers", "min_pods", "max_pods",
                      "min_super_pods", "max_super_pods", "min_bytes", "max_bytes", "data_type"):
            self.assertIn(field, match)
        self.assertEqual(match["min_ranks"], 8)
        self.assertEqual(match["max_npus_per_server"], 8)
        self.assertEqual(match["data_type"], "fp32")
        # 第一条：8192/16384 均 aiv 胜 → 合并
        self.assertEqual((match["min_bytes"], match["max_bytes"]), (8192, 16384))
        self.assertEqual(rules[0]["engine"], "aiv")
        self.assertEqual(rules[0]["executor"], "sole")
        self.assertEqual(rules[0]["template"], "mesh")
        self.assertEqual(rules[0]["cost"], 0.0)
        # 第二条：32768 ccums，补缝后向左扩展
        self.assertEqual(rules[1]["match"]["min_bytes"], 16385)
        self.assertEqual(rules[1]["match"]["max_bytes"], 32768)
        self.assertEqual(rules[1]["engine"], "ccums")

    def test_unspecified_pods_sentinel_skipped(self):
        # pods/super_pods=0（采集侧未指定哨兵）时 match 不写该维度（单机错位规避）
        rows = [make_row(size_bytes="8192", **{"ranks.pods": "0", "ranks.super_pods": "0"})]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(total, 1)
        match = conf["op_types"]["allreduce"]["rules"][0]["match"]
        for absent in ("min_pods", "max_pods", "min_super_pods", "max_super_pods"):
            self.assertNotIn(absent, match)
        # 显式指定的维度仍写入
        self.assertEqual(match["min_ranks"], 8)
        self.assertEqual(match["max_servers"], 1)

    def test_zero_ranks_sentinel_skipped(self):
        # 哨兵跳过逻辑对全部 5 个拓扑字段一致生效（空值同样按 0 处理）
        rows = [make_row(size_bytes="8192", **{"ranks.ranks": "0", "ranks.servers": ""})]
        conf, _ = optimize_config.build_conf(rows)
        match = conf["op_types"]["allreduce"]["rules"][0]["match"]
        for absent in ("min_ranks", "max_ranks", "min_servers", "max_servers"):
            self.assertNotIn(absent, match)
        self.assertEqual(match["min_npus_per_server"], 8)

    def test_end_to_end_files(self):
        # 端到端：CSV 文件 → conf JSON 文件
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.csv")
            dst = os.path.join(tmp, "out.json")
            write_csv(src, [make_row(alg_bandwidth="1.0"),
                            make_row(alg_bandwidth="3.0", engine="aiv")])
            argv = ["--input", src, "--output", dst]
            with mock.patch.object(sys, "argv", ["optimize_config.py"] + argv):
                self.assertEqual(optimize_config.main(argv), 0)
            import json
            with open(dst, encoding="utf-8") as handle:
                conf = json.load(handle)
            self.assertEqual(conf["version"], 1)
            rules = conf["op_types"]["allreduce"]["rules"]
            self.assertEqual(rules[0]["engine"], "aiv")


class OptBuffsizeTest(unittest.TestCase):
    """O11 HCCL_BUFFSIZE 分组与规则绑定。"""

    def test_buffsize_bound_into_match(self):
        # 采集显式设置 buffsize 时规则等值绑定 buffer_size（= buffsize × 2MB），运行环境不同则不命中
        conf, _ = optimize_config.build_conf([make_row()])
        match = conf["op_types"]["allreduce"]["rules"][0]["match"]
        self.assertEqual(match["buffer_size"], 2048 * 2 * 1024 * 1024)

    def test_buffsize_empty_skipped(self):
        # 采集未显式设置（空）时跳过绑定，规则全局生效（依赖两端环境一致）
        conf, _ = optimize_config.build_conf([make_row(**{"HCCL_BUFFSIZE": ""})])
        self.assertNotIn("buffer_size", conf["op_types"]["allreduce"]["rules"][0]["match"])

    def test_mixed_buffsize_independent_rules(self):
        # 不同 buffsize 独立分组选优：bytes 区间重叠也各自产规则、各自绑定（运行时等值互斥，共存安全）
        rows = [
            make_row(alg_bandwidth="9.0"),
            make_row(engine="aiv", alg_bandwidth="5.0", **{"HCCL_BUFFSIZE": "4096"}),
        ]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(total, 2)
        rules = conf["op_types"]["allreduce"]["rules"]
        self.assertEqual(rules[0]["match"]["buffer_size"], 2048 * 2 * 1024 * 1024)
        self.assertEqual(rules[1]["match"]["buffer_size"], 4096 * 2 * 1024 * 1024)

    def test_dedup_not_across_buffsize(self):
        # 同算法同 size 不同 buffsize 是不同采集点：去重 key 含 buffsize，不互相吞
        rows = [
            make_row(alg_bandwidth="1.0", timestamp="20261012090000000"),
            make_row(alg_bandwidth="2.0", **{"HCCL_BUFFSIZE": "4096"}),
        ]
        deduped, dropped = optimize_config.dedup_latest(rows)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(deduped), 2)


class ProfRunOneTimeoutTest(unittest.TestCase):
    """P04+ run_one 超时：挂死轮被杀、落失败、不阻塞采集。"""

    def test_run_one_timeout_returns_failure(self):
        # 模拟挂死进程：sleep 超过 timeout → TimeoutExpired → (输出, stderr, False)
        cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
        import time
        start = time.monotonic()
        stdout, stderr, ok = perf_test.run_one(cmd, env=None, timeout=2)
        elapsed = time.monotonic() - start
        self.assertFalse(ok)
        self.assertLess(elapsed, 15)  # 远小于 sleep(30)，证明被超时杀掉
        self.assertIn("run timeout after 2s", stderr)

    def test_run_one_normal_no_timeout(self):
        # 正常进程不受 timeout 影响，返回码透传
        cmd = [sys.executable, "-c", "print('hello')"]
        stdout, _, ok = perf_test.run_one(cmd, env=None, timeout=30)
        self.assertTrue(ok)
        self.assertIn("hello", stdout)


def make_probe_algo(**overrides):
    """构造一条算法池项（默认 aicpu_sole{mesh}；index 缺省由 make_probe_group 按位置回填）。"""
    algo = {"algo": "aicpu_sole{mesh}", "reg_name": "AicpuAllReduceSoleMesh"}
    algo.update(overrides)
    return algo


def make_probe_group(algos=None, sizes=None, **overrides):
    """构造一个分组条目（默认 allreduce / sum / fp32，算法池单算法，档位 8K 用 [0]）。

    sizes 未给时按算法池长度推导：每档位引用池内全部下标；index 缺省按数组位置回填。
    """
    algos = algos if algos is not None else [make_probe_algo()]
    for pos, algo in enumerate(algos):
        algo.setdefault("index", pos)
    full = list(range(len(algos)))
    group = {"op": "allreduce", "reduce_op": "sum", "data_type": "fp32",
             "algo_count": len(algos), "algos": algos,
             "sizes": sizes if sizes is not None else [[8192, full]]}
    group.update(overrides)
    return group


def make_probe_payload(groups=None, unresolved=None):
    """构造一份 probe 名单 JSON payload（拓扑 8 卡单机 / sum,prod / fp32,fp16）。"""
    return {
        "schema_version": tuner_common.PROBE_LIST_SCHEMA_VERSION,
        "probe_time": "2026-09-29 12:00:00",
        "topo": {"ranks.ranks": 8, "ranks.npus_per_server": 8, "ranks.servers": 1,
                 "ranks.pods": 1, "ranks.super_pods": 1},
        "probe_params": {"op_types": ["allreduce"], "reduce_ops": ["sum", "prod"],
                         "data_types": ["fp32", "fp16"], "engines": list(perf_test.PROF_ENGINES),
                         "size_params": {"minbytes": 1024, "maxbytes": 1048576,
                                         "stepbytes": None, "stepfactor": 2.0}},
        "groups": groups if groups is not None else [],
        "unresolved": unresolved or [],
    }


class SplitAlgoNameTest(unittest.TestCase):
    """split_algo_name 注册名反分解（规则对齐 alg_attrs_registry.cc ParseAlgName）。"""

    def test_single_template(self):
        self.assertEqual(tuner_common.split_algo_name("AicpuAllReduceSoleMeshOneShot"),
                         ("allreduce", "aicpu", "sole", ("meshoneshot",)))

    def test_multi_template_greedy_tokens(self):
        # 长词优先：MeshConcur 先于 Mesh；NHRAicpuReduce 整词命中
        self.assertEqual(tuner_common.split_algo_name("AicpuAllGatherSequenceMeshConcurNHR"),
                         ("allgather", "aicpu", "sequence", ("meshconcur", "nhr")))
        self.assertEqual(tuner_common.split_algo_name("AicpuAllReduceSequenceMeshNHRAicpuReduce"),
                         ("allreduce", "aicpu", "sequence", ("mesh", "nhraicpureduce")))
        self.assertEqual(tuner_common.split_algo_name("AicpuAllReduceConcurMeshTwoShotNHR"),
                         ("allreduce", "aicpu", "concur", ("meshtwoshot", "nhr")))

    def test_engine_prefixes(self):
        self.assertEqual(tuner_common.split_algo_name("CcuSchedAllReduceParallelMeshNHR")[1], "ccusched")
        self.assertEqual(tuner_common.split_algo_name("CcuMSAllReduceSoleMesh")[1], "ccums")
        self.assertEqual(tuner_common.split_algo_name("AivAllReduceSoleMesh")[1], "aiv")
        self.assertEqual(tuner_common.split_algo_name("DpuAllReduceSequenceMeshNHR")[1], "dpu")

    def test_collect_domain_only_ops(self):
        # barrier/send 等非采集算子反分解成功但 op 不在采集域 → None
        self.assertIsNone(tuner_common.split_algo_name("AicpuSendSoleMesh"))

    def test_blacklist_removed_multilink_meshconcurrent_valid(self):
        # 旧黑名单已删：multilink / meshconcurrent 回归采集域 13 枚举，反分解应成功
        self.assertEqual(tuner_common.split_algo_name("AicpuAllReduceSoleNHRMultiLink"),
                         ("allreduce", "aicpu", "sole", ("nhrmultilink",)))
        self.assertEqual(tuner_common.split_algo_name("AicpuAllReduceSoleMeshConcurrent"),
                         ("allreduce", "aicpu", "sole", ("meshconcurrent",)))
        # omnipipe executor 不在采集域 6 枚举内
        self.assertIsNone(tuner_common.split_algo_name("AicpuAllReduceOmniPipeMeshNHR"))

    def test_unparsable_rejected(self):
        # 未知 engine 前缀 / 缺模板 / 空串 / 非 opType 中缀
        self.assertIsNone(tuner_common.split_algo_name("XpuAllReduceSoleMesh"))
        self.assertIsNone(tuner_common.split_algo_name("AicpuAllReduceSole"))
        self.assertIsNone(tuner_common.split_algo_name(""))
        self.assertIsNone(tuner_common.split_algo_name("AicpuBadOpSoleMesh"))
        # 反分解结果与 Forward 语义可往返：3D 串重组后可被 parse_algos 接受
        for reg_name in ("AicpuAllReduceSoleMeshOneShot", "CcuSchedAllReduceParallelMeshNHR"):
            _op, _engine, executor, templates = tuner_common.split_algo_name(reg_name)
            self.assertEqual(tuner_common.parse_algos(tuner_common.algo_to_env(executor, list(templates))),
                             [(executor, list(templates))])


class AlgoStringTest(unittest.TestCase):
    """make_algo_string / parse_algo_string：名单 algos 项与 HCCL_ALGO 写法互转。"""

    def test_roundtrip(self):
        cases = [("aicpu", "sole", ["mesh"], "aicpu_sole{mesh}"),
                 ("aiv", "sole", ["meshoneshot"], "aiv_sole{meshoneshot}"),
                 ("dpu", "sequence", ["mesh", "nhr"], "dpu_sequence{mesh,nhr}"),
                 ("ccums", "parallel", ["meshconcurrent", "nhr"], "ccums_parallel{meshconcurrent,nhr}")]
        for engine, executor, templates, algo_str in cases:
            self.assertEqual(tuner_common.make_algo_string(engine, executor, templates), algo_str)
            self.assertEqual(tuner_common.parse_algo_string(algo_str), (engine, executor, templates))

    def test_engine_prefix_required(self):
        for bad in ("sole{mesh}", "sole_mesh", "aicpu_sole{mesh};aiv_sole{mesh}"):
            with self.assertRaises(ValueError):
                tuner_common.parse_algo_string(bad)

    def test_executor_template_validated(self):
        # executor{tpl} 段复用 parse_algos 的权威枚举校验
        with self.assertRaises(ValueError):
            tuner_common.parse_algo_string("aicpu_unknownexecutor{mesh}")
        with self.assertRaises(ValueError):
            tuner_common.parse_algo_string("aicpu_sole{unknowntpl}")


class ProfParseCostTablesTest(unittest.TestCase):
    """parse_cost_tables：costTable 日志 → 候选条目（probe 真相源）。"""

    def test_parse_anchor_paired_rows(self):
        log = (
            "[INFO] HCCL(1,pid:100):x.cc:111:[SelectorEngine] SelectMinCost: costTable count=3, "
            "opType=3, dataSize=8192.\n"
            "[INFO] ...:[SelectorEngine] | idx | algName                                          | cost         |\n"
            "[INFO] ...:[SelectorEngine] |   0 | AicpuAllReduceSoleMeshOneShot                    |         1.23 |\n"
            "[INFO] ...:[SelectorEngine] |   1 | AivAllReduceSoleMesh                             |      2.345678 |\n"
            "[INFO] ...:[SelectorEngine] |   2 | AicpuAllReduceSoleNHRMultiLink                   |        -1.00 |\n"
        )
        hits, has_anchor, nonempty = perf_test.parse_cost_tables(log)
        self.assertTrue(has_anchor)
        # cost<0 行（不适用候选）丢弃，正 cost 行带档位落表
        self.assertEqual(hits, [(8192, "AicpuAllReduceSoleMeshOneShot", 1.23),
                                (8192, "AivAllReduceSoleMesh", 2.345678)])

    def test_multi_rank_and_multi_size_dedupe(self):
        log = (
            "SelectMinCost: costTable count=1, opType=3, dataSize=8192.\n"
            "|   0 | AicpuAllReduceSoleMeshOneShot                    |         1.23 |\n"
            "SelectMinCost: costTable count=1, opType=3, dataSize=8192.\n"
            "|   0 | AicpuAllReduceSoleMeshOneShot                    |         1.23 |\n"
            "SelectMinCost: costTable count=1, opType=3, dataSize=16384.\n"
            "|   0 | AicpuAllReduceSoleMesh                           |         0.50 |\n"
        )
        hits, _, _ = perf_test.parse_cost_tables(log)
        # 同 (size, reg) 的多 rank 重复去重，不同档位分开记录
        self.assertEqual(hits, [(8192, "AicpuAllReduceSoleMeshOneShot", 1.23),
                                (16384, "AicpuAllReduceSoleMesh", 0.5)])

    def test_five_column_table_not_matched(self):
        # selected 附加表（5 列）与表头行（idx 非数字）不得误收
        log = (
            "SelectMinCost: costTable count=1, opType=3, dataSize=8192.\n"
            "| idx | algName                                          | cost         |\n"
            "|   0 | AicpuAllReduceSoleMeshOneShot                    |         1.23 |\n"
            "+-----+--------------------------------------------------+--------------+\n"
            "|   0 | AicpuAllReduceSoleMeshOneShot                    |         1.23 | aicpu | 100 |\n"
        )
        hits, _, _ = perf_test.parse_cost_tables(log)
        self.assertEqual(hits, [(8192, "AicpuAllReduceSoleMeshOneShot", 1.23)])

    def test_no_anchor_no_hits(self):
        hits, has_anchor, nonempty = perf_test.parse_cost_tables("plain hccl_test stdout without selector logs")
        self.assertFalse(has_anchor)
        self.assertFalse(nonempty)
        self.assertEqual(hits, [])

    def test_dfx_rows_attached_to_next_anchor(self):
        # DFX 行（构建期，无 dataSize）挂到其后最近 SelectMinCost 锚点的档位
        log = (
            "====== [DFX_CostTableDump] algoCount=2 ======\n"
            "  [DFX_CostTableDump] [1/2] algName=AicpuAllReduceSoleMeshOneShot, cost=1.230000\n"
            "  [DFX_CostTableDump] [2/2] algName=AivAllReduceSoleMesh, cost=-1.000000\n"
            "SelectMinCost: costTable count=2, opType=3, dataSize=8192.\n"
            "|   0 | AicpuAllReduceSoleMeshOneShot                    |         1.23 |\n"
            "|   1 | AivAllReduceSoleMesh                             |        -1.00 |\n"
            "====== [DFX_CostTableDump] algoCount=1 ======\n"
            "  [DFX_CostTableDump] [1/1] algName=AicpuAllReduceSoleMesh, cost=0.500000\n"
            "SelectMinCost: costTable count=1, opType=3, dataSize=16384.\n"
        )
        hits, has_anchor, _nonempty = perf_test.parse_cost_tables(log)
        self.assertTrue(has_anchor)
        # DFX 与表格行按 (size, reg) 去重；cost<0 丢弃；第二个 DFX 挂 16384
        self.assertEqual(hits, [(8192, "AicpuAllReduceSoleMeshOneShot", 1.23),
                                (16384, "AicpuAllReduceSoleMesh", 0.5)])

    def test_dfx_anchor_only_counts_as_anchor(self):
        # 只有 DFX 锚点也算有 costTable（fail-fast 判据放宽），行挂 size=0 通配档位
        log = (
            "====== [DFX_CostTableDump] algoCount=1 ======\n"
            "  [DFX_CostTableDump] [1/1] algName=AicpuAllReduceSoleMesh, cost=0.500000\n"
        )
        hits, has_anchor, _nonempty = perf_test.parse_cost_tables(log)
        self.assertTrue(has_anchor)
        self.assertEqual(hits, [(0, "AicpuAllReduceSoleMesh", 0.5)])

    def test_dfx_empty_algname_skipped(self):
        # algName 为空（注册名为 null）的行不收
        log = (
            "====== [DFX_CostTableDump] algoCount=1 ======\n"
            "  [DFX_CostTableDump] [1/1] algName=, cost=0.500000\n"
            "SelectMinCost: costTable count=1, opType=3, dataSize=8192.\n"
        )
        hits, has_anchor, _nonempty = perf_test.parse_cost_tables(log)
        self.assertTrue(has_anchor)
        self.assertEqual(hits, [])

    def test_empty_table_vs_all_filtered(self):
        # count=0 锚点 = 表为空（nonempty=False）；count>0 但行全被过滤 = 表非空（nonempty=True）
        empty_log = "SelectMinCost: costTable count=0, opType=3, dataSize=8192.\n"
        hits, has_anchor, nonempty = perf_test.parse_cost_tables(empty_log)
        self.assertTrue(has_anchor)
        self.assertFalse(nonempty)
        self.assertEqual(hits, [])
        filtered_log = (
            "SelectMinCost: costTable count=2, opType=3, dataSize=8192.\n"
            "|   0 | AicpuAllReduceSoleMeshOneShot                    |        -1.00 |\n"
            "|   1 | AicpuAllReduceSoleNHRMultiLink                   |        -1.00 |\n"
        )
        hits, has_anchor, nonempty = perf_test.parse_cost_tables(filtered_log)
        self.assertTrue(has_anchor)
        self.assertTrue(nonempty)
        self.assertEqual(hits, [])

    def test_op_type_filter_drops_cross_op_rows(self):
        # op_type 给定时按注册名反分解逐行过滤：可判定为他算子的行丢弃（含引擎前缀才可判定），
        # 反分解失败的行保留；DFX pending 行挂其后最近锚点的档位
        log = (
            "====== [DFX_CostTableDump] algoCount=3 ======\n"
            "  [DFX_CostTableDump] [1/3] algName=AicpuAllGatherSoleMesh, cost=1.000000\n"
            "  [DFX_CostTableDump] [2/3] algName=AicpuAllReduceSoleMesh, cost=0.500000\n"
            "  [DFX_CostTableDump] [3/3] algName=UBXAllReduceXpu, cost=0.800000\n"
            "SelectMinCost: costTable count=2, opType=3, dataSize=8192.\n"
            "|   0 | AicpuAllGatherSoleMesh                            |         1.00 |\n"
            "|   1 | AicpuAllReduceSoleMesh                            |         0.50 |\n"
        )
        hits, has_anchor, _nonempty = perf_test.parse_cost_tables(log, "allreduce")
        self.assertTrue(has_anchor)
        # AicpuAllGather（表格 + DFX 双路）解析出 allgather 被丢；UBX 反解析失败保留挂 8192
        self.assertEqual(hits, [(8192, "AicpuAllReduceSoleMesh", 0.5),
                                (8192, "UBXAllReduceXpu", 0.8)])


class ProfScanProbeLogsTest(unittest.TestCase):
    """scan_probe_logs：slog 目录多文件汇总。"""

    def test_scan_merges_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, "device_0")
            os.makedirs(sub)
            with open(os.path.join(tmp, "plog_1.log"), "w", encoding="utf-8") as handle:
                handle.write("SelectMinCost: costTable count=1, opType=3, dataSize=8192.\n"
                             "|   0 | AicpuAllReduceSoleMeshOneShot | 1.23 |\n")
            with open(os.path.join(sub, "plog_2.log"), "w", encoding="utf-8") as handle:
                handle.write("SelectMinCost: costTable count=1, opType=3, dataSize=8192.\n"
                             "|   0 | AivAllReduceSoleMesh | 2.00 |\n")
            hits, has_anchor, _nonempty = perf_test.scan_probe_logs(tmp)
            self.assertTrue(has_anchor)
            self.assertEqual(len(hits), 2)

    def test_scan_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            hits, has_anchor, nonempty = perf_test.scan_probe_logs(tmp)
            self.assertFalse(has_anchor)
            self.assertFalse(nonempty)
            self.assertEqual(hits, [])


class ProfProbeRoundMarkTest(unittest.TestCase):
    """probe_round_mark：轮级四态标记（失败/无日志/空表/候选全被过滤）。"""

    def test_mark_four_states(self):
        cases = [
            (False, False, False, 0, "[failed]"),
            (True, False, False, 0, "[no costTable log]"),
            (True, True, False, 0, "[costTable empty]"),
            (True, True, True, 0, "[0 valid]"),
            (True, True, True, 3, ""),
            (False, True, True, 2, ""),  # 进程失败但已解析到候选，按有数据算
        ]
        for run_ok, has_anchor, nonempty, hit_count, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    perf_test.probe_round_mark(run_ok, has_anchor, nonempty, hit_count), expected)


class ProfProbeRoundTest(unittest.TestCase):
    """_probe_round：每轮独立日志子目录，隔离跨轮候选行；返回 (hits, has_anchor, nonempty, run_ok)。"""

    def test_round_dir_isolated(self):
        # 子目录按 (op_reduce_dtype_engine) 命名，slog 注入指向子目录，仅本轮行入 hits
        written = {}

        def fake_run_one(cmd, env, timeout):
            round_dir = env["ASCEND_PROCESS_LOG_PATH"]
            written["dir"] = round_dir
            os.makedirs(round_dir, exist_ok=True)
            with open(os.path.join(round_dir, "plog.log"), "w", encoding="utf-8") as handle:
                handle.write("SelectMinCost: costTable count=1, opType=3, dataSize=8192.\n"
                             "|   0 | AivAllReduceSoleMesh | 2.00 |\n")
            return "", "", True

        with tempfile.TemporaryDirectory() as tmp:
            args = mock.Mock(np_total=8, hostfile=None, run_timeout=60, minbytes="1K")
            profile = perf_test.TestProfile(npus=8, minbytes="1K", maxbytes="1M", stepbytes=None,
                                            stepfactor=2, warmup_iters=perf_test.PROBE_WARMUP,
                                            iters=perf_test.PROBE_ITERS)
            with mock.patch.object(perf_test, "run_one", side_effect=fake_run_one):
                hits, has_anchor, nonempty, run_ok = perf_test._probe_round(
                    args, profile, "hccl_test", "aiv", "allreduce", "fp32", "sum", tmp,
                    "mpich", seq=7)
        self.assertEqual(os.path.basename(written["dir"]), "allreduce_sum_fp32_aiv")
        self.assertTrue(has_anchor)
        self.assertTrue(nonempty)
        self.assertTrue(run_ok)
        self.assertEqual(hits, [(8192, "AivAllReduceSoleMesh", 2.0)])

    def test_round_failed_vs_no_anchor(self):
        # run_ok 区分失败轮与无锚点轮：进程失败打 FAILED warning（带 cmd）；
        # 无锚点但进程成功（日志缺失/表为空）不打日志，仅返回值区分，由 probe_round_mark 打标记
        cases = [(False, False, "failed"), (True, False, "no costTable log")]
        for run_ok, has_anchor, mark in cases:
            with self.subTest(mark=mark):
                with tempfile.TemporaryDirectory() as tmp:
                    args = mock.Mock(np_total=8, hostfile=None, run_timeout=60, minbytes="1K")
                    profile = perf_test.TestProfile(npus=8, minbytes="1K", maxbytes="1M",
                                                    stepbytes=None, stepfactor=2,
                                                    warmup_iters=perf_test.PROBE_WARMUP,
                                                    iters=perf_test.PROBE_ITERS)
                    if run_ok:
                        with mock.patch.object(perf_test, "run_one",
                                               return_value=("", "", True)):
                            hits, got_anchor, _nonempty, got_ok = perf_test._probe_round(
                                args, profile, "hccl_test", "aiv", "allreduce", "fp32", "sum",
                                tmp, "mpich", seq=1)
                    else:
                        with mock.patch.object(perf_test, "run_one",
                                               return_value=("", "", False)):
                            with self.assertLogs(perf_test.logger, level="WARNING") as logs:
                                hits, got_anchor, _nonempty, got_ok = perf_test._probe_round(
                                    args, profile, "hccl_test", "aiv", "allreduce", "fp32", "sum",
                                    tmp, "mpich", seq=1)
                self.assertEqual((hits, got_anchor, got_ok), ([], has_anchor, run_ok))
                if not run_ok:
                    self.assertIn("FAILED, fall back to log scan", logs.output[0])

    def test_round_drops_sub_minbytes_sizes(self):
        # 建链期 HCCL 内部控制面操作（barrier/flag）的选行串入：dataSize=4 锚点低于
        # --minbytes，其档位行（含挂到该锚点的 DFX 全量表行）一并丢弃
        def fake_run_one(cmd, env, timeout):
            round_dir = env["ASCEND_PROCESS_LOG_PATH"]
            os.makedirs(round_dir, exist_ok=True)
            with open(os.path.join(round_dir, "plog.log"), "w", encoding="utf-8") as handle:
                handle.write(
                    "====== [DFX_CostTableDump] algoCount=1 ======\n"
                    "  [DFX_CostTableDump] [1/1] algName=AicpuAllReduceSoleMesh, cost=0.900000\n"
                    "SelectMinCost: costTable count=1, opType=3, dataSize=4.\n"
                    "|   0 | AicpuAllReduceSoleMesh                        |         0.90 |\n"
                    "SelectMinCost: costTable count=1, opType=3, dataSize=8192.\n"
                    "|   0 | AicpuAllReduceSoleMesh                        |         1.20 |\n")
            return "", "", True

        with tempfile.TemporaryDirectory() as tmp:
            args = mock.Mock(np_total=8, hostfile=None, run_timeout=60, minbytes="1K")
            profile = perf_test.TestProfile(npus=8, minbytes="1K", maxbytes="1M", stepbytes=None,
                                            stepfactor=2, warmup_iters=perf_test.PROBE_WARMUP,
                                            iters=perf_test.PROBE_ITERS)
            with mock.patch.object(perf_test, "run_one", side_effect=fake_run_one):
                hits, _anchor, _nonempty, _ok = perf_test._probe_round(
                    args, profile, "hccl_test", "aicpu", "allreduce", "fp32", "sum", tmp,
                    "mpich", seq=1)
        self.assertEqual(hits, [(8192, "AicpuAllReduceSoleMesh", 1.2)])

    def test_round_default_minbytes_floor(self):
        # 未传 --minbytes：hccl_test 默认 64M 单点扫描，下限同取 64M，建链期 4B 行仍被丢弃
        with tempfile.TemporaryDirectory() as tmp:
            args = mock.Mock(np_total=8, hostfile=None, run_timeout=60, minbytes=None)
            profile = perf_test.TestProfile(npus=8, minbytes=None, maxbytes=None, stepbytes=None,
                                            stepfactor=None, warmup_iters=perf_test.PROBE_WARMUP,
                                            iters=perf_test.PROBE_ITERS)
            log = ("SelectMinCost: costTable count=1, opType=3, dataSize=4.\n"
                   "|   0 | AicpuAllReduceSoleMesh                        |         0.90 |\n")
            with mock.patch.object(perf_test, "run_one", return_value=(log, "", True)):
                hits, _anchor, _nonempty, _ok = perf_test._probe_round(
                    args, profile, "hccl_test", "aicpu", "allreduce", "fp32", "sum", tmp,
                    "mpich", seq=1)
        self.assertEqual(hits, [])


class ProfSizeCoverageTest(unittest.TestCase):
    """档位参数归一 / 请求档位展开 / 计划前档位覆盖自检。"""

    def test_size_params_normalized(self):
        # "1K"/"1M"/"2" 归一为字节与数值，等价于手写 1024/1048576/2.0
        args = mock.Mock(minbytes="1K", maxbytes="1M", stepbytes=None, stepfactor="2")
        self.assertEqual(perf_test._size_params(args),
                         {"minbytes": 1024, "maxbytes": 1048576,
                          "stepbytes": None, "stepfactor": 2.0})

    def test_expand_requested_sizes(self):
        # 等比：1K→2K→4K；线性：含终点；未给区间返回 None；未给步进仅首档可预知
        self.assertEqual(perf_test.expand_requested_sizes(1024, 4096, None, 2.0),
                         [1024, 2048, 4096])
        self.assertEqual(perf_test.expand_requested_sizes(1024, 3072, 1024, None),
                         [1024, 2048, 3072])
        self.assertIsNone(perf_test.expand_requested_sizes(None, None, None, None))
        self.assertEqual(perf_test.expand_requested_sizes(1024, 4096, None, None), [1024])

    def test_check_size_coverage(self):
        # 分组档位缺 4096 → 一条 WARN 且点名 (op,reduce,dtype)；分组整体缺席 → 全部档位缺失
        payload = make_probe_payload(groups=[{
            "op": "allreduce", "reduce_op": "sum", "data_type": "fp32", "algo_count": 1,
            "algos": [{"index": 0, "algo": "aicpu_sole{mesh}", "reg_name": "AicpuAllReduceSoleMesh"}],
            "sizes": [[1024, [0]], [2048, [0]]]}])
        warns = perf_test.check_size_coverage(payload, ["allreduce"], ["fp32"], ["sum"],
                                              [1024, 2048, 4096])
        self.assertEqual(len(warns), 1)
        self.assertIn("4096", warns[0])
        self.assertIn("(allreduce,sum,fp32)", warns[0])
        # 全覆盖 → 无告警；未给档位（默认序列）→ 不核对
        self.assertEqual(perf_test.check_size_coverage(payload, ["allreduce"], ["fp32"], ["sum"],
                                                       [1024, 2048]), [])
        self.assertEqual(perf_test.check_size_coverage(payload, ["allreduce"], ["fp32"], ["sum"],
                                                       None), [])
        # 超过 5 档缺失时截断展示并带计数
        warns = perf_test.check_size_coverage(payload, ["allreduce"], ["fp32"], ["sum"],
                                              list(range(8)))
        self.assertEqual(len(warns), 1)
        self.assertIn("(+3 more)", warns[0])


class ProfAssembleProbeGroupTest(unittest.TestCase):
    """assemble_probe_group：候选反分解 → 算法池 + 档位索引 / unresolved。"""

    def test_split_pool_sizes_and_unresolved(self):
        # 算法池全档位唯一一份（HCCL_ALGO 写法、不存 cost），sizes 记档位 → 池下标；
        # unresolved 按注册名去重
        hits = [(8192, "AicpuAllReduceSoleMeshOneShot", 1.23),
                (8192, "AivAllReduceSoleMesh", 2.0),
                (8192, "AivAllReduceSoleMesh", 2.0),
                (16384, "AicpuAllReduceSoleMesh", 4.0),
                (16384, "XpuAllReduceSoleMesh", 3.0),
                (16384, "XpuAllReduceSoleMesh", 3.0)]
        group, unresolved = perf_test.assemble_probe_group(hits, "allreduce", "sum", "fp32")
        self.assertEqual(group["op"], "allreduce")
        self.assertEqual(group["reduce_op"], "sum")
        self.assertEqual(group["data_type"], "fp32")
        self.assertEqual([a["algo"] for a in group["algos"]],
                         ["aicpu_sole{meshoneshot}", "aiv_sole{mesh}", "aicpu_sole{mesh}"])
        self.assertEqual([a["index"] for a in group["algos"]], [0, 1, 2])
        self.assertEqual(group["algo_count"], 3)
        # 档位升序、索引引用算法池；8192 命中 OneShot+AIV，16384 仅命中 aicpu mesh
        self.assertEqual(group["sizes"], [[8192, [0, 1]], [16384, [2]]])
        self.assertEqual(len(unresolved), 1)  # 同名重复行去重
        self.assertEqual(unresolved[0]["reg_name"], "XpuAllReduceSoleMesh")

    def test_op_mismatch_goes_unresolved(self):
        # 反分解 op 与本轮探测算子不符（日志混入等）→ 显式记 unresolved
        group, unresolved = perf_test.assemble_probe_group(
            [(8192, "AicpuAllGatherSoleMesh", 1.0)], "allreduce", "sum", "fp32")
        self.assertIsNone(group)
        self.assertEqual(len(unresolved), 1)


class TunerProbeListJsonTest(unittest.TestCase):
    """probe 名单 JSON 读写、拓扑指纹与覆盖面校验。"""

    def test_save_load_roundtrip(self):
        payload = make_probe_payload(groups=[make_probe_group()])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "list.json")
            tuner_common.save_probe_list(path, payload)
            loaded = tuner_common.load_probe_list(path)
        self.assertEqual(loaded["groups"], payload["groups"])
        self.assertEqual(loaded["schema_version"], tuner_common.PROBE_LIST_SCHEMA_VERSION)

    def test_save_sizes_one_line_per_entry(self):
        # sizes 每档位压成单行（size 与下标不换行），且落盘结果可正常读回
        payload = make_probe_payload(groups=[make_probe_group(sizes=[[8192, [0]], [16384, [0]]])])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "list.json")
            tuner_common.save_probe_list(path, payload)
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            loaded = tuner_common.load_probe_list(path)
        self.assertIn("[8192, [0]]", text)
        self.assertIn("[16384, [0]]", text)
        self.assertEqual(loaded["groups"], payload["groups"])

    def test_load_rejects_bad_version_and_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "list.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"schema_version": 999}, handle)
            with self.assertRaises(ValueError):
                tuner_common.load_probe_list(path)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"schema_version": tuner_common.PROBE_LIST_SCHEMA_VERSION, "topo": {}}, handle)
            with self.assertRaises(ValueError):
                tuner_common.load_probe_list(path)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(make_probe_payload(groups=[make_probe_group(algos=[{"algo": "aicpu_sole{mesh}"}])]),
                          handle)
            with self.assertRaises(ValueError):  # 算法项缺 reg_name
                tuner_common.load_probe_list(path)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(make_probe_payload(groups=[make_probe_group(algos=[make_probe_algo(index=1)])]),
                          handle)
            with self.assertRaises(ValueError):  # index 与数组位置不符
                tuner_common.load_probe_list(path)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(make_probe_payload(groups=[make_probe_group(algo_count=2)]), handle)
            with self.assertRaises(ValueError):  # algo_count 与池长度不符
                tuner_common.load_probe_list(path)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(make_probe_payload(groups=[make_probe_group(sizes=[[8192, [0, 5]]])]), handle)
            with self.assertRaises(ValueError):  # 档位索引越界
                tuner_common.load_probe_list(path)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(make_probe_payload(groups=[]), handle)
            with self.assertRaises(ValueError):  # 全空名单拒绝复用，触发重探
                tuner_common.load_probe_list(path)

    def test_topo_fingerprint_matches(self):
        payload = make_probe_payload()
        self.assertEqual(tuner_common.topo_fingerprint(payload["topo"]),
                         tuner_common.topo_fingerprint(payload["topo"]))
        other = dict(payload["topo"], **{"ranks.ranks": 16})
        self.assertNotEqual(tuner_common.topo_fingerprint(payload["topo"]),
                            tuner_common.topo_fingerprint(other))

    def test_probe_list_covers(self):
        payload = make_probe_payload()
        engines = list(perf_test.PROF_ENGINES)
        size_params = {"minbytes": 1024, "maxbytes": 1048576, "stepbytes": None, "stepfactor": 2.0}
        # 需求 ⊆ 探测范围 + 档位参数一致 → 通过；任一维度超出/不一致 → 不通过
        self.assertTrue(tuner_common.probe_list_covers(payload, ["allreduce"], ["sum", "prod"],
                                                       ["fp32", "fp16"], engines, size_params))
        self.assertTrue(tuner_common.probe_list_covers(payload, ["allreduce"], ["sum"], ["fp32"],
                                                       ["aiv"], size_params))
        self.assertFalse(tuner_common.probe_list_covers(payload, ["allreduce"], ["sum", "max"],
                                                        ["fp32"], engines, size_params))
        self.assertFalse(tuner_common.probe_list_covers(payload, ["allreduce"], ["sum"],
                                                        ["fp32", "fp64"], engines, size_params))
        self.assertFalse(tuner_common.probe_list_covers(payload, ["allreduce"], ["sum"], ["fp32"],
                                                        engines + ["host"], size_params))
        # op 维度：名单只探过 allreduce，请求 allgather → 不覆盖（防跨 op 误复用）
        self.assertFalse(tuner_common.probe_list_covers(payload, ["allgather"], ["sum"], ["fp32"],
                                                        engines, size_params))
        # 档位维度：区间不同（上次 1M 这次 2G）→ 不覆盖；步进不同 → 不覆盖
        self.assertFalse(tuner_common.probe_list_covers(
            payload, ["allreduce"], ["sum"], ["fp32"], engines,
            dict(size_params, maxbytes=2 * 1024 ** 3)))
        self.assertFalse(tuner_common.probe_list_covers(
            payload, ["allreduce"], ["sum"], ["fp32"], engines, dict(size_params, stepfactor=4.0)))
        # 未归一的字符串形参不相等：归一由 perf_test._size_params 负责（见 ProfSizeCoverageTest）
        self.assertFalse(tuner_common.probe_list_covers(
            payload, ["allreduce"], ["sum"], ["fp32"], engines,
            {"minbytes": "1K", "maxbytes": "1M", "stepbytes": None, "stepfactor": 2}))
        # 旧名单缺 op_types / size_params 字段 → 不覆盖，触发重探
        payload["probe_params"].pop("op_types")
        self.assertFalse(tuner_common.probe_list_covers(payload, ["allreduce"], ["sum"], ["fp32"],
                                                        engines, size_params))
        payload["probe_params"]["op_types"] = ["allreduce"]
        payload["probe_params"].pop("size_params")
        self.assertFalse(tuner_common.probe_list_covers(payload, ["allreduce"], ["sum"], ["fp32"],
                                                        engines, size_params))


class ProfProbeReuseTest(unittest.TestCase):
    """load_or_run_probe 自动复用：匹配即复用，缺失/不匹配/损坏自动重探。"""

    def _run(self, payload_on_disk, topo=None, reduce_ops=("sum",), data_types=("fp32",),
             engines=("aicpu",)):
        """落盘指定名单后调用 load_or_run_probe，返回 (结果, run_probe 调用 Mock)。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "list.json")
            if payload_on_disk is not None:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(payload_on_disk, handle)
            args = mock.Mock(probe_output=path, minbytes="1K", maxbytes="1M",
                             stepbytes=None, stepfactor=2)
            topo = topo if topo is not None else make_probe_payload()["topo"]
            with mock.patch.object(perf_test, "run_probe") as run_probe_mock:
                run_probe_mock.return_value = "PROBED"
                result = perf_test.load_or_run_probe(args, ["allreduce"], list(reduce_ops),
                                                     list(data_types), list(engines), {}, topo, "mpich")
        return result, run_probe_mock

    def test_reuse_when_match(self):
        # 拓扑指纹一致 + 覆盖面满足 → 复用名单（磁盘回读的新对象），不触发探测
        payload = make_probe_payload(groups=[make_probe_group()])
        result, run_probe_mock = self._run(payload)
        self.assertEqual(result, payload)
        self.assertIsNot(result, payload)
        run_probe_mock.assert_not_called()

    def test_reprobe_when_missing(self):
        result, run_probe_mock = self._run(None)
        self.assertEqual(result, "PROBED")
        run_probe_mock.assert_called_once()

    def test_reprobe_when_topo_mismatch(self):
        payload = make_probe_payload(groups=[make_probe_group()])
        other_topo = dict(payload["topo"], **{"ranks.ranks": 16})
        result, run_probe_mock = self._run(payload, topo=other_topo)
        self.assertEqual(result, "PROBED")
        run_probe_mock.assert_called_once()

    def test_reprobe_when_dimension_short(self):
        # 请求的 reduce/引擎/算子任一维度超出名单探测范围 → 自动重探
        cases = [
            dict(reduce_ops=("sum", "max")),                            # reduce 维度超范围
            dict(payload_engines=["aicpu"], engines=("aicpu", "aiv")),  # 引擎维度超范围
            dict(payload_ops=["alltoall"]),                             # 算子维度超范围
        ]
        for case in cases:
            payload = make_probe_payload()
            if "payload_engines" in case:
                payload["probe_params"]["engines"] = case["payload_engines"]
            if "payload_ops" in case:
                payload["probe_params"]["op_types"] = case["payload_ops"]
            result, run_probe_mock = self._run(
                payload, reduce_ops=case.get("reduce_ops", ("sum",)),
                engines=case.get("engines", ("aicpu",)))
            self.assertEqual(result, "PROBED", msg=str(case))
            run_probe_mock.assert_called_once()

    def test_reprobe_when_corrupt(self):
        # schema 版本不符 / 结构残缺按不可用处理，自动重探
        result, run_probe_mock = self._run({"schema_version": 999})
        self.assertEqual(result, "PROBED")
        run_probe_mock.assert_called_once()
        result, run_probe_mock = self._run({"schema_version": tuner_common.PROBE_LIST_SCHEMA_VERSION, "topo": {}})
        self.assertEqual(result, "PROBED")
        run_probe_mock.assert_called_once()


class ProfMakeCollectPlanTest(unittest.TestCase):
    """make_collect_plan + build_run_combos：probe 名单 ∩ CLI 维度。"""

    def test_union_semantics_and_dedupe(self):
        # 并集口径：同算法多 dtype / 多档位条目去重为一组组合
        payload = make_probe_payload(groups=[
            make_probe_group(),
            make_probe_group(reduce_op="prod"),
            make_probe_group(data_type="fp16", sizes=[[16384, [0]]]),
            make_probe_group(algos=[make_probe_algo(algo="aiv_sole{mesh}",
                                                    reg_name="AivAllReduceSoleMesh")]),
        ])
        plan = perf_test.make_collect_plan(payload, ["allreduce"], ["aicpu", "aiv"],
                                           ["fp32"], ["sum"], None)
        self.assertEqual(plan.valid_by_op_engine[("allreduce", "aicpu")], [("sole", ("mesh",))])
        self.assertEqual(plan.valid_by_op_engine[("allreduce", "aiv")], [("sole", ("mesh",))])

    def test_explicit_algos_intersection(self):
        payload = make_probe_payload(groups=[
            make_probe_group(),
            make_probe_group(algos=[make_probe_algo(algo="aicpu_sole{meshoneshot}",
                                                    reg_name="AicpuAllReduceSoleMeshOneShot")]),
        ])
        plan = perf_test.make_collect_plan(payload, ["allreduce"], ["aicpu"], ["fp32"],
                                           ["sum"], [("sole", ["meshoneshot"])])
        self.assertEqual(plan.valid_by_op_engine[("allreduce", "aicpu")],
                         [("sole", ("meshoneshot",))])

    def test_supported_keys_full_dim(self):
        # 行级校验键含 reduce_op/dtype/size 完整维度；同池多档位经索引解出
        payload = make_probe_payload(groups=[
            make_probe_group(sizes=[[8192, [0]], [16384, [0]]]),
        ])
        plan = perf_test.make_collect_plan(payload, ["allreduce"], ["aicpu"], ["fp32"], ["sum"], None)
        self.assertEqual(plan.supported_keys, {
            ("allreduce", "aicpu", "sole", ("mesh",), "sum", "fp32", 8192),
            ("allreduce", "aicpu", "sole", ("mesh",), "sum", "fp32", 16384),
        })

    def test_build_run_combos_reduce_dim(self):
        # reduce 类算子按 reduce_ops 展开；非 reduce 类固定 NA 不乘
        payload = make_probe_payload(groups=[
            make_probe_group(),
            make_probe_group(reduce_op="prod"),
            make_probe_group(op="broadcast", reduce_op="NA",
                             algos=[make_probe_algo(algo="dpu_sequence{mesh,nhr}",
                                                    reg_name="DpuBroadcastSequenceMeshNHR")]),
        ])
        plan = perf_test.make_collect_plan(payload, ["allreduce", "broadcast"], ["aicpu", "dpu"],
                                           ["fp32"], ["sum", "prod"], None)
        combos = perf_test.build_run_combos(plan)
        allreduce_combos = [c for c in combos if c[0] == "allreduce"]
        broadcast_combos = [c for c in combos if c[0] == "broadcast"]
        # allreduce: 2 reduce_ops × 1 aicpu 组合；broadcast: 1（NA）× 1 dpu 组合
        self.assertEqual(len(allreduce_combos), 2)
        self.assertEqual(len(broadcast_combos), 1)
        self.assertEqual(broadcast_combos[0][2], "NA")
        self.assertEqual(len(combos), 3)

    def test_empty_plan(self):
        plan = perf_test.make_collect_plan(make_probe_payload(), ["allreduce"], ["aicpu"],
                                           ["fp32"], ["sum"], None)
        self.assertEqual(plan.valid_combos, 0)
        self.assertEqual(perf_test.build_run_combos(plan), [])


class ProfRowCheckResultTest(unittest.TestCase):
    """_row_check_result：行级回退校验（并集采集下不在名单的档位标 fallback）。"""

    SUPPORTED = {("allreduce", "aicpu", "sole", ("mesh",), "sum", "fp32", 8192)}

    def _ctx(self):
        return perf_test.RunContext(
            args=None, profile=None, exe_paths={}, hccl_buffsize="", env_keys=[], topo={},
            total_runs=1, mpi_flavor="mpich", supported_keys=self.SUPPORTED)

    def test_supported_key_keeps_check(self):
        combo = ("allreduce", "fp32", "sum", "aicpu", "sole", ("mesh",))
        self.assertEqual(perf_test._row_check_result(self._ctx(), combo, 8192, "success"), "success")

    def test_unsupported_size_marked_fallback(self):
        # 同算法在 16384 档位不在名单（opCustomCheck 过滤）→ 该行标 fallback
        combo = ("allreduce", "fp32", "sum", "aicpu", "sole", ("mesh",))
        self.assertEqual(perf_test._row_check_result(self._ctx(), combo, 16384, "success"), "fallback")

    def test_failed_check_not_overridden(self):
        # failed 行保留原状（审计口径优先于回退标记）
        combo = ("allreduce", "fp32", "sum", "aicpu", "sole", ("mesh",))
        self.assertEqual(perf_test._row_check_result(self._ctx(), combo, 16384, "failed"), "failed")

    def test_reduce_op_and_dtype_dim_matched(self):
        # reduce_op / dtype 维度不同即不在名单
        combo = ("allreduce", "fp32", "prod", "aicpu", "sole", ("mesh",))
        self.assertEqual(perf_test._row_check_result(self._ctx(), combo, 8192, "success"), "fallback")
        combo = ("allreduce", "fp16", "sum", "aicpu", "sole", ("mesh",))
        self.assertEqual(perf_test._row_check_result(self._ctx(), combo, 8192, "success"), "fallback")


class ProfRowCheckWildcardTest(unittest.TestCase):
    """_row_check_result 的 size=0 通配：DFX-only 名单（档位未知）按算法存在性放行。"""

    SUPPORTED = {("allreduce", "aicpu", "sole", ("mesh",), "sum", "fp32", 0)}

    def _ctx(self):
        return perf_test.RunContext(
            args=None, profile=None, exe_paths={}, hccl_buffsize="", env_keys=[], topo={},
            total_runs=1, mpi_flavor="mpich", supported_keys=self.SUPPORTED)

    def test_wildcard_size_zero_keeps_check(self):
        combo = ("allreduce", "fp32", "sum", "aicpu", "sole", ("mesh",))
        self.assertEqual(perf_test._row_check_result(self._ctx(), combo, 8192, "success"), "success")
        self.assertEqual(perf_test._row_check_result(self._ctx(), combo, 999999, "success"), "success")

    def test_wildcard_no_match_still_fallback(self):
        combo = ("allreduce", "fp32", "prod", "aicpu", "sole", ("mesh",))
        self.assertEqual(perf_test._row_check_result(self._ctx(), combo, 8192, "success"), "fallback")


if __name__ == "__main__":
    unittest.main(verbosity=2)
