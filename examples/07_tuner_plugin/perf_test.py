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

"""Tuner Plugin 算法数据采集脚本。

采集前先跑 probe 阶段：按 (op × reduce_op × dtype) 分组，组内按引擎分轮拉起
hccl_test（逐引擎下发 -a 圈定候选，dpu 轮不下发，不设 HCCL_ALGO，-w 0 -n 1），从
slog 日志的 costTable 收集当前环境实际支持的算法名单并落盘 JSON；再按名单 ∩ CLI
维度生成采集计划，逐轮拉起 hccl_test 解析 stdout 表格落 CSV，供 optimize_config.py
生成最优配置。探测依赖 -a（仅 950 系有效），910B 暂不支持。

拓扑前提：仅 level0=MESH_1D 或 960 系设备走新选择器（有 costTable 日志），其余拓扑
解析不到锚点会 fail-fast。

用法示例：
    python3 perf_test.py --op-types allreduce --engines aicpu,aiv \\
        --minbytes 1K --maxbytes 1M --stepfactor 2 --data-types fp32 \\
        --reduce-ops sum,prod \\
        --np-total 16 --npus 8 --hostfile hostfile --output hccl_perf.csv
"""

import argparse
import csv
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import namedtuple

from tuner_common import (
    CSV_FIELDS, DEVICE_ENGINES, ENGINE_ALL, EXECUTOR_TYPES, OP_TYPES,
    PROBE_LIST_SCHEMA_VERSION, RANKS_FIELDS, TEMPLATE_TYPES,
    algo_to_env, load_probe_list, make_algo_string, now_timestamp,
    parse_algo_string, parse_algos, probe_list_covers, save_probe_list, split_algo_name, topo_fingerprint,
)

# ===== 本脚本特有常量 =====

# op_type → hccl_test 可执行文件映射（./bin/ 下）
OP_TO_EXE = {
    "allreduce": "all_reduce_test",
    "allgather": "all_gather_test",
    "broadcast": "broadcast_test",
    "reduce": "reduce_test",
    "reduce_scatter": "reduce_scatter_test",
    "scatter": "scatter_test",
    "alltoall": "alltoall_test",
    "alltoallv": "alltoallv_test",
}

# data_type 采集域（hccl_test -d 全部 16 种）
DATA_TYPES = (
    "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
    "fp16", "fp32", "fp64", "bfp16", "fp8e5m2", "fp8e4m3", "fp8e8m0", "hif8",
)

# reduce 类算子（hccl_test -o 仅对这些算子有效）
REDUCE_OP_TYPES = ("allreduce", "reduce", "reduce_scatter")
REDUCE_OPS = ("sum", "prod", "max", "min")

# engine → hccl_test -a 下发值（常量表集中维护；dpu 无 -a 值，不下发）
ENGINE_TO_ACCEL = {
    "aicpu": "aicpu_ts",
    "aiv": "aiv_only",
    "ccums": "ccu_ms",
    "ccusched": "ccu_sched",
}

# 采集侧引擎白名单：probe 按引擎分轮探测后全量放开，ccu 引擎采集轮按 -a 单引擎
# 下发不回退；dpu 由 accel=None 分支处理
PROF_ENGINES = ("aicpu", "aiv", "dpu", "ccums", "ccusched")

# ===== probe 探测参数 =====

# 展开模式是单引擎开关（950 系 hccl_test 在 HCCL_OP_EXPANSION_MODE 未设置时强制
# ccu_sched，不存在"一轮拿全引擎"），probe 按引擎分轮下发 -a 圈定候选；dpu 轮不下发
# -a（DPU 场景默认展示备选；950 上该轮实际探到 ccusched 候选，按注册名归类无害）。
# 探测依赖 -a（仅 950 系有效），910B 暂不支持。
PROBE_WARMUP = 0
PROBE_ITERS = 1
# probe 轮 slog 注入：INFO 级（costTable 为 HCCL_INFO 打印），落专用目录；存原值恢复
PROBE_LOG_LEVEL = "1"
PROBE_ENV_KEYS = ("ASCEND_GLOBAL_LOG_LEVEL", "ASCEND_PROCESS_LOG_PATH")

# hccl_test 对 allgather 的 -b/-e 是总量（每卡输入 = 总量/rank_size），HCCL costTable
# 锚点 dataSize 记每卡输入；命中档位需 ×rank_size 换算回请求口径再过滤/落盘，否则会被
# 下限过滤误杀（reduce_scatter/scatter 虽也均分，但 HCCL inputSize 已乘回总量，无需换算）
PROBE_RESCALE_OPS = ("allgather",)

# hccl_test stdout 表头列名（去掉单位后缀）→ CSV 字段映射
# 注：实际二进制打印 "aveg_time(us):"（hccl_opbase_rootinfo_base.h 源码拼写），
#     文档写 "avg_time(us):"，两者都收。
STDOUT_COL_MAP = {
    "data_size": "size_bytes",
    "avg_time": "alg_latency(us)",
    "aveg_time": "alg_latency(us)",
    "alg_bandwidth": "alg_bandwidth(GB/s)",
    "check_result": "check_result",
}

MIN_PYTHON = (3, 8)

# hccl_test 相对 $ASCEND_HOME_PATH 的默认子目录
DEFAULT_BIN_SUBDIR = "tools/hccl_test/bin"

MPIRUN_MISSING_HINT = (
    "mpirun not found in PATH. Please install MPICH 4.1.3 "
    "(https://www.mpich.org/downloads/) or Open MPI 4.1.5 "
    "(https://www.open-mpi.org/software/ompi/v4.1/).")
HCCL_TEST_MISSING_HINT = (
    "hccl_test executable not found: {path}. Please compile hccl_test from CANN "
    "source (oam-tools src/hccl_test) and place binaries under "
    "$ASCEND_HOME_PATH/tools/hccl_test/bin/."
)
ASCEND_HOME_MISSING_HINT = (
    "ASCEND_HOME_PATH not set. Please source the CANN environment script first, "
    "e.g. 'source ${INSTALL_DIR}/bin/setenv.bash', so hccl_test can be located "
    "at $ASCEND_HOME_PATH/tools/hccl_test/bin/."
)

# 模块级 logger：main() 内 basicConfig 统一配置
logger = logging.getLogger("perf_test")


class ProfTestError(Exception):
    """采集脚本参数 / 环境错误。"""


class PreflightError(ProfTestError):
    """启动自检（preflight）失败。"""


# ===== 纯函数（L0 单测入口）=====

def parse_bytes(spec):
    """解析数据量字符串，支持 K/M/G 后缀（大小写不敏感）与小数。

    "1K" -> 1024，"4M" -> 4194304，"2G" -> 2147483648，"512" -> 512，"1.5K" -> 1536。
    """
    spec = str(spec).strip()
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([kKmMgG]?)", spec)
    if match is None:
        raise ValueError("invalid bytes spec: {!r}".format(spec))
    num = float(match.group(1))
    suffix = match.group(2).lower()
    multiplier = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}[suffix]
    return int(round(num * multiplier))


def expand_list(value, valid, name):
    """展开逗号分隔 CLI 列表；支持 all；逐项校验白名单。"""
    value = (value or "").strip()
    if not value or value == "all":
        return list(valid)
    items = [v.strip() for v in value.split(",") if v.strip()]
    for item in items:
        if item not in valid:
            raise ValueError("invalid {} {!r} (valid: {})".format(name, item, ",".join(valid)))
    return items


def build_algos(algos_arg, executors_arg, templates_arg):
    """算法组合双入口：--algos 优先透传解析；否则 --executors × --templates 笛卡尔积。

    三参数均未指定时返回 None：由 make_collect_plan 直接采用 probe 名单全量组合，
    避免单模板笛卡尔积静默剪掉多级算法。
    """
    if algos_arg:
        return parse_algos(algos_arg)
    if executors_arg is None and templates_arg is None:
        return None
    executors = expand_list(executors_arg, EXECUTOR_TYPES, "executor")
    templates = expand_list(templates_arg, TEMPLATE_TYPES, "template")
    return [(executor, [tpl]) for executor in executors for tpl in templates]


def engine_to_accelerator(engine):
    """engine 规范名 → hccl_test -a 下发值；dpu 返回 None（不走 -a）。"""
    if engine not in ENGINE_ALL:
        raise ValueError("invalid engine {!r} (valid: {})".format(engine, ",".join(ENGINE_ALL)))
    return ENGINE_TO_ACCEL.get(engine)


def reduce_type_for_op(op_type, reduce_op):
    """reduce 类算子返回 reduce_op，其余固定 NA（CSV reduce_type 字段口径）。"""
    return reduce_op if op_type in REDUCE_OP_TYPES else "NA"


def detect_mpi_env():
    """探测当前进程 MPI 环境，返回 (rank, size)；无 MPI 环境返回 (0, 1)。

    Open MPI: OMPI_COMM_WORLD_RANK/SIZE；MPICH/PMI: PMI_RANK/PMI_SIZE。
    """
    for rank_key, size_key in (("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
                               ("PMI_RANK", "PMI_SIZE")):
        if rank_key in os.environ and size_key in os.environ:
            return int(os.environ[rank_key]), int(os.environ[size_key])
    return 0, 1


def flavor_from_mpi_path(mpirun_path):
    """mpirun 真实路径 → flavor：路径含 openmpi 或指向 orterun → openmpi，其余 mpich。"""
    real = os.path.realpath(mpirun_path)
    if "openmpi" in real.lower() or os.path.basename(real).lower() == "orterun":
        return "openmpi"
    return "mpich"


def detect_mpi_flavor():
    """MPI 类别探测：OMPI 变量优先（覆盖 mpirun -np 1 拉起场景），否则读 mpirun
    真实路径（发行版符号链接会暴露实现名），均未命中按 mpich 兜底。"""
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        return "openmpi"
    mpirun = shutil.which("mpirun")
    return flavor_from_mpi_path(mpirun) if mpirun else "mpich"


def check_mpi_env():
    """MPI 环境监测：返回 flavor 供命令构造，并拦截嵌套运行。

    脚本自身已在 MPI 会话内（size>1）时抛错——此时再拉 mpirun 会按 rank 数
    放大采集进程数，采集无法收敛。必须在会话外（登录节点）运行本脚本。
    """
    rank, size = detect_mpi_env()
    if size > 1:
        raise PreflightError(
            "perf_test.py is running inside an MPI session (rank {} of {}); run it "
            "outside mpirun, otherwise each round launches {} nested MPI ranks".format(
                rank, size, size))
    return detect_mpi_flavor()


def _hostfile_npus_of(line):
    """解析单行 hostfile 的每机 NPU 数，无法识别返回 None。

    OpenMPI 语法取 slots=N；MPICH 语法取首 token 的 ":N" 后缀（前段不含冒号才认，
    规避裸 IPv6 地址尾段误判）。纯 IP / 主机名行返回 None，跳过校验。
    """
    tokens = line.split("#", 1)[0].split()
    if not tokens:
        return None
    for token in tokens[1:]:
        if token.startswith("slots="):
            suffix = token[len("slots="):]
            return int(suffix) if suffix.isdigit() else None
    head, sep, tail = tokens[0].rpartition(":")
    if sep and ":" not in head and tail.isdigit():
        return int(tail)
    return None


def count_hostfile_lines(hostfile):
    """hostfile 非空行数（派生 servers；注释 # 与空行忽略）。

    同时校验均质组网：各行可识别的 NPU 数（MPICH 的 IP:N / OpenMPI 的 slots=N）
    互不一致时启动即报错——异构组网会静默推导出错误 -p，产出整批无效数据。
    """
    servers = 0
    counts = []
    with open(hostfile, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            servers += 1
            npus = _hostfile_npus_of(line)
            if npus is not None:
                counts.append((servers, npus))
    if len({npus for _, npus in counts}) > 1:
        detail = ", ".join("line {}: {} NPUs".format(no, npus) for no, npus in counts)
        raise ValueError(
            "heterogeneous hostfile {}: per-server NPU counts differ ({}); "
            "homogeneous networking is required".format(hostfile, detail))
    return servers


def derive_topo(np_total, npus, hostfile, pods, super_pods):
    """派生 ranks.* 拓扑 5 字段：ranks / npus_per_server / servers / pods / super_pods。

    npus 未指定时按均质组网推导 np_total // servers（servers 来自 hostfile 行数，
    单机为 1）；除不尽时无法确定每 server 卡数，要求显式指定 --npus。
    """
    servers = count_hostfile_lines(hostfile) if hostfile else 1
    if npus is None:
        if np_total % servers != 0:
            raise ValueError(
                "cannot derive npus per server: np_total={} not divisible by servers={}; "
                "specify --npus explicitly".format(np_total, servers))
        npus = np_total // servers
    values = (np_total, npus, servers, pods, super_pods)
    return dict(zip(RANKS_FIELDS, values))


# ===== costTable 日志解析（probe 真相源）=====

# 锚点行（selector_engine.cc SelectMinCost 打印，dataSize 标识本张表所属数据量档位）
_COST_ANCHOR_RE = re.compile(r"SelectMinCost: costTable count=(\d+), opType=(-?\d+), dataSize=(\d+)")
# 候选表行：| idx | algName | cost |（行尾锚定，天然排除后续 5 列 selected 表与表头）
_COST_ROW_RE = re.compile(r"\|\s*(\d+)\s*\|\s*([A-Za-z][A-Za-z0-9]{1,48})\s*\|\s*(-?\d+(?:\.\d+)?)\s*\|\s*$")
# DFX 锚点/数据行（cost_table.cc DumpCostTable 打印，构建期先于 SelectMinCost）：
#   ====== [DFX_CostTableDump] algoCount=%d ======
#     [DFX_CostTableDump] [%d/%d] algName=%s, cost=%.6f
_DFX_ANCHOR_RE = re.compile(r"====== \[DFX_CostTableDump\] algoCount=(\d+) ======")
_DFX_ROW_RE = re.compile(r"\[DFX_CostTableDump\] \[\d+/\d+\] algName=(\S+), cost=(-?\d+(?:\.\d+)?)")


def parse_cost_tables(text, op_type=None):
    """从日志文本提取 costTable 候选行，返回 (hits, has_anchor, nonempty_anchor)。

    hits 为 [(size_bytes, reg_name, cost)]，仅收 cost>=0 行（负值=不适用候选，
    selector 选 minCost 时本就跳过）。两路关键字：
    - SelectMinCost 锚点带 dataSize，其后的表格行直接按锚点档位归档；
    - DFX_CostTableDump 行在构建期先于锚点出现且自身无 dataSize，先缓存、
      挂到其后最近锚点的档位，全程无锚点可挂的记 size=0（档位未知，
      行级校验按算法存在性通配放行）。
    has_anchor 表示日志出现过任一 costTable 锚点；nonempty_anchor 表示存在
    count>0 的锚点（区分"表为空"与"有候选但全被过滤"，排查方向不同）。
    多 rank 打印的重复行按 (size_bytes, reg_name) 去重。
    op_type 给定时按注册名反分解逐行过滤：可判定为其他算子的行直接丢弃
    （构建期全量表串入的噪声），反分解失败的行保留、交由分组阶段归 unresolved。
    """
    hits, has_anchor, nonempty_anchor = [], False, False
    seen = set()
    current_size = None  # 最近 SelectMinCost 锚点的 dataSize
    pending_dfx = []  # [(reg_name, cost)]，等待其后最近锚点配对

    def keep(reg_name):
        if op_type is None:
            return True
        parsed = split_algo_name(reg_name)
        return parsed is None or parsed[0] == op_type

    def emit(size_bytes, reg_name, cost):
        if cost < 0 or not keep(reg_name):
            return
        key = (size_bytes, reg_name)
        if key not in seen:
            seen.add(key)
            hits.append((size_bytes, reg_name, cost))

    for line in text.splitlines():
        anchor = _COST_ANCHOR_RE.search(line)
        if anchor:
            has_anchor = True
            nonempty_anchor = nonempty_anchor or int(anchor.group(1)) > 0
            current_size = int(anchor.group(3))
            for reg_name, cost in pending_dfx:
                emit(current_size, reg_name, cost)
            pending_dfx = []
            continue
        dfx_anchor = _DFX_ANCHOR_RE.search(line)
        if dfx_anchor:
            has_anchor = True
            nonempty_anchor = nonempty_anchor or int(dfx_anchor.group(1)) > 0
            current_size = None  # 新一轮 costTable 构建，旧表格窗口关闭
            continue
        dfx = _DFX_ROW_RE.search(line)
        if dfx:
            pending_dfx.append((dfx.group(1), float(dfx.group(2))))
            continue
        row = _COST_ROW_RE.search(line)
        if row is None or current_size is None:
            continue
        emit(current_size, row.group(2), float(row.group(3)))
    for reg_name, cost in pending_dfx:
        emit(0, reg_name, cost)
    return hits, has_anchor, nonempty_anchor


def scan_probe_logs(log_dir, op_type=None):
    """扫描 probe 日志目录全部文件（slog 按 rank/device 分文件），汇总 costTable 候选。"""
    hits, has_anchor, nonempty_anchor = [], False, False
    for root, _dirs, files in os.walk(log_dir):
        for name in files:
            try:
                with open(os.path.join(root, name), "r", encoding="utf-8", errors="ignore") as handle:
                    text = handle.read()
            except OSError:
                continue
            file_hits, file_anchor, file_nonempty = parse_cost_tables(text, op_type)
            hits += file_hits
            has_anchor = has_anchor or file_anchor
            nonempty_anchor = nonempty_anchor or file_nonempty
    return hits, has_anchor, nonempty_anchor


def assemble_probe_group(raw_hits, op_type, reduce_op, data_type):
    """costTable 候选反分解并聚成一个分组（算法池 + 档位索引），返回 (group, unresolved)。

    group 为 {op, reduce_op, data_type, algo_count, algos[], sizes[]}：algos 按
    HCCL_ALGO 写法（make_algo_string 合成，如 "aicpu_sole{mesh}"）首次出现顺序入池，
    池项带显式 index（=数组位置，与 sizes 引用对齐）；sizes 为 [数据量, [算法池下标...]]
    升序列表，档位间只重复编号不重复内容；无有效
    候选时 group 为 None。注册名无法反分解（UBX/Pcie 等非标后缀）或反分解 op 与
    本轮不符的条目记入 unresolved（按注册名去重，显式可见，默认不进采集计划）。
    """
    pool, pool_index = [], {}  # 算法池（去重）；algo 串 -> 池内下标
    size_algos = {}  # size_bytes -> {algo 串}
    unresolved, unresolved_seen = [], set()
    for size_bytes, reg_name, _cost in raw_hits:
        split = split_algo_name(reg_name)
        if split is None:
            if reg_name not in unresolved_seen:
                unresolved_seen.add(reg_name)
                unresolved.append({"reg_name": reg_name,
                                   "reason": "unparsable or out of collect domain"})
            continue
        algo_op, engine, executor, templates = split
        if algo_op != op_type:
            if reg_name not in unresolved_seen:
                unresolved_seen.add(reg_name)
                unresolved.append({"reg_name": reg_name,
                                   "reason": "op mismatch: parsed={} round={}".format(algo_op, op_type)})
            continue
        algo_str = make_algo_string(engine, executor, templates)
        if algo_str not in pool_index:
            pool_index[algo_str] = len(pool)
            pool.append({"index": len(pool), "algo": algo_str, "reg_name": reg_name})
        size_algos.setdefault(size_bytes, set()).add(algo_str)
    if not pool:
        return None, unresolved
    group = {
        "op": op_type, "reduce_op": reduce_op, "data_type": data_type,
        "algo_count": len(pool), "algos": pool,
        "sizes": [[size, sorted(pool_index[algo] for algo in size_algos[size])]
                  for size in sorted(size_algos)],
    }
    return group, unresolved


# ===== preflight 启动自检 =====

def check_python_version(version_info=None):
    """Python 版本检查：>= 3.8，不满足报错退出。"""
    version_info = version_info or sys.version_info
    if version_info < MIN_PYTHON:
        raise PreflightError(
            "Python >= {}.{} required, got {}.{}. Please upgrade Python.".format(
                MIN_PYTHON[0], MIN_PYTHON[1], version_info[0], version_info[1]))
    return True


def resolve_bin_dir(bin_dir=None, env=None):
    """hccl_test bin 目录 = $ASCEND_HOME_PATH + 相对路径拼接。

    bin_dir 为相对子目录（默认 tools/hccl_test/bin）；绝对路径时直接使用。
    ASCEND_HOME_PATH 未设置抛 PreflightError。
    """
    sub_dir = bin_dir or DEFAULT_BIN_SUBDIR
    if os.path.isabs(sub_dir):
        return sub_dir
    home = (os.environ if env is None else env).get("ASCEND_HOME_PATH", "").strip()
    if not home:
        raise PreflightError(ASCEND_HOME_MISSING_HINT)
    return os.path.join(home, sub_dir)


def exe_path_of(op_type, bin_dir=None, env=None):
    """op_type → hccl_test 可执行文件完整路径（$ASCEND_HOME_PATH 拼接）。"""
    return os.path.join(resolve_bin_dir(bin_dir, env), OP_TO_EXE[op_type])


def preflight(op_types, bin_dir=None, env=None):
    """启动自检：Python 版本 / mpirun 存在 / MPI 环境 / hccl_test 可执行文件存在。

    任一关键项不满足即抛 PreflightError（调用方报错退出）；不触发任何下载。
    通过时返回 MPI flavor（"openmpi"/"mpich"）。脚本仅用 Python 标准库。
    """
    check_python_version()
    if shutil.which("mpirun") is None:
        raise PreflightError(MPIRUN_MISSING_HINT)
    flavor = check_mpi_env()
    for op_type in op_types:
        exe_path = exe_path_of(op_type, bin_dir, env)
        if not os.path.isfile(exe_path):
            raise PreflightError(HCCL_TEST_MISSING_HINT.format(path=exe_path))
    return flavor


# ===== CSV 写 =====

class CsvWriter:
    """CSV 追加写：新文件写表头，已有文件续写（表头校验防混排）。

    增量采集多轮 run 追加同一 CSV，optimize_config 侧按 timestamp 去重。
    """

    def __init__(self, path):
        need_header = not os.path.exists(path) or os.path.getsize(path) == 0
        self._handle = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        if need_header:
            self._writer.writerow(CSV_FIELDS)
        else:
            self._ensure_trailing_newline(path)
            self._check_existing_header(path)

    def _ensure_trailing_newline(self, path):
        """追加前检查末字节：非换行则补一个，避免续写行与旧行粘连成脏行。

        常见于上一轮进程被强杀（kill -9 / 超时清理）导致最后一行未完整落盘。
        """
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            last_byte = handle.read(1)
        if last_byte != b"\n":
            self._handle.write("\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def write_row(self, row):
        self._writer.writerow([row.get(field, "") for field in CSV_FIELDS])

    def close(self):
        self._handle.close()

    def _check_existing_header(self, path):
        """已有文件表头必须与当前 17 字段完全一致，否则拒绝追加（防异构混排）。"""
        with open(path, newline="", encoding="utf-8") as handle:
            existing = next(csv.reader(handle), None)
        if existing != list(CSV_FIELDS):
            self._handle.close()
            raise ValueError(
                "existing CSV header mismatch: {}. Expected {} fields {}. Remove/rename the "
                "file or use a new --output path.".format(
                    path, len(CSV_FIELDS), ",".join(CSV_FIELDS)))


# ===== CLI 与主流程 =====

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tuner Plugin 算法数据采集：probe 算法名单 + mpirun + hccl_test 采集落 CSV")
    parser.add_argument("--op-types", required=True,
                        help="算子白名单 8 种，逗号分隔多值，支持 all")
    parser.add_argument("--minbytes", default=None, help="数据量区间起始（K/M/G 后缀），对应 -b")
    parser.add_argument("--maxbytes", default=None, help="数据量区间结束（K/M/G 后缀），对应 -e")
    parser.add_argument("--stepbytes", type=int, default=None, help="线性步进（Bytes），对应 -i，与 --stepfactor 互斥")
    parser.add_argument("--stepfactor", type=float, default=None, help="等比因子，对应 -f，与 --stepbytes 互斥")
    parser.add_argument("--data-types", default="fp32", help="dtype 16 种，逗号分隔多值，默认 fp32，对应 -d")
    parser.add_argument("--reduce-ops", default="sum",
                        help="reduce 类算子操作类型，逗号分隔多值（sum,prod），默认 sum，对应 -o；"
                             "probe 按该维度展开，非 reduce 类算子不受影响")
    parser.add_argument("--engines", default="all",
                        help="采集引擎（aicpu/aiv/dpu/ccums/ccusched），all=全部")
    parser.add_argument("--algos", default=None,
                        help="算法串（HCCL_ALGO 语义，';' 分隔，如 sole{mesh};parallel{mesh,nhr}），"
                             "与 probe 名单求交集")
    parser.add_argument("--executors", default=None,
                        help="快捷交集入口，默认用名单全量 executor（--algos 存在时忽略）")
    parser.add_argument("--templates", default=None,
                        help="快捷交集入口，默认用名单全量 template（--algos 存在时忽略）")
    parser.add_argument("--warmup-iters", type=int, default=10, help="预热次数，默认 10，对应 -w")
    parser.add_argument("--iters", type=int, default=20, help="迭代次数，默认 20，对应 -n")
    parser.add_argument("--run-timeout", type=int, default=600, metavar="SECONDS",
                        help="单轮 mpirun 超时秒数，默认 600；超时杀进程树，该轮落 failed 并继续采集")
    parser.add_argument("--pods", type=int, default=0,
                        help="pod 数（拓扑字段，用户指定）；0=未指定（默认）")
    parser.add_argument("--super-pods", type=int, default=0,
                        help="super pod 数（拓扑字段，用户指定）；0=未指定（默认）")
    parser.add_argument("--hostfile", default=None, help="多机 hostfile 文件（派生 servers），单机省略")
    parser.add_argument("--npus", type=int, default=None,
                        help="每 server NPU 数，对应 hccl_test -p（必传）；"
                             "缺省时按 np_total/servers 推导，除不尽需显式指定")
    parser.add_argument("--np-total", type=int, required=True, help="总 rank 数（派生 ranks），对应 mpirun -n")
    parser.add_argument("--output", default="hccl_perf.csv", help="CSV 输出路径（追加模式）")
    parser.add_argument("--probe-output", default=None,
                        help="probe 算法名单 JSON 路径，默认 <output>.algos.json；"
                             "已有名单且拓扑/参数匹配时自动复用，删除该文件可强制重探")
    parser.add_argument("--bin-dir", default=DEFAULT_BIN_SUBDIR,
                        help="hccl_test 可执行文件目录：相对 $ASCEND_HOME_PATH 的子目录"
                             "（默认 tools/hccl_test/bin），也可给绝对路径")
    return parser.parse_args(argv)


def validate_args(args):
    """CLI 参数白名单校验：op-type / engine / dtype / reduce-op / 区间 / 步进互斥。"""
    op_types = expand_list(args.op_types, OP_TYPES, "op_type")
    # v 系列（allgatherv / reducescatterv / alltoallvc）不在白名单，expand_list 直接报错
    if args.engines.strip() == "all":
        engines = list(PROF_ENGINES)
    else:
        engines = expand_list(args.engines, PROF_ENGINES, "engine")
    data_types = expand_list(args.data_types, DATA_TYPES, "data_type")
    reduce_ops = expand_list(args.reduce_ops, REDUCE_OPS, "reduce_op")
    if args.stepbytes is not None and args.stepfactor is not None:
        raise ValueError("--stepbytes and --stepfactor are mutually exclusive")
    if (args.minbytes is None) != (args.maxbytes is None):
        raise ValueError("--minbytes and --maxbytes must be given together")
    if args.minbytes is not None:
        if parse_bytes(args.minbytes) > parse_bytes(args.maxbytes):
            raise ValueError("--minbytes > --maxbytes")
    if args.hostfile and not os.path.isfile(args.hostfile):
        raise ValueError("hostfile not found: {}".format(args.hostfile))
    return op_types, engines, data_types, reduce_ops


def format_duration(seconds):
    """秒数 → 人读时长："1h 02m 03s" / "02m 03s" / "12.3s"（<60s 保留 1 位小数）。"""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return "{:.1f}s".format(seconds)
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "{}h {:02d}m {:02d}s".format(h, m, s)
    return "{:02d}m {:02d}s".format(m, s)


def run_one(cmd, env, timeout=600):
    """执行单条 mpirun 命令，返回 (stdout 文本, stderr 文本, 是否成功)。

    超时（默认 600s）时杀掉整个进程组（mpirun 起独立进程组，远端 rank 子进程
    随之终止），该轮按失败处理并继续后续采集，避免单轮挂死阻塞整个采集流程。
    """
    kwargs = {"env": env, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
              "universal_newlines": True}
    if os.name == "posix":
        kwargs["start_new_session"] = True  # 独立进程组，超时可整组清理
    proc = subprocess.Popen(cmd, **kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return stdout, stderr, proc.returncode == 0
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        logger.warning("[perf_test] WARN run timeout ({}s), killed: {}".format(timeout, " ".join(cmd)))
        return "", "[perf_test] run timeout after {}s: {}".format(timeout, " ".join(cmd)), False


def _kill_process_tree(proc):
    """超时清理：杀 mpirun 所在进程组（含远端 rank）。"""
    try:
        if os.name == "posix":
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()


def tail_lines(text, limit=15):
    """取文本末尾若干非空行（失败轮 WARN 直接携带报错原因，如缺 so / dlopen 失败）。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def dump_raw_output(output_path, tag, stdout, stderr):
    """每轮原始输出取证 dump：stdout/stderr 全文追加写 <output>.raw。"""
    raw_path = output_path + ".raw"
    with open(raw_path, "a", encoding="utf-8") as raw:
        raw.write("===== {} =====\n".format(tag))
        raw.write("----- stdout -----\n{}\n".format(stdout))
        raw.write("----- stderr -----\n{}\n".format(stderr))
    return raw_path


# ===== probe 探测阶段 =====

def _probe_env_setup(log_dir):
    """probe 轮环境注入：INFO 级 slog 落专用目录 + 清 HCCL_ALGO / HCCL_OP_EXPANSION_MODE。

    ASCEND_PROCESS_LOG_PATH 指向 log_dir（scan_probe_logs 扫描同一目录）；清
    HCCL_ALGO（残留会让 costTable 被 HCCL_ALGO 收窄）与 HCCL_OP_EXPANSION_MODE
    （语义同 -a，残留会把候选收敛到单引擎）。OpenMPI 的 -x 从父进程环境读值，故
    直接写 os.environ；返回原值快照供 finally 恢复。
    """
    saved = {key: os.environ.get(key)
             for key in PROBE_ENV_KEYS + ("HCCL_ALGO", "HCCL_OP_EXPANSION_MODE")}
    os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = PROBE_LOG_LEVEL
    os.environ["ASCEND_PROCESS_LOG_PATH"] = log_dir
    os.environ.pop("HCCL_ALGO", None)
    os.environ.pop("HCCL_OP_EXPANSION_MODE", None)
    return saved


def _probe_env_restore(saved):
    """恢复 probe 前的环境变量快照（原无则删除）。"""
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def probe_round_mark(run_ok, has_anchor, nonempty_anchor, hit_count):
    """轮级异常标记四态：失败 / 无 costTable 日志 / 空表 / 候选全被过滤，排查方向不同。"""
    if not has_anchor:
        return "[failed]" if not run_ok else "[no costTable log]"
    if hit_count == 0:
        return "[costTable empty]" if not nonempty_anchor else "[0 valid]"
    return ""


def merge_probe_hits(hits, op_type, rank_size, min_bytes):
    """三路命中汇总去重并按扫描下限过滤，返回 [(size_bytes, reg_name, cost)]。

    allgather 档位先 ×rank_size 换算回请求口径（锚点 dataSize 是每卡输入）；低于
    扫描下限的档位是 HCCL 建链期内部控制面操作（barrier/flag）的选型日志串入，丢弃；
    未传 --minbytes 时 hccl_test 按 64M 单点扫（hccl_test_common.h 默认值），同用作
    下限；size=0 通配档位（无锚点 DFX 行）保留，是 fallback 判定的合法数据。
    """
    rescale = rank_size if op_type in PROBE_RESCALE_OPS else 1
    seen, merged = set(), []
    for size_bytes, reg_name, cost in hits:
        size_bytes *= rescale
        key = (size_bytes, reg_name)
        if key not in seen:
            seen.add(key)
            if size_bytes == 0 or size_bytes >= min_bytes:
                merged.append((size_bytes, reg_name, cost))
    return merged


def _probe_round(args, profile, exe_path, engine, op_type, data_type, reduce_op, log_dir, mpi_flavor,
                 seq=0):
    """拉起单轮 probe hccl_test，返回 (hits, has_anchor, nonempty_anchor, run_ok)。

    参数与采集轮对齐（-b/-e/-i/-f、-d、-o），按引擎下发 -a 圈定候选（dpu 无 -a 值
    不下发，默认展开）；不设 HCCL_ALGO；每轮用独立日志子目录（共享目录 + 全量重扫
    会把历史轮次行读进本轮，跨算子显形为 op mismatch、同算子跨 dtype 静默污染
    sizes），stdout/stderr 同样解析兜底。run_ok 区分"进程失败"与"无锚点"，分组
    汇总行据此打四态标记（probe_round_mark）。
    """
    accel = ENGINE_TO_ACCEL.get(engine)
    test_args = build_test_args(profile, op_type, data_type, reduce_op, accel=accel)
    round_dir = os.path.join(log_dir, "{}_{}_{}_{}".format(op_type, reduce_op, data_type, engine))
    os.makedirs(round_dir, exist_ok=True)
    cmd = build_mpirun_cmd(args.np_total, args.hostfile, [exe_path] + test_args,
                           env_keys=list(PROBE_ENV_KEYS), flavor=mpi_flavor)
    if seq == 1:
        # 首轮打完整命令（含 Open MPI 自动补齐参数），后续轮仅测试参数不同
        logger.info("[perf_test]     cmd: {}".format(" ".join(cmd)))
    env = os.environ.copy()
    env["ASCEND_PROCESS_LOG_PATH"] = round_dir
    stdout, stderr, run_ok = run_one(cmd, env, timeout=args.run_timeout)
    if not run_ok:
        logger.warning("[perf_test] probe #{} FAILED, fall back to log scan\n  cmd: {}".format(
            seq, " ".join(cmd)))
    hits, has_anchor, nonempty = parse_cost_tables(stdout, op_type)
    more_hits, more_anchor, more_nonempty = parse_cost_tables(stderr, op_type)
    hits += more_hits
    has_anchor = has_anchor or more_anchor
    nonempty = nonempty or more_nonempty
    # slog 落文件是主通路（stdout 未开重定向），只扫本轮子目录兜住全部 rank 日志
    log_hits, log_anchor, log_nonempty = scan_probe_logs(round_dir, op_type)
    hits += log_hits
    has_anchor = has_anchor or log_anchor
    nonempty = nonempty or log_nonempty
    # stdout/stderr 与日志文件三路汇总后去重（同 key 保留首个）+ 下限过滤（含 allgather 档位换算）
    min_bytes = parse_bytes(args.minbytes) if args.minbytes else 64 * 1024 * 1024
    return merge_probe_hits(hits, op_type, args.np_total, min_bytes), has_anchor, nonempty, run_ok


def run_probe(args, op_types, reduce_ops, data_types, engines, exe_paths, topo, mpi_flavor):
    """探测当前环境支持的算法名单，落盘 JSON 并返回 payload。

    每 (op × reduce_op × dtype) 一组，组内按引擎分轮下发 -a 圈定候选（dpu 轮不下发），
    各轮 hits 合并后聚成一个分组；每轮打印与采集轮同构的配置行 + 结果行（轮级四态
    异常标记、稳态 ETA），组末一行聚合；全部轮次收不到 costTable 锚点才 fail-fast
    （仅 level0=MESH_1D 或 960 系设备走新选择器打印 costTable）。
    """
    profile = TestProfile(npus=args.npus, minbytes=args.minbytes, maxbytes=args.maxbytes,
                          stepbytes=args.stepbytes, stepfactor=args.stepfactor,
                          warmup_iters=PROBE_WARMUP, iters=PROBE_ITERS)
    probe_log_dir = os.path.abspath(args.probe_output + ".logs")
    if os.path.isdir(probe_log_dir):
        shutil.rmtree(probe_log_dir)
    os.makedirs(probe_log_dir)
    groups, unresolved = [], []
    any_anchor = False
    first_run_cost = None  # 首轮耗时（含环境冷启动），ETA 稳态外推时剔除
    seq = 0
    total_rounds = 0
    for op_type in op_types:
        total_rounds += len(engines) * len(data_types) * (
            len(reduce_ops) if op_type in REDUCE_OP_TYPES else 1)
    # ETA 稳态外推基于 monotonic 时钟（time.time 与 monotonic 相减为无意义大负数）
    probe_started = time.monotonic()
    logger.info("[perf_test] === probe start: {} rounds, logs {} ===".format(
        total_rounds, probe_log_dir))
    saved = _probe_env_setup(probe_log_dir)
    try:
        for op_type in op_types:
            for reduce_op in (reduce_ops if op_type in REDUCE_OP_TYPES else ["NA"]):
                for data_type in data_types:
                    # 非 reduce 算子无 reduce_op，配置行与聚合行不打印该字段
                    if op_type in REDUCE_OP_TYPES:
                        group_label = "{} {} {}".format(op_type, reduce_op, data_type)
                    else:
                        group_label = "{} {}".format(op_type, data_type)
                    group_hits, group_anchor = [], False
                    for engine in engines:
                        seq += 1
                        accel = ENGINE_TO_ACCEL.get(engine)
                        accel_cell = " accel={}".format(accel) if accel else ""
                        # 每轮配置行与采集轮同构，先打再拉起（长运行期间日志可见）
                        if op_type in REDUCE_OP_TYPES:
                            logger.info("[perf_test] [{}/{}] op={} dtype={} reduce_op={} "
                                        "engine={}{}".format(
                                            seq, total_rounds, op_type, data_type, reduce_op,
                                            engine, accel_cell))
                        else:
                            logger.info("[perf_test] [{}/{}] op={} dtype={} engine={}{}".format(
                                seq, total_rounds, op_type, data_type, engine, accel_cell))
                        round_start = time.monotonic()
                        hits, has_anchor, nonempty, run_ok = _probe_round(
                            args, profile, exe_paths[op_type], engine, op_type, data_type,
                            reduce_op, probe_log_dir, mpi_flavor, seq)
                        duration = time.monotonic() - round_start
                        # ETA 稳态均耗时外推（剔除首轮冷启动），首轮无基准不打
                        eta = "--"
                        if first_run_cost is None:
                            first_run_cost = duration
                        else:
                            steady_avg = (time.monotonic() - probe_started - first_run_cost) / (seq - 1)
                            eta = format_duration(steady_avg * (total_rounds - seq))
                        group_hits += hits
                        group_anchor = group_anchor or has_anchor
                        hit_sizes = {size_bytes for size_bytes, _reg, _cost in hits}
                        mark = probe_round_mark(run_ok, has_anchor, nonempty, len(hits))
                        logger.info("[perf_test]     -> {} hits / {} sizes{} ({}, ETA {})".format(
                            len(hits), len(hit_sizes), mark, format_duration(duration), eta))
                    any_anchor = any_anchor or group_anchor
                    round_group, round_unresolved = assemble_probe_group(group_hits, op_type,
                                                                         reduce_op, data_type)
                    if round_group is not None:
                        groups.append(round_group)
                    unresolved += round_unresolved
                    logger.info("[perf_test] probe {} -> {} algos / {} sizes{}".format(
                        group_label,
                        round_group["algo_count"] if round_group is not None else 0,
                        len(round_group["sizes"]) if round_group is not None else 0,
                        ", {} unresolved".format(len(round_unresolved)) if round_unresolved else ""))
                    # 分组级只报异常：有效候选全被过滤（unresolved 明细看名单 JSON）
                    if group_anchor and round_group is None:
                        logger.warning("[perf_test] WARN probe found no valid candidate: {} "
                                       "(all candidates filtered or unparsable)".format(group_label))
    finally:
        _probe_env_restore(saved)
    if not any_anchor:
        raise ProfTestError(
            "no costTable log found under {}. Selector costTable is only printed on "
            "level0=MESH_1D or 960-series topology; also check ASCEND_PROCESS_LOG_PATH "
            "reachability on every node and ASCEND_GLOBAL_LOG_LEVEL>=INFO.".format(probe_log_dir))
    if not groups:
        # 全空名单没有采集价值，落盘后还会被复用校验放行；直接报错暴露，逐轮标记见上方日志
        raise ProfTestError(
            "probe finished but every group is empty (all candidates filtered or unparsable); "
            "check the per-round marks ([costTable empty]/[0 valid]) and unresolved entries in "
            "logs under {}".format(probe_log_dir))
    payload = {
        "schema_version": PROBE_LIST_SCHEMA_VERSION,
        "probe_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topo": topo,
        "probe_params": {"op_types": op_types, "reduce_ops": reduce_ops,
                         "data_types": data_types, "engines": engines,
                         "size_params": _size_params(args)},
        "groups": groups,
        "unresolved": unresolved,
    }
    save_probe_list(args.probe_output, payload)
    logger.info("[perf_test] probe done: {} groups, {} unresolved -> {} (took {}s)".format(
        len(groups), len(unresolved), args.probe_output, int(time.monotonic() - probe_started)))
    return payload


def _size_params(args):
    """档位参数归一（字节/数值）：复用校验用，"1K" 与 "1024" 归一后相等。"""
    return {
        "minbytes": parse_bytes(args.minbytes) if args.minbytes is not None else None,
        "maxbytes": parse_bytes(args.maxbytes) if args.maxbytes is not None else None,
        "stepbytes": int(args.stepbytes) if args.stepbytes is not None else None,
        "stepfactor": float(args.stepfactor) if args.stepfactor is not None else None,
    }


def expand_requested_sizes(minbytes, maxbytes, stepbytes, stepfactor):
    """按 -b/-e/-i/-f 语义展开请求档位（与 hccl_test 扫点一致）。

    未给 --minbytes 返回 None（hccl_test 默认序列不可预知）；给了区间但未给步进时
    仅首档可预知，返回单元素列表做最小核对。
    """
    if minbytes is None:
        return None
    sizes, cur = [], int(minbytes)
    while cur <= maxbytes:
        sizes.append(cur)
        if stepbytes is not None:
            cur += int(stepbytes)
        elif stepfactor is not None:
            cur = max(cur + 1, int(cur * stepfactor))
        else:
            break
    return sizes


def check_size_coverage(probe_payload, op_types, data_types, reduce_ops, requested_sizes):
    """计划前自检：请求档位在某 (op, reduce_op, dtype) 分组无任何算法时生成 WARN 文案。

    返回文案列表（空列表 = 全覆盖）；典型根因是该档位候选被 HCCL customCheck
    全量过滤，对应采集行将标 fallback。
    """
    if not requested_sizes:
        return []
    groups = {(g["op"], g["reduce_op"], g["data_type"]): {s for s, _ in g["sizes"]}
              for g in probe_payload["groups"]}
    warnings = []
    for op_type in op_types:
        for reduce_op in (reduce_ops if op_type in REDUCE_OP_TYPES else ["NA"]):
            for data_type in data_types:
                covered = groups.get((op_type, reduce_op, data_type), set())
                missing = [s for s in requested_sizes if s not in covered]
                if missing:
                    shown = ", ".join(str(s) for s in missing[:5])
                    suffix = " (+{} more)".format(len(missing) - 5) if len(missing) > 5 else ""
                    warnings.append("sizes {}{} have no supported algorithm for "
                                    "({},{},{})".format(shown, suffix, op_type, reduce_op, data_type))
    return warnings


def load_or_run_probe(args, op_types, reduce_ops, data_types, engines, exe_paths, topo, mpi_flavor):
    """名单获取：已有名单且拓扑指纹/参数覆盖面匹配则复用，否则自动重新探测并覆盖落盘。"""
    if os.path.isfile(args.probe_output):
        try:
            payload = load_probe_list(args.probe_output)
        except (OSError, ValueError) as err:
            logger.warning("[perf_test] WARN probe list unusable ({}), re-probing".format(err))
        else:
            fp_file, fp_cur = topo_fingerprint(payload["topo"]), topo_fingerprint(topo)
            if fp_file != fp_cur:
                logger.warning("[perf_test] WARN probe list topo mismatch: file={} current={}, "
                               "re-probing".format(fp_file, fp_cur))
            elif not probe_list_covers(payload, op_types, reduce_ops, data_types, engines,
                                       _size_params(args)):
                logger.warning(
                    "[perf_test] WARN probe list params do not cover request, re-probing\n"
                    "  list   : {}\n"
                    "  request: op_types={} reduce_ops={} data_types={} engines={} "
                    "size_params={}".format(payload.get("probe_params"), op_types, reduce_ops,
                                            data_types, engines, _size_params(args)))
            else:
                logger.info("[perf_test] probe list reused (generated at {}): {} groups -> {}".format(
                    payload.get("probe_time", "unknown"), len(payload["groups"]), args.probe_output))
                return payload
    return run_probe(args, op_types, reduce_ops, data_types, engines, exe_paths, topo, mpi_flavor)


# ===== 采集计划与执行 =====

# 采集轮固定参数（整个采集过程不变，来自 CLI；与每轮变化的 op/dtype/reduce_op/engine 分离）
TestProfile = namedtuple("TestProfile", [
    "npus", "minbytes", "maxbytes", "stepbytes", "stepfactor",
    "warmup_iters", "iters",
])

# 采集计划：维度输入 + 名单交集结果（make_collect_plan 的返回契约）
CollectPlan = namedtuple("CollectPlan", [
    "op_types", "data_types", "reduce_ops", "engines",
    "valid_by_op_engine", "valid_combos", "total_runs", "supported_keys",
])

# 采集执行环境：CLI args + 固定参数 + 路径/拓扑/MPI 环境（每轮共享）
RunContext = namedtuple("RunContext", [
    "args", "profile", "exe_paths", "hccl_buffsize", "env_keys", "topo", "total_runs",
    "mpi_flavor", "supported_keys",
])


def make_test_profile(args):
    """CLI args → TestProfile（固定参数打包）。"""
    return TestProfile(npus=args.npus, minbytes=args.minbytes, maxbytes=args.maxbytes,
                       stepbytes=args.stepbytes, stepfactor=args.stepfactor,
                       warmup_iters=args.warmup_iters, iters=args.iters)


def build_test_args(profile, op_type, data_type, reduce_op, accel=None):
    """构造 hccl_test 可执行文件参数（exe 之后部分）。

    profile 为轮固定参数（TestProfile）；op_type/data_type/reduce_op 为本轮维度。
    -b/-e 传原始字符串（hccl_test 接受 K/M/G 后缀）；-i/-f 二选一；
    -o 仅 reduce 类算子下发；accel 为 -a 下发值（None 不下发，probe/采集各自决定）。
    """
    args = []
    if profile.npus:
        args += ["-p", str(profile.npus)]
    if profile.minbytes is not None:
        args += ["-b", profile.minbytes]
    if profile.maxbytes is not None:
        args += ["-e", profile.maxbytes]
    if profile.stepbytes is not None:
        args += ["-i", str(profile.stepbytes)]
    elif profile.stepfactor is not None:
        args += ["-f", str(profile.stepfactor)]
    args += ["-d", data_type]
    if op_type in REDUCE_OP_TYPES:
        args += ["-o", reduce_op]
    args += ["-w", str(profile.warmup_iters), "-n", str(profile.iters)]
    if accel is not None:
        args += ["-a", accel]
    return args


# Open MPI 远端不继承首节点环境，官方要求显式 -x 透传的环境变量清单（hccl_test 文档）
OPENMPI_FORWARD_ENV_KEYS = (
    "ASCEND_HOME_PATH", "LD_LIBRARY_PATH",
    "HCCL_SOCKET_FAMILY", "HCCL_SOCKET_IFNAME", "HCCL_CONNECT_TIMEOUT", "HCCL_BUFFSIZE",
)

# Open MPI 默认安装路径前缀（这些值下 Open MPI 可自行定位，不必加 --prefix）
_DEFAULT_MPI_PREFIXES = ("", "/", "/usr", "/usr/local")


def derive_mpi_prefix():
    """从 mpirun 真实路径推导 Open MPI 安装前缀（<prefix>/bin/mpirun）；默认路径返回 None。"""
    mpirun = shutil.which("mpirun")
    if not mpirun:
        return None
    real = os.path.realpath(mpirun)
    if "/bin/" not in real:
        return None
    prefix = real.rsplit("/bin/", 1)[0]
    return None if prefix in _DEFAULT_MPI_PREFIXES else prefix


def build_mpirun_cmd(np_total, hostfile, exe_argv, env_keys=None, flavor=None):
    """构造 mpirun 命令（MPICH / Open MPI 双支持）。

    exe_argv 为被拉起进程的完整 argv（exe 路径 + 参数）。
    MPICH:   mpirun [-f hostfile] -n N <exe> ...（全环境随进程转发）
    OpenMPI: mpirun [--prefix P] [-hostfile hostfile] -n N [-x ENV ...]
             [--allow-run-as-root] [--mca opal_set_max_sys_limits 1] <exe> ...
    OpenMPI 远端不继承首节点环境：自动补齐 --prefix（非默认安装路径时）、官方
    清单与 env_keys 的 -x 透传（仅透传当前有值的变量，去重）、root 豁免与系统
    限制放开；网卡指定类参数（--mca btl_tcp_if_include）不自动生成。
    """
    flavor = flavor or detect_mpi_flavor()
    cmd = ["mpirun"]
    if flavor == "openmpi":
        prefix = derive_mpi_prefix()
        if prefix:
            cmd += ["--prefix", prefix]
    if hostfile:
        cmd += (["-hostfile", hostfile] if flavor == "openmpi" else ["-f", hostfile])
    cmd += ["-n", str(np_total)]
    if flavor == "openmpi":
        forward = []
        for key in OPENMPI_FORWARD_ENV_KEYS + tuple(env_keys or ()):
            if key not in forward and os.environ.get(key):
                forward.append(key)
        for key in forward:
            cmd += ["-x", key]
        if hasattr(os, "getuid") and os.getuid() == 0:
            cmd += ["--allow-run-as-root"]
        cmd += ["--mca", "opal_set_max_sys_limits", "1"]
    cmd += exe_argv
    return cmd


_ITER_RE = re.compile(r"iters\s+is\s+(\d+).*warmup_iters\s+is\s+(\d+)")


def parse_header_info(text):
    """解析 stdout 首行回显 '... iters is 20, warmup_iters is 5' → (iters, warmup_iters)。

    用于校验 -w/-n 是否真正下发生效（hccl_test 回显实际生效值）。
    找不到回显行返回 None。
    """
    for line in text.splitlines():
        match = _ITER_RE.search(line)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _norm_col(name):
    """表头列名归一化：去空白 / 冒号，截掉 "(单位)" 后缀，转小写。"""
    name = name.strip().rstrip(":").strip()
    paren = name.find("(")
    if paren > 0:
        name = name[:paren]
    return name.strip().lower()


def _parse_header_row(cells):
    """尝试把 '|' 分隔的 cells 解析为表头；是表头返回列映射，否则 None。

    须至少包含 data_size 列才认作表头行。
    """
    mapped = [(STDOUT_COL_MAP.get(_norm_col(cell)), cell) for cell in cells]
    return mapped if any(field == "size_bytes" for field, _ in mapped) else None


def _parse_data_row(columns, cells):
    """按表头列映射把 cells 解析为数据行 dict；解析失败返回 None。

    size_bytes 取 int，alg_latency / alg_bandwidth 取 float（非法值整行丢弃）。
    """
    row = {}
    for (field, _), cell in zip(columns, cells):
        if field is None:
            continue
        if field == "size_bytes":
            try:
                row[field] = int(cell)
            except ValueError:
                return None
        elif field in ("alg_latency(us)", "alg_bandwidth(GB/s)"):
            try:
                row[field] = float(cell)
            except ValueError:
                return None
        else:
            row[field] = cell
    return row if "size_bytes" in row else None


def parse_stdout(text):
    """解析 hccl_test stdout 表格，返回行字典列表。

    按表头列名匹配（非固定索引，兼容列序差异）；data_size/avg_time/alg_bandwidth/
    check_result 四项落 dict；首行汇总信息、表头行、空行、列数不符行忽略。
    """
    rows = []
    columns = None  # [(csv_field or None, 原始列名), ...]
    for line in text.splitlines():
        if "|" not in line:
            continue  # 汇总行 / 空行 / 提示信息忽略
        cells = [cell.strip() for cell in line.split("|")]
        if columns is None:
            columns = _parse_header_row(cells)
            continue
        if len(cells) != len(columns):
            continue  # 列数不匹配的行（如日志前缀）忽略
        row = _parse_data_row(columns, cells)
        if row is not None:
            rows.append(row)
    return rows


def parse_run_output(stdout, stderr):
    """从 mpirun 输出解析数据行：stdout 优先，无行时回退 stderr。

    部分 mpirun 实现 / 重定向场景下 hccl_test 表格会落到 stderr，
    两个流用同一解析器（列数不匹配的噪声行天然被忽略）。
    """
    rows = parse_stdout(stdout)
    if not rows:
        rows = parse_stdout(stderr)
    return rows


def make_collect_plan(probe_payload, op_types, engines, data_types, reduce_ops, algos):
    """probe 名单 ∩ CLI 维度生成采集计划。

    - valid_by_op_engine：{(op, engine): [(executor, tuple(templates)), ...]}，
      名单并集口径（算法在任一 reduce_op/dtype/size 通过即采），保序去重；
    - algos 非 None 时再与显式算法集求交；
    - supported_keys：(op, engine, executor, templates, reduce_op, dtype, size) 集合，
      采集行级校验用（不在名单的行按 fallback 标记）；
    - total_runs：实际执行轮数（build_run_combos 展开后），全被剪时为 0。
    """
    valid_by_op_engine = {}
    for op_type in op_types:
        for engine in engines:
            valid, seen = [], set()
            for group in probe_payload["groups"]:
                if group["op"] != op_type:
                    continue
                for algo in group["algos"]:
                    algo_engine, executor, templates = parse_algo_string(algo["algo"])
                    if algo_engine != engine:
                        continue
                    key = (executor, tuple(templates))
                    if key in seen or key[0] not in EXECUTOR_TYPES:
                        continue
                    seen.add(key)
                    valid.append(key)
            valid_by_op_engine[(op_type, engine)] = valid
    if algos is not None:
        allow = {(executor, tuple(templates)) for executor, templates in algos}
        valid_by_op_engine = {key: [v for v in values if v in allow]
                              for key, values in valid_by_op_engine.items()}
    valid_combos = sum(len(values) for values in valid_by_op_engine.values())
    supported_keys = set()
    for group in probe_payload["groups"]:
        parsed = {algo["index"]: parse_algo_string(algo["algo"]) for algo in group["algos"]}
        for size_bytes, indices in group["sizes"]:
            for idx in indices:
                algo_engine, executor, templates = parsed[idx]
                supported_keys.add((group["op"], algo_engine, executor, tuple(templates),
                                    group["reduce_op"], group["data_type"], size_bytes))
    # total_runs 在 build_run_combos 后回填（reduce_op 维度按算子类别展开）
    return CollectPlan(op_types=op_types, data_types=data_types, reduce_ops=reduce_ops,
                       engines=engines, valid_by_op_engine=valid_by_op_engine,
                       valid_combos=valid_combos, total_runs=0, supported_keys=supported_keys)


def build_run_combos(plan):
    """展开实际执行的采集组合：[(op, dtype, reduce_op, engine, executor, templates), ...]。

    reduce_op 仅对 reduce 类算子展开（其余 NA），与 hccl_test -o 语义一致。
    """
    combos = []
    for op_type, data_type in itertools.product(plan.op_types, plan.data_types):
        for reduce_op in (plan.reduce_ops if op_type in REDUCE_OP_TYPES else ["NA"]):
            for engine in plan.engines:
                for executor, templates in plan.valid_by_op_engine[(op_type, engine)]:
                    combos.append((op_type, data_type, reduce_op, engine, executor, templates))
    return combos


def log_collect_plan(ctx, plan, probe_payload, total_runs):
    """打印采集计划（测试范围总览）：维度、名单统计、拓扑、迭代数与输出模式。"""
    args = ctx.args
    logger.info("[perf_test] ================= collect plan =================")
    logger.info("[perf_test] op_types   : {}".format(",".join(plan.op_types)))
    logger.info("[perf_test] data_types : {}".format(",".join(plan.data_types)))
    logger.info("[perf_test] reduce_ops : {} (reduce ops only: {})".format(
        ",".join(plan.reduce_ops),
        ",".join(op for op in plan.op_types if op in REDUCE_OP_TYPES) or "none"))
    logger.info("[perf_test] engines    : {}".format(",".join(plan.engines)))
    # 每算子 × 引擎的名单组合数（值为 0 表示该引擎该算子整组跳过）
    skipped = []
    for op_type in plan.op_types:
        counts = ", ".join("{}={}".format(engine, len(plan.valid_by_op_engine[(op_type, engine)]))
                           for engine in plan.engines)
        logger.info("[perf_test] valid algos : {:<14} {}".format(op_type, counts))
        skipped += ["{}[{}]".format(op_type, engine) for engine in plan.engines
                    if not plan.valid_by_op_engine[(op_type, engine)]]
    if skipped:
        logger.warning("[perf_test] WARN skipped (no candidates): {}".format(", ".join(skipped)))
    unresolved = probe_payload.get("unresolved", [])
    if unresolved:
        logger.info("[perf_test] unresolved : {} names excluded (e.g. {})".format(
            len(unresolved), ", ".join(item["reg_name"] for item in unresolved[:5])))
    sp = _size_params(args)
    for warn in check_size_coverage(probe_payload, plan.op_types, plan.data_types, plan.reduce_ops,
                                    expand_requested_sizes(sp["minbytes"], sp["maxbytes"],
                                                           sp["stepbytes"], sp["stepfactor"])):
        logger.warning("[perf_test] WARN {} , rows will be fallback".format(warn))
    size_desc = "{}..{}".format(args.minbytes, args.maxbytes) if args.minbytes else "default"
    step_desc = ("stepbytes {}".format(args.stepbytes) if args.stepbytes is not None
                 else "stepfactor {}".format(args.stepfactor) if args.stepfactor is not None
                 else "hccl_test default step")
    logger.info("[perf_test] size range : {} ({})".format(size_desc, step_desc))
    logger.info("[perf_test] topo       : ranks={} npus/server={} servers={} pods={} super_pods={}".format(
        ctx.topo["ranks.ranks"], ctx.topo["ranks.npus_per_server"], ctx.topo["ranks.servers"],
        ctx.topo["ranks.pods"], ctx.topo["ranks.super_pods"]))
    logger.info("[perf_test] mpi        : {} (mpirun command flavor)".format(ctx.mpi_flavor))
    logger.info("[perf_test] iters      : warmup={} iters={}".format(args.warmup_iters, args.iters))
    if ctx.hccl_buffsize:
        logger.info("[perf_test] env        : HCCL_BUFFSIZE={}".format(ctx.hccl_buffsize))
    logger.info("[perf_test] total runs : {}".format(total_runs))
    out_mode = ("append" if os.path.exists(args.output) and os.path.getsize(args.output) > 0 else "new")
    logger.info("[perf_test] output     : {} ({})".format(args.output, out_mode))
    logger.info("[perf_test] =================================================")


def _base_csv_row(ctx, combo):
    """构造单轮 CSV 行的公共字段（成功行覆写性能值，失败行覆写 check_result）。"""
    op_type, data_type, reduce_op, engine, executor, templates = combo
    return {
        "op_type": op_type,
        "size_bytes": "",
        "data_type": data_type,
        "reduce_type": reduce_type_for_op(op_type, reduce_op),
        "engine": engine,
        "algorithm.executor_type": executor,
        "algorithm.template_type": ",".join(templates),
        "HCCL_BUFFSIZE": ctx.hccl_buffsize,
        **ctx.topo,
    }


def _exec_round(ctx, combo, run_idx, elapsed_total, first_run_cost):
    """拉起单轮 mpirun 采集并解析，返回 (rows, stdout, stderr, ok, duration)。

    含本轮配置/cmd/进度日志、-w/-n 回显校验与 ETA 预估。
    first_run_cost：首轮实际耗时，尚无基准时为 None（run_idx==1 时由本轮耗时回填）。
    """
    op_type, data_type, reduce_op, engine, executor, templates = combo
    args = ctx.args
    algo_str = algo_to_env(executor, templates)
    # 先打本轮配置再拉起 mpirun（长时间运行期间日志可见）
    logger.info("[perf_test] [{}/{}] op={} dtype={} reduce_op={} engine={} algo={}".format(
        run_idx, ctx.total_runs, op_type, data_type, reduce_op, engine, algo_str))
    test_args = build_test_args(ctx.profile, op_type, data_type, reduce_op,
                                accel=engine_to_accelerator(engine))
    cmd = build_mpirun_cmd(args.np_total, args.hostfile,
                           [ctx.exe_paths[op_type]] + test_args, env_keys=ctx.env_keys,
                           flavor=ctx.mpi_flavor)
    env = os.environ.copy()
    env["HCCL_ALGO"] = algo_str
    start = time.monotonic()
    stdout, stderr, ok = run_one(cmd, env, timeout=args.run_timeout)
    duration = time.monotonic() - start
    # 完整命令打印（排查 -w/-n 等参数是否真的下发）
    logger.info("[perf_test]     cmd: HCCL_ALGO={} {}".format(algo_str, " ".join(cmd)))
    rows = parse_run_output(stdout, stderr) if ok else []
    # 校验 -w/-n 生效：hccl_test 首行回显实际生效值，与 CLI 配置不符即 WARN
    echo_iters = parse_header_info(stdout) or parse_header_info(stderr)
    if echo_iters is not None and echo_iters != (args.iters, args.warmup_iters):
        logger.warning("[perf_test] WARN warmup/iters mismatch: cli -w {} -n {} but "
                       "binary echoes warmup_iters={} iters={}".format(
                           args.warmup_iters, args.iters, echo_iters[1], echo_iters[0]))
    # ETA 用稳态均耗时外推（剔除首轮建链/so 加载等冷启动开销）；首轮无稳态基准不打
    eta = "--"
    if run_idx > 1 and first_run_cost is not None:
        steady_avg = (elapsed_total + duration - first_run_cost) / (run_idx - 1)
        eta = format_duration(steady_avg * (ctx.total_runs - run_idx))
    logger.info("[perf_test]     -> {} rows ({}, ETA {})".format(
        len(rows), format_duration(duration), eta))
    return rows, stdout, stderr, ok, duration


def _write_failed_row(ctx, writer, combo, ok, stderr):
    """失败/无结果轮审计：落 check_result=failed/noresult 的 CSV 行并回显原因。

    性能字段留空；ok=False 时 stderr 尾部进日志（缺 so、dlopen 失败等）。
    """
    op_type, data_type, reduce_op, engine, executor, templates = combo
    algo_str = algo_to_env(executor, templates)
    row = _base_csv_row(ctx, combo)
    row["check_result"] = "failed" if not ok else "noresult"
    row["timestamp"] = now_timestamp()
    writer.write_row(row)
    logger.warning("[perf_test] WARN {}: op={} dtype={} engine={} algo={} "
                   "(raw output saved to {})".format(
                       "run failed" if not ok else "no parsed rows",
                       op_type, data_type, engine, algo_str, ctx.args.output + ".raw"))
    if not ok:
        for line in tail_lines(stderr):
            logger.warning("[perf_test]     stderr | {}".format(line))


def _row_check_result(ctx, combo, size_bytes, check):
    """行级回退校验：名单 per-size 集不含该 (algo, 档位) → 实际执行的是回退算法。

    并集口径采集下，probe 在该档位过滤掉的算法由 selector 静默回退，
    该行标 fallback 供 optimize_config 与 failed 一并剔除。
    精确档位未命中时回查 size=0 通配键（DFX-only 名单档位未知，按算法存在性放行）。
    """
    op_type, data_type, reduce_op, engine, executor, templates = combo
    key = (op_type, engine, executor, tuple(templates), reduce_op, data_type, size_bytes)
    if check == "failed":
        return check
    if key in ctx.supported_keys:
        return check
    wildcard_key = key[:-1] + (0,)
    return check if wildcard_key in ctx.supported_keys else "fallback"


def _run_single(ctx, writer, combo, run_idx, elapsed_total, first_run_cost):
    """执行单轮采集：拉起 mpirun → 解析输出 → 结果落 CSV。

    返回 (解析出的数据行数, 本轮耗时秒)。失败/无结果轮落一行审计记录
    （check_result=failed/noresult）。原始输出 dump 到 <output>.raw 取证。
    """
    op_type, data_type, reduce_op, engine, executor, templates = combo
    algo_str = algo_to_env(executor, templates)
    rows, stdout, stderr, ok, duration = _exec_round(ctx, combo, run_idx, elapsed_total,
                                                     first_run_cost)
    dump_raw_output(
        ctx.args.output, "op={} dtype={} reduce_op={} engine={} algo={}".format(
            op_type, data_type, reduce_op, engine, algo_str), stdout, stderr)
    if not rows:
        _write_failed_row(ctx, writer, combo, ok, stderr)
        return 0, duration
    base_row = _base_csv_row(ctx, combo)
    timestamp = now_timestamp()
    for row in rows:
        writer.write_row({
            **base_row,
            "size_bytes": row.get("size_bytes"),
            "check_result": _row_check_result(ctx, combo, row.get("size_bytes"),
                                              row.get("check_result", "")),
            "alg_bandwidth(GB/s)": row.get("alg_bandwidth(GB/s)"),
            "alg_latency(us)": row.get("alg_latency(us)"),
            "timestamp": timestamp,
        })
    return len(rows), duration


def _collect(ctx, writer, combos):
    """循环执行全部采集轮，返回 (runs_ok, runs_fail, total_rows)。

    失败/无结果轮也写了 1 行审计记录，计入 total_rows。
    """
    runs_ok, runs_fail, total_rows = 0, 0, 0
    elapsed_total = 0.0
    first_run_cost = None  # 首轮耗时（含冷启动），第 1 轮后回填，ETA 稳态外推时剔除
    for run_idx, combo in enumerate(combos, 1):
        n_rows, duration = _run_single(ctx, writer, combo, run_idx, elapsed_total,
                                       first_run_cost)
        first_run_cost = duration if run_idx == 1 else first_run_cost
        elapsed_total += duration
        if n_rows:
            runs_ok += 1
        else:
            runs_fail += 1
        total_rows += n_rows or 1
    return runs_ok, runs_fail, total_rows


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = parse_args(argv)
    if not args.probe_output:
        args.probe_output = args.output + ".algos.json"
    try:
        op_types, engines, data_types, reduce_ops = validate_args(args)
        algos = build_algos(args.algos, args.executors, args.templates)
    except ValueError as err:
        logger.error("[perf_test] argument error: {}".format(err))
        return 1
    if algos is not None and not algos:
        logger.error("[perf_test] argument error: empty algo set")
        return 1

    try:
        mpi_flavor = preflight(op_types, args.bin_dir)
    except PreflightError as err:
        logger.error("[perf_test] preflight failed: {}".format(err))
        return 1

    try:
        topo = derive_topo(args.np_total, args.npus, args.hostfile, args.pods, args.super_pods)
    except ValueError as err:
        logger.error("[perf_test] argument error: {}".format(err))
        return 1
    exe_paths = {op_type: exe_path_of(op_type, args.bin_dir).replace("\\", "/")
                 for op_type in op_types}
    try:
        payload = load_or_run_probe(args, op_types, reduce_ops, data_types, engines, exe_paths, topo, mpi_flavor)
    except ProfTestError as err:
        logger.error("[perf_test] probe failed: {}".format(err))
        return 1
    except (OSError, ValueError) as err:
        logger.error("[perf_test] probe list error: {}".format(err))
        return 1

    plan = make_collect_plan(payload, op_types, engines, data_types, reduce_ops, algos)
    combos = build_run_combos(plan)
    plan = plan._replace(total_runs=len(combos))
    if not combos:
        logger.error("[perf_test] error: no algo combo left after probe list ∩ CLI filters "
                     "(op-types [{}], engines [{}], algos [{}])".format(
                         ",".join(op_types), ",".join(engines),
                         "explicit" if algos is not None else "list"))
        return 1

    hccl_buffsize = os.environ.get("HCCL_BUFFSIZE", "")
    # OpenMPI 需 -x 显式传递到远端节点：HCCL_ALGO 每轮必设；HCCL_BUFFSIZE 存在才传
    env_keys = ["HCCL_ALGO"] + (["HCCL_BUFFSIZE"] if hccl_buffsize else [])
    ctx = RunContext(args=args, profile=make_test_profile(args), exe_paths=exe_paths,
                     hccl_buffsize=hccl_buffsize, env_keys=env_keys, topo=topo,
                     total_runs=len(combos), mpi_flavor=mpi_flavor,
                     supported_keys=plan.supported_keys)

    # 采集计划（测试范围总览）：先打全量维度再开跑
    log_collect_plan(ctx, plan, payload, len(combos))
    try:
        writer = CsvWriter(args.output)
    except ValueError as err:
        logger.error("[perf_test] error: {}".format(err))
        return 1
    with writer:
        runs_ok, runs_fail, total_rows = _collect(ctx, writer, combos)

    logger.info("[perf_test] done: runs_ok={} runs_fail={} rows={} output={}".format(
        runs_ok, runs_fail, total_rows, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
