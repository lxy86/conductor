#!/usr/bin/env bash
# 公共环境定位 —— 被 scripts/*.sh source，**不要直接跑**。
#
# 为什么要有这个文件：项目原本在 25 个脚本里各自硬编码
#     source /root/miniforge3/etc/profile.d/conda.sh
#     cd /root/pf-wire || exit 1
# `/root/pf-wire` 是旧机器（root 用户 + 手工建的 ASCII 软链）的路径。
# 换机器后用户名/家目录一变（本机是 /home/code，conda 在 /home/code/miniforge3），
# **25 个脚本全部 `cd` 失败**，而报错还是 `cd: $'/root/pf-wire\r'` 这种看不懂的形态
# （CRLF 一旦混进来，见 tests/test_repo_hygiene.py）。
#
# 用法（替换掉原来那 4 行）：
#     source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
#
# 定位规则（按优先级）：
#   1. $PFWIRE_HOME  —— 显式指定项目根（ASCII 路径）
#   2. $PFWIRE_CONDA_BASE —— 显式指定 conda 根
#   3. 本文件的上两级目录（即仓库根），要求它在 conda 环境的 sys.path 里可达
#   4. conda info --base
set -u

# ---- conda 根 -------------------------------------------------------------
if [ -z "${PFWIRE_CONDA_BASE:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        PFWIRE_CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    fi
fi
# conda 不在 PATH 时，试常见的 miniforge/miniconda 位置（登录 shell 才有 PATH）。
# ⚠ $HOME 不保证有值：`wsl -d X -- bash -c '...'` 这种非登录非交互的调用里
# HOME 可能是空的（2026-10-26 实测），于是 "${HOME}/miniforge3" 会展开成
# "/miniforge3" 而全部落空。用 `getent passwd` 兜一层，再退回 /root。
if [ -z "${PFWIRE_CONDA_BASE:-}" ]; then
    _home="${HOME:-}"
    if [ -z "$_home" ]; then
        _home="$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)"
    fi
    : "${_home:=/root}"
    for _cand in \
        "${_home}/miniforge3" "${_home}/miniconda3" "${_home}/mambaforge" \
        /opt/conda /opt/miniforge3 /opt/miniconda3 \
        /root/miniforge3 /root/miniconda3; do
        if [ -f "$_cand/etc/profile.d/conda.sh" ]; then
            PFWIRE_CONDA_BASE="$_cand"
            break
        fi
    done
    unset _home
fi

if [ -n "${PFWIRE_CONDA_BASE:-}" ] && [ -f "$PFWIRE_CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$PFWIRE_CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "[_env.sh] 找不到 conda。设 PFWIRE_CONDA_BASE=/path/to/conda 后重试。" >&2
    exit 1
fi

# ---- 环境名 ---------------------------------------------------------------
# 默认 pf-wire；setup_env.sh 里也是这个名字（D-0/D-46）。
: "${PFWIRE_ENV:=pf-wire}"

# ---- 项目根 ---------------------------------------------------------------
if [ -z "${PFWIRE_HOME:-}" ]; then
    # 本文件在 <repo>/scripts/_env.sh，上两级即仓库根
    PFWIRE_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [ ! -f "$PFWIRE_HOME/pyproject.toml" ]; then
    echo "[_env.sh] $PFWIRE_HOME 不像仓库根（没有 pyproject.toml）。" >&2
    echo "         设 PFWIRE_HOME=<项目 ASCII 路径> 后重试。" >&2
    exit 1
fi

export PFWIRE_HOME PFWIRE_CONDA_BASE
