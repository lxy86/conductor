#!/usr/bin/env bash
# 建立/更新 conda 环境 pf-wire。在 WSL2 里跑：bash scripts/setup_env.sh
# 分两步是因为 adios4dolfinx 0.11 只在 PyPI 上，且必须 --no-deps（见 DECISIONS.md D-0）。
set -euo pipefail

ENV_NAME=pf-wire
ADIOS4DOLFINX_VERSION=0.11.0
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[1/3] 环境 $ENV_NAME 已存在，按 environment.yml 更新"
  conda env update -n "$ENV_NAME" -f "$REPO_DIR/environment.yml" --prune
else
  echo "[1/3] 创建环境 $ENV_NAME"
  conda env create -f "$REPO_DIR/environment.yml" -y
fi

conda activate "$ENV_NAME"

echo "[2/3] pip 安装 adios4dolfinx==$ADIOS4DOLFINX_VERSION（--no-deps）"
pip install --no-deps --no-build-isolation "adios4dolfinx==$ADIOS4DOLFINX_VERSION"

echo "[3/3] 以可编辑模式安装 pfwire（--no-deps，依赖全由 conda 提供）"
pip install -e "$REPO_DIR" --no-deps --no-build-isolation

# conda env create 会在 cwd 留下临时 requirements 文件
rm -f "$REPO_DIR"/condaenv.*.requirements.txt

echo
echo "完成。验收：conda activate $ENV_NAME && pytest tests/test_env.py"
