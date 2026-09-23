#!/usr/bin/env bash
# 建立/更新 conda 环境 pf-wire。在 WSL2 里跑：bash scripts/setup_env.sh
#
# 分步是因为两个包都不能从 conda-forge 直接装：
#   - adios4dolfinx 0.11 只在 PyPI 上，且必须 --no-deps（见 DECISIONS.md D-0）；
#   - dolfinx_contact（Wells-Group/asimov-contact）conda-forge 上没有，只能源码编译（D-46）。
#
# 跳过接触求解器：SKIP_CONTACT=1 bash scripts/setup_env.sh
set -euo pipefail

ENV_NAME=pf-wire
ADIOS4DOLFINX_VERSION=0.11.0
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# asimov-contact **钉死在一个 commit，不跟 main 漂**（D-46 风险表）。
# f605fff = "Merge PR #187 dokken/contact-update-for-0.11"（2026-08-03），
# 正是把该库从 dolfinx 0.10 升到 0.11 的那次合并 —— 0.11 支持很新，必须记下来。
CONTACT_REPO=https://github.com/Wells-Group/asimov-contact.git
CONTACT_COMMIT=f605fffdcabe470e6d20525c424aa985b36b3c42
CONTACT_SRC="$REPO_DIR/.ext/asimov-contact"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[1/5] 环境 $ENV_NAME 已存在，按 environment.yml 更新"
  conda env update -n "$ENV_NAME" -f "$REPO_DIR/environment.yml" --prune
else
  echo "[1/5] 创建环境 $ENV_NAME"
  conda env create -f "$REPO_DIR/environment.yml" -y
fi

conda activate "$ENV_NAME"

echo "[2/5] pip 安装 adios4dolfinx==$ADIOS4DOLFINX_VERSION（--no-deps）"
pip install --no-deps --no-build-isolation "adios4dolfinx==$ADIOS4DOLFINX_VERSION"

echo "[3/5] 以可编辑模式安装 pfwire（--no-deps，依赖全由 conda 提供）"
pip install -e "$REPO_DIR" --no-deps --no-build-isolation

if [ "${SKIP_CONTACT:-0}" = "1" ]; then
  echo "[4/5] 跳过 dolfinx_contact（SKIP_CONTACT=1）"
  echo "[5/5] 跳过"
else
  echo "[4/5] 源码编译 dolfinx_contact @ ${CONTACT_COMMIT:0:12}"
  mkdir -p "$REPO_DIR/.ext"
  if [ -d "$CONTACT_SRC/.git" ]; then
    git -C "$CONTACT_SRC" fetch --depth 1 origin "$CONTACT_COMMIT"
  else
    rm -rf "$CONTACT_SRC"
    git init -q "$CONTACT_SRC"
    git -C "$CONTACT_SRC" remote add origin "$CONTACT_REPO"
    git -C "$CONTACT_SRC" fetch --depth 1 origin "$CONTACT_COMMIT"
  fi
  git -C "$CONTACT_SRC" checkout -q FETCH_HEAD

  # 用 conda 的编译器，不用系统 /usr/bin/c++ —— 与 conda 编出来的 libdolfinx
  # 混用不同 toolchain 会出 ABI 问题。cxx-compiler 包提供 $CXX/$CC。
  export CC="${CC:-$CONDA_PREFIX/bin/mpicc}"
  export CXX="${CXX:-$CONDA_PREFIX/bin/mpicxx}"
  # Release 而非 README 里的 Developer：Developer 开断言与额外检查，长算例会明显变慢。
  cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
        -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
        -B "$CONTACT_SRC/build-contact" -S "$CONTACT_SRC/cpp/"
  cmake --build "$CONTACT_SRC/build-contact" --target install

  echo "[5/5] 安装 dolfinx_contact 的 python 绑定（--no-build-isolation，nanobind 由 conda 提供）"
  pip install --no-build-isolation "$CONTACT_SRC/python/"
fi

# conda env create 会在 cwd 留下临时 requirements 文件
rm -f "$REPO_DIR"/condaenv.*.requirements.txt

echo
echo "完成。验收："
echo "  conda activate $ENV_NAME && pytest tests/test_env.py"
echo "  python -c 'import dolfinx_contact; print(dolfinx_contact.__file__)'"
