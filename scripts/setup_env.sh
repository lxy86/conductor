#!/usr/bin/env bash
# 建立/更新 conda 环境 pf-wire。在 WSL2 里跑：bash scripts/setup_env.sh
#
# 分步是因为两个包都不能从 conda-forge 直接装：
#   - adios4dolfinx 0.11 只在 PyPI 上，且必须 --no-deps（见 DECISIONS.md D-0）；
#   - dolfinx_contact（Wells-Group/asimov-contact）conda-forge 上没有，只能源码编译（D-46）。
#
# 跳过接触求解器：SKIP_CONTACT=1 bash scripts/setup_env.sh
set -euo pipefail

ENV_NAME="${PFWIRE_ENV:-pf-wire}"
ADIOS4DOLFINX_VERSION=0.11.0
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# asimov-contact **钉死在一个 commit，不跟 main 漂**（D-46 风险表）。
# f605fff = "Merge PR #187 dokken/contact-update-for-0.11"（2026-08-03），
# 正是把该库从 dolfinx 0.10 升到 0.11 的那次合并 —— 0.11 支持很新，必须记下来。
CONTACT_REPO=https://github.com/Wells-Group/asimov-contact.git
CONTACT_COMMIT=f605fffdcabe470e6d20525c424aa985b36b3c42
CONTACT_SRC="$REPO_DIR/.ext/asimov-contact"

# 定位 conda：不能写 `source "$(conda info --base)/..."` —— 非交互式
# `bash -lc` 不走 .bashrc，conda 不在 PATH 时那行会展开成
# `/etc/profile.d/conda.sh`（家目录都没了），报 `conda: command not found`。
# 2026-10-26 实测踩过。_env.sh 按 $PFWIRE_CONDA_BASE → conda info --base
# → 常见安装位置三级兜底，建环境这一步尤其不能倒在这里。
# shellcheck source=./_env.sh
source "$REPO_DIR/scripts/_env.sh"
# _env.sh 里 PFWIRE_ENV 默认就是 pf-wire；ENV_NAME 在第 11 行已按它取过值。

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

  # ---- 清掉从别处带来的构建缓存 -------------------------------------------
  # CMake 把**绝对路径**写进 CMakeCache.txt。换机器/换项目路径后（本机从
  # `/root/pf-wire` 变成 `/mnt/d/疲劳寿命`），cmake 直接拒绝复用：
  #   CMake Error: The current CMakeCache.txt directory <新路径> is different
  #   than the directory <旧路径> where CMakeCache.txt was created.
  # 而 .ext/ 在 .gitignore 里、`build-contact/` 又不会被 setup 脚本清理，
  # 于是**只要拷过目录就必踩**（2026-10-26 实测）。
  # 判据：cache 里记录的构建目录 ≠ 当前，就整个删掉重来（它是纯产物，删了不心疼）。
  CONTACT_BUILD="$CONTACT_SRC/build-contact"
  if [ -f "$CONTACT_BUILD/CMakeCache.txt" ]; then
    CACHED_DIR=$(sed -n 's|^# For build in directory: ||p' "$CONTACT_BUILD/CMakeCache.txt" | head -1)
    if [ -n "$CACHED_DIR" ] && [ "$CACHED_DIR" != "$CONTACT_BUILD" ]; then
      echo "  构建缓存来自别处（$CACHED_DIR），删掉重建"
      rm -rf "$CONTACT_BUILD"
    fi
  fi

  # ---- 必打的两个补丁（D-48）----------------------------------------------
  # 上游 python/CMakeLists.txt 建 nanobind 模块时**没开 stable ABI**，
  # 而 conda-forge 的 dolfinx 绑定是 cpp.abi3.so（stable ABI）。两者 ABI 不同
  # 就不共享 nanobind 类型注册表，于是 ContactProblem 构造时报
  #   incompatible function arguments ... dolfinx::mesh::MeshTags<int>
  # 签名里打印的是**裸 C++ 类型名**，正是「认不出对方类型」的特征。
  # 只加 STABLE_ABI 不够：nanobind 仅在 CMake 找到 Development.SABIModule
  # 时才真正启用它，否则**静默忽略**（产物仍是 cpp.cpython-312-*.so）。
  # 补丁**逐行处理**，不做跨行匹配 —— 跨行模式在 shell heredoc 里极易被转义弄坏。
  python - "$CONTACT_SRC/python/CMakeLists.txt" <<'PATCH_EOF'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
lines = path.read_text().splitlines()
out = []
done_sabi = False
done_abi3 = False
in_nb = False
for line in lines:
    if line.startswith("find_package(Python") and "Development.SABIModule" not in line:
        line = line.replace("Development REQUIRED", "Development Development.SABIModule REQUIRED")
        done_sabi = "Development.SABIModule" in line
    if "Development.SABIModule" in line:
        done_sabi = True
    if line.strip().startswith("nanobind_add_module("):
        in_nb = True
    elif in_nb and line.strip() == ")":
        in_nb = False
    if in_nb and line.strip() == "STABLE_ABI":
        done_abi3 = True
    if in_nb and line.strip() == "NOMINSIZE" and not done_abi3:
        out.append("  STABLE_ABI")
        done_abi3 = True
    out.append(line)
if not (done_sabi and done_abi3):
    raise SystemExit(
        "补丁锚点失效（SABIModule=%s, STABLE_ABI=%s），上游可能已改动 CMakeLists"
        % (done_sabi, done_abi3))
path.write_text("\n".join(out) + "\n")
print("  已打 stable-ABI 补丁（Development.SABIModule + STABLE_ABI）")
PATCH_EOF

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
echo "  ^ 必须是 cpp.abi3.so；若是 cpp.cpython-312-*.so 则 stable-ABI 补丁没生效（D-48）"
