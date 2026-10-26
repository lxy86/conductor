"""仓库卫生：防住那些「不是物理、但会白白烧掉几小时」的低级问题。"""

from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def shell_scripts():
    return sorted(REPO.glob("scripts/*.sh")) + sorted(REPO.glob("data/*.sh"))


@pytest.mark.parametrize("path", shell_scripts(), ids=lambda p: p.name)
def test_shell_scripts_use_lf_line_endings(path: pathlib.Path):
    """shell 脚本必须是 LF 行尾。

    CRLF 会让 WSL 里的 bash 直接读不了，报的还是一串看不懂的错：
        set: - : invalid option
        conda: command not found
        cd: $'/root/pf-wire\r': No such file or directory
        syntax error near unexpected token `$'do\r''
    根因是在 Windows 上用 pathlib.Path.write_text() 改脚本 —— 文本模式会把
    LF 翻成 CRLF。本项目已经因此白跑过两次，所以钉一条测试。
    改脚本要么用 bash heredoc，要么 write_bytes / write_text(newline="\n")。
    """
    data = path.read_bytes()
    assert b"\r\n" not in data, (
        "%s 是 CRLF 行尾，WSL 里的 bash 跑不了。"
        "用 heredoc 重写，或 write_text(..., newline='\n')" % path.name
    )


def test_shell_scripts_do_not_hardcode_home_paths():
    """shell 脚本不得硬编码 `/root/...` 之类的绝对家目录路径。

    2026-10-26 换机器实测：`scripts/*.sh` 里有 25 个各自写
        source /root/miniforge3/etc/profile.d/conda.sh
        cd /root/pf-wire || exit 1
    新机器上用户是 `code`、conda 在 `/home/code/miniforge3`、项目在
    `/mnt/d/疲劳寿命`，**25 个脚本全部 `cd` 失败**，且报错还是
    `cd: $'/root/pf-wire\r'` 这种看不出根因的形态。

    改法：统一 source `scripts/_env.sh`（按 `PFWIRE_HOME` /
    `conda info --base` / 常见安装位置三级定位），不要在脚本里写死路径。
    本测试把这条钉住，防止再长回去。

    **范围只含 `scripts/`**：`data/_*.sh` 是「用一次就扔」的探针脚手架
    （.gitignore 里 `data/_*.py` / `data/_*.log` 同理，见 CLAUDE.md 陷阱 10），
    本来就不随仓库走，不该被这条约束管住 —— 而且它们有 200+ 个，
    逐个改造的收益远小于维护成本。
    """
    offenders = []
    for path in sorted(REPO.glob("scripts/*.sh")):
        if path.name == "_env.sh":
            # _env.sh 自己要在 fallback 列表里枚举候选路径，跳过
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue  # 注释里的示例/说明不算
            if "/root/" in line:
                offenders.append(f"{path.name}:{lineno}: {stripped}")

    assert not offenders, (
        "以下 shell 脚本硬编码了 /root/ 路径，换机器必然 cd 失败：\n  "
        + "\n  ".join(offenders)
        + "\n改成 source scripts/_env.sh（见 _env.sh 的 docstring）。"
    )
