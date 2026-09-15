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
