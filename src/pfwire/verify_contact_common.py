"""两个接触验证脚本（Phase 4.3 Hertz / 4.4 Cattaneo–Mindlin）共用的小工具。"""

from __future__ import annotations

import pathlib
from typing import Callable, Sequence

import yaml

__all__ = ["fnum", "load_config", "report"]


def fnum(node, *keys) -> float:
    """取配置值并强制成 float。

    PyYAML 走 YAML 1.1：`1e-7`、`500e6` 这种不带指数符号的写法会被解析成
    **字符串**（CLAUDE.md 陷阱 2）。配置里的数值一律经这里取。
    """
    for k in keys[:-1]:
        node = node[k]
    return float(node[keys[-1]])


def load_config(path: pathlib.Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("配置根节点必须是映射：%s" % path)
    return cfg


def report(say: Callable[[str], None],
           checks: Sequence[tuple[str, float, float, float]]):
    """打印对拍表并返回 (是否全通过, 可序列化的行)。

    每条 check 是 (名称, FE 值, 参考值, 相对差阈值)。
    """
    say("")
    say("  %-16s %12s %12s %10s %10s %s"
        % ("量", "FE", "解析", "相对差", "阈值", "判定"))
    ok = True
    rows = []
    for name, fe, ref, lim in checks:
        rel = abs(fe - ref) / abs(ref) if ref else float("inf")
        passed = rel < lim
        ok = ok and passed
        say("  %-16s %12.6g %12.6g %9.2f%% %9.1f%% %s"
            % (name, fe, ref, 100 * rel, 100 * lim, "通过" if passed else "**未通过**"))
        rows.append({"name": name, "fe": fe, "ref": ref, "rel": rel,
                     "tol": lim, "passed": passed})
    return ok, rows
