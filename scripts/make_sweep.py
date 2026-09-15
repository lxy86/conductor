"""生成 Phase 2.5 的 S–N 扫描配置（PLAN 2.5）。

以 configs/fretting_base.yaml 为模板，**只改 loading.sigma_a 与 output.dir**，
其余（p0、Q/μP、μ、R、网格、求解器设置）逐字不动 —— S–N 曲线要求单参数变化，
顺手改别的会让曲线失去意义。
"""

from __future__ import annotations

import argparse
import pathlib

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "configs" / "fretting_base.yaml"
OUT_DIR = REPO / "configs" / "sweep"

#: PLAN 2.5：σ_a ∈ {20, 27, 35, 45, 55, 65} MPa
SIGMA_A_MPA = (20, 27, 35, 45, 55, 65)


def case_name(sigma_mpa: int) -> str:
    return "sn_sigma%02d" % sigma_mpa


def build(sigma_mpa: int, template: dict) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(template))   # 深拷贝
    cfg["loading"]["sigma_a"] = float(sigma_mpa) * 1.0e6
    cfg["output"]["dir"] = "data/%s" % case_name(sigma_mpa)
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 S–N 扫描配置")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的配置")
    args = ap.parse_args(argv)

    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    if template.get("case_type") != "fretting":
        raise SystemExit("模板不是 fretting 工况：%s" % TEMPLATE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for s in SIGMA_A_MPA:
        path = OUT_DIR / ("%s.yaml" % case_name(s))
        if path.exists() and not args.force:
            print("  跳过（已存在）%s" % path.relative_to(REPO))
            continue
        header = (
            "# Phase 2.5 S–N 扫描，σ_a = %d MPa。**由 scripts/make_sweep.py 生成，"
            "不要手改**\n"
            "# —— 手改会让它与 configs/fretting_base.yaml 脱钩，而 S–N 曲线的前提\n"
            "#    是六个工况除 σ_a 外逐字相同。要改就改模板再重新生成。\n" % s
        )
        path.write_text(
            header + yaml.safe_dump(build(s, template), sort_keys=False,
                                    allow_unicode=True),
            encoding="utf-8",
        )
        written.append(path.relative_to(REPO))
        print("  写入 %s（σ_a = %d MPa）" % (path.relative_to(REPO), s))

    print("共 %d 个工况，%d 个新写入" % (len(SIGMA_A_MPA), len(written)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
