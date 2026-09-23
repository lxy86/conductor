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

#: PLAN 2.6：固定 σ_a = 35 MPa，p0 ∈ {210, 240, 255} MPa（D-28 的可用窗口）。
#: 240 就是基准工况（= sn_sigma35），不重复生成。
#: 下界为什么不是 D-27 写的 206：a ∝ p0，p0 = 206 给出 a/ℓ = 4.99，
#: **违反 D-1 的 a ≥ 5ℓ**。真正可用的窗口是 [210, 260]。
P0_MPA = (210, 255)

#: PLAN 2.7：网格无关性。基准是 h = ℓ/4；细化档用 ℓ/N。
#: ℓ/6 已跑出 N_f 差 −17.3%（D-40，超过 10% 验收线），故再加 ℓ/8：
#: 三档才能估出收敛阶 p 并做 Richardson 外推，两档只能说「差多少」、
#: 说不了「收敛没收敛」。ℓ = 2 µm。
ELL = 2.0e-6
MESH_DIVISORS = (6, 8)

#: PLAN 2.7 的延伸（用户 2026-09-23 指示）：在**第二个载荷**上验证
#: 「扩展寿命修正系数 k 与载荷无关」这条假设 —— D-41 只在 σ_a = 35 MPa 上
#: 细化过，而全套 S–N / p0 结果的解析修正都压在这一个系数上。
#: 取 σ_a = 27 MPa：它的萌生占比只有 9.3%，修正幅度 −14.8%，
#: 是 S–N 扫描里判别力最强的一档；几何不变、只改本体应力，正对着该假设。
#: （σ_a = 45 的修正幅度只有 −1.5%，小于噪声，测不出东西。）
MESH_CHECK = ((27, 6),)


HEADER_P0 = (
    "# Phase 2.6 参数影响，p0 = %d MPa（σ_a 固定 35 MPa）。"
    "**由 scripts/make_sweep.py 生成，不要手改**" + chr(10)
)
HEADER_MESH = (
    "# Phase 2.7 网格无关性，h_fine = ell/%d（基准是 ell/4）。"
    "**由 scripts/make_sweep.py 生成，不要手改**" + chr(10)
)


def case_name(sigma_mpa: int) -> str:
    return "sn_sigma%02d" % sigma_mpa


def build(sigma_mpa: int, template: dict) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(template))   # 深拷贝
    cfg["loading"]["sigma_a"] = float(sigma_mpa) * 1.0e6
    cfg["output"]["dir"] = "data/%s" % case_name(sigma_mpa)
    return cfg


def build_p0(p0_mpa: int, template: dict) -> dict:
    """PLAN 2.6：只改 contact.p0。

    **不要顺手改网格**：加密区半宽是 refine_halfwidth_factor × a，而 a 由
    loads 从 p0 算出（D-25），配置里不写死 —— p0 一改，加密区自动跟着变。
    """
    cfg = yaml.safe_load(yaml.safe_dump(template))
    cfg["contact"]["p0"] = float(p0_mpa) * 1.0e6
    cfg["output"]["dir"] = "data/p0_%03d" % p0_mpa
    return cfg


def build_mesh(template: dict, divisor: int) -> dict:
    """PLAN 2.7：只改 h_fine（ℓ/4 -> ℓ/divisor）与采样间距。

    sample_spacing 必须跟着 h_fine 走：它原本就取 = h_fine（「再细无意义，
    d 是 P1」），若只细化网格而不细化采样，裂纹深度仍按 0.5 µm 分箱，
    网格无关性就成了「两次都量到同一个箱子」的假相符。
    """
    h = ELL / float(divisor)
    cfg = yaml.safe_load(yaml.safe_dump(template))
    cfg["mesh"]["h_fine"] = h
    cfg["failure"]["sample_spacing"] = h
    cfg["output"]["dir"] = "data/mesh_l%d" % divisor
    return cfg


def build_mesh_check(sigma_mpa: int, divisor: int, template: dict) -> dict:
    """同时改 σ_a 与 h_fine：在另一个载荷上复刻 PLAN 2.7 的网格细化。"""
    cfg = build(sigma_mpa, template)
    h = ELL / float(divisor)
    cfg["mesh"]["h_fine"] = h
    cfg["failure"]["sample_spacing"] = h
    cfg["output"]["dir"] = "data/%s_l%d" % (case_name(sigma_mpa), divisor)
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

    for p0 in P0_MPA:
        path = OUT_DIR / ("p0_%03d.yaml" % p0)
        if path.exists() and not args.force:
            print("  跳过（已存在）%s" % path.relative_to(REPO))
            continue
        path.write_text(
            HEADER_P0 % p0
            + yaml.safe_dump(build_p0(p0, template), sort_keys=False,
                             allow_unicode=True),
            encoding="utf-8")
        written.append(path.relative_to(REPO))
        print("  写入 %s（p0 = %d MPa）" % (path.relative_to(REPO), p0))

    for div in MESH_DIVISORS:
        path = OUT_DIR / ("mesh_l%d.yaml" % div)
        if path.exists() and not args.force:
            print("  跳过（已存在）%s" % path.relative_to(REPO))
            continue
        path.write_text(
            HEADER_MESH % div
            + yaml.safe_dump(build_mesh(template, div), sort_keys=False,
                             allow_unicode=True),
            encoding="utf-8")
        written.append(path.relative_to(REPO))
        print("  写入 %s（h_fine = ℓ/%d = %.4f µm）"
              % (path.relative_to(REPO), div, ELL / div * 1e6))

    for sigma, div in MESH_CHECK:
        path = OUT_DIR / ("%s_l%d.yaml" % (case_name(sigma), div))
        if path.exists() and not args.force:
            print("  跳过（已存在）%s" % path.relative_to(REPO))
            continue
        path.write_text(
            "# PLAN 2.7 延伸：在 sigma_a = %d MPa 上复刻 ell/%d 细化，"
            "用于验证修正系数 k 与载荷无关。"
            "**由 scripts/make_sweep.py 生成，不要手改**" % (sigma, div) + chr(10)
            + yaml.safe_dump(build_mesh_check(sigma, div, template),
                             sort_keys=False, allow_unicode=True),
            encoding="utf-8")
        written.append(path.relative_to(REPO))
        print("  写入 %s（σ_a = %d MPa, h = ℓ/%d）"
              % (path.relative_to(REPO), sigma, div))

    total = (len(SIGMA_A_MPA) + len(P0_MPA) + len(MESH_DIVISORS)
             + len(MESH_CHECK))
    print("共 %d 个工况，%d 个新写入" % (total, len(written)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
