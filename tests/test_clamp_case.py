"""6.5 T4（W' 的 δ 搜索）与 T5（起裂/转向度量）的单元测试（D-67 §4、§6）。

两者都做成纯函数，故这里**不跑有限元**：T4 注入解析 W(δ)，T5 喂合成 d 场。

T4：
  T4a 割线法在线性 W(δ) 上收敛到 < 1%；
  T4b 在非线性（Hertz 型 W ∝ δ^{3/2}）上同样收敛；
  T4c 不收敛即抛异常（恒值 W(δ)，永远到不了目标）。
T5：
  T5a 竖直裂纹：θ_init、θ_later 均 ≈ 0；
  T5b 倾斜 + 转向：先斜 15° 再拐 35°；
  T5c 表面连通性：孤立斑块不算裂纹（reached_nuc=False）；
  T5d 质心位置与已知几何一致。
"""

import math
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

from pfwire.clamp_case import crack_metrics, secant_delta

W_TARGET = 1.75e6          # W'（N/m，单位厚度），D-67 §3
K_LINEAR = 1.0e11          # 线性 W(δ) 的刚度


# --------------------------------------------------------------------------
# T4
# --------------------------------------------------------------------------
def test_secant_converges_on_linear_response():
    """T4a：线性 W(δ) = k δ 上，割线法收敛到 < 1%（D-67 §3 的容差）。

    只判「容差内」，不判「等于解析根」：割线法在 1% 容差满足时就返回，
    这是设计意图（省 FE 解），断言到 1e-12 是自相矛盾。
    """
    W = lambda d: K_LINEAR * d
    d = secant_delta(W, W_TARGET, delta0=17.0e-6, rtol=1.0e-2, max_iter=20)
    assert abs(W(d) - W_TARGET) <= 1.0e-2 * W_TARGET
    # 线性 W(δ) 上割线法一步到位（两点确定一条直线），故这里可以严格比
    assert d == pytest.approx(W_TARGET / K_LINEAR, rel=1.0e-12)


def test_secant_converges_on_nonlinear_response():
    """T4b：Hertz 型 W(δ) = k δ^{3/2} 上同样收敛到 < 1%。"""
    K = 1.0e13
    W = lambda d: K * d**1.5
    d = secant_delta(W, W_TARGET, delta0=17.0e-6, rtol=1.0e-2, max_iter=20)
    assert abs(W(d) - W_TARGET) <= 1.0e-2 * W_TARGET
    # 非线性下割线法在容差处就停，故与解析根只比到 1% 量级
    assert d == pytest.approx((W_TARGET / K) ** (2.0 / 3.0), rel=1.0e-2)
    # 收紧容差应当收敛得更准（验证容差确实起作用，不是碰巧）
    d_tight = secant_delta(W, W_TARGET, delta0=17.0e-6, rtol=1.0e-8, max_iter=20)
    assert d_tight == pytest.approx((W_TARGET / K) ** (2.0 / 3.0), rel=1.0e-6)


def test_secant_raises_when_target_is_unreachable():
    """T4c：W(δ) 单调但**渐近饱和**、目标够不到时抛「未收敛」（硬约束 6）。

    注入 W = W_max(1 − e^{−δ/d_s})：单调增但上界 W_max < W_target，割线法会一路
    外推到很大的 δ 却永远差着 (W_target − W_max)，最终被 max_iter 拦住。
    这是「目标不可达」这一类失败，与「W 不单调」区分开（后者由下一条测）。
    """
    W_max = 0.5 * W_TARGET
    W = lambda d: W_max * (1.0 - math.exp(-d / 1.0e-6))
    with pytest.raises(RuntimeError, match="未收敛"):
        secant_delta(W, W_TARGET, delta0=1.0e-6, rtol=1.0e-2, max_iter=20)


def test_secant_raises_when_W_is_not_monotone():
    """T4 辅助：W(δ) 恒值（两点 W 相同）时报「不是单调」，不是悄悄返回坏结果。"""
    with pytest.raises(RuntimeError, match="不是单调"):
        secant_delta(lambda d: 0.0, W_TARGET, delta0=17.0e-6, rtol=1.0e-2, max_iter=5)


def test_secant_returns_immediately_when_start_is_on_target():
    """T4 辅助：起点已在容差内则不再迭代（省一次 FE 解）。"""
    calls = []

    def W(d):
        calls.append(d)
        return K_LINEAR * d

    d = secant_delta(W, W_TARGET, delta0=W_TARGET / K_LINEAR, rtol=1.0e-2)
    assert d == pytest.approx(W_TARGET / K_LINEAR)
    assert len(calls) == 1, "起点已达标时应只调用一次 W(δ)，实际 %d 次" % len(calls)


def test_secant_validates_arguments():
    """T4 辅助：非法容差/迭代数直接拒绝。"""
    with pytest.raises(ValueError):
        secant_delta(lambda d: d, W_TARGET, rtol=0.0)
    with pytest.raises(ValueError):
        secant_delta(lambda d: d, W_TARGET, max_iter=0)


# --------------------------------------------------------------------------
# T5：合成 d 场
# --------------------------------------------------------------------------
def _grid(n_x=2001, depth_max=150.0e-6, n_depth=1501):
    """规则的 (x, depth) 采样网格，深度向下为正、表面为 0。

    分辨率要足够细：6 µm 切片上若只有几个网格点，质心的离散误差会吃掉 0.5°
    的角度容差（实测 dx = 1 µm 时 θ_init 读出 14.04° 而非 15°）。取 dx = 0.2 µm、
    dd = 0.1 µm，离散误差降到 0.05° 以下。
    """
    xs = np.linspace(-200.0e-6, 200.0e-6, n_x)
    depths = np.linspace(0.0, depth_max, n_depth)
    return xs, depths


def _vertical_band(xs, depths, x0=0.0, halfwidth=3.0e-6, top=0.0, bottom=120.0e-6):
    """一条竖直带：|x − x0| <= halfwidth 且 top <= depth <= bottom 处 d = 1。"""
    d = np.zeros((len(depths), len(xs)))
    X, D = np.meshgrid(xs, depths)
    d[(np.abs(X - x0) <= halfwidth) & (D >= top) & (D <= bottom)] = 1.0
    return d


def _kinked_band(xs, depths, x0=0.0, halfwidth=3.0e-6,
                 theta1_deg=15.0, theta2_deg=35.0, kink_depth=6.0e-6,
                 bottom=120.0e-6):
    """先斜 theta1、在 kink_depth 处拐成 theta2 的两段带（深度向下为正）。"""
    d = np.zeros((len(depths), len(xs)))
    t1, t2 = math.radians(theta1_deg), math.radians(theta2_deg)
    X, D = np.meshgrid(xs, depths)
    # 中心线 x_c(depth)：depth <= kink 时按 theta1，之后按 theta2
    xc = np.where(D <= kink_depth,
                  x0 + np.tan(t1) * D,
                  x0 + np.tan(t1) * kink_depth + np.tan(t2) * (D - kink_depth))
    d[np.abs(X - xc) <= halfwidth / math.cos(min(abs(t1), abs(t2)))] = 1.0
    d[D > bottom] = 0.0
    d[D < 0.0] = 0.0
    return d


def test_vertical_crack_has_zero_angles():
    """T5a：竖直裂纹的 θ_init、θ_later 均 ≈ 0（< 0.5°）。"""
    xs, depths = _grid()
    d = _vertical_band(xs, depths)
    m = crack_metrics(d, xs, depths)
    assert m is not None and m.reached_nuc
    assert m.theta_init_deg == pytest.approx(0.0, abs=0.5)
    assert m.theta_later_deg == pytest.approx(0.0, abs=0.5)
    assert m.x_init == pytest.approx(0.0, abs=1e-6)


def test_kinked_crack_gives_init_and_later_angles():
    """T5b：先斜 15° 再拐 35°，θ_init ≈ 15°、θ_later ≈ 35°（各 < 0.5°）。

    注意 θ_later 是「6 µm 质心 → 100 µm 质心」的连线：拐点在 6 µm 处，故第二段
    走的是 theta2 那一段，角度应读回 35°。
    """
    xs, depths = _grid()
    d = _kinked_band(xs, depths)
    m = crack_metrics(d, xs, depths)
    assert m is not None and m.reached_nuc
    assert m.theta_init_deg == pytest.approx(15.0, abs=0.5)
    assert m.theta_later_deg == pytest.approx(35.0, abs=0.5)


def test_isolated_damage_patch_is_not_a_crack():
    """T5c：与表面不连通的孤立斑块不算裂纹（reached_nuc=False）。"""
    xs, depths = _grid()
    d = np.zeros((len(depths), len(xs)))
    X, D = np.meshgrid(xs, depths)
    # 悬空斑块：深度 40–120 µm 的一块，表面（depth=0）没有损伤与之相连
    d[(np.abs(X) <= 10.0e-6) & (D >= 40.0e-6) & (D <= 120.0e-6)] = 1.0
    m = crack_metrics(d, xs, depths)
    assert m is not None
    assert not m.reached_nuc, "孤立斑块不应算作表面连通裂纹"
    assert m.depth_max == 0.0
    assert math.isnan(m.theta_init_deg)


def test_centroids_match_known_geometry():
    """T5d：质心坐标与已知几何一致（< 1 个网格间距）。"""
    xs, depths = _grid()
    d = _vertical_band(xs, depths, x0=20.0e-6)
    m = crack_metrics(d, xs, depths)
    dx = xs[1] - xs[0]
    dd = depths[1] - depths[0]
    assert m.centroid_6[0] == pytest.approx(20.0e-6, abs=dx)
    assert m.centroid_100[0] == pytest.approx(20.0e-6, abs=dx)
    # 切片深度取「最接近 6 µm / 100 µm 的那一行」
    assert m.centroid_6[1] == pytest.approx(6.0e-6, abs=dd)
    assert m.centroid_100[1] == pytest.approx(100.0e-6, abs=dd)


def test_no_damage_returns_no_nucleation():
    """T5 辅助：d 全为 0 时 reached_nuc=False，且不抛异常。"""
    xs, depths = _grid()
    m = crack_metrics(np.zeros((len(depths), len(xs))), xs, depths)
    assert m is not None
    assert not m.reached_nuc
    assert m.dmax == 0.0


def test_shape_mismatch_is_rejected():
    """T5 辅助：d 的形状与坐标不匹配时直接报错（静默错位比报错更难查）。"""
    xs, depths = _grid()
    with pytest.raises(ValueError, match="d_field 形状"):
        crack_metrics(np.zeros((3, 3)), xs, depths)


# --------------------------------------------------------------------------
# T6：验收判据（纯函数，不依赖 FEM）
# --------------------------------------------------------------------------
def _metrics(x_init, theta_init, theta_later, reached=True, depth_max=120.0e-6):
    """构造一个 CrackMetrics（只填判据用到的字段）。"""
    from pfwire.clamp_case import CrackMetrics
    return CrackMetrics(x_init=x_init, centroid_6=(x_init, 6.0e-6),
                        centroid_100=(x_init, 100.0e-6),
                        theta_init_deg=theta_init, theta_later_deg=theta_later,
                        reached_nuc=reached, dmax=1.0, depth_max=depth_max)


def test_acceptance_passes_for_inclined_turning_crack_at_edge():
    """T6a：起裂在接触边缘、倾斜 15°、转向 25° -> 三条判据全通过。"""
    from pfwire.clamp_case import judge_acceptance
    a_contact, a_pl = 2.917e-3, 2.389e-3
    m = _metrics(x_init=a_contact, theta_init=15.0, theta_later=40.0)
    r = judge_acceptance(m, N_nuc=1.0e5, a_contact=a_contact, a_pl=a_pl)
    assert r.passed, r.details
    assert r.checks == {"nucleated": True, "x_init_in_window": True,
                        "inclined": True, "turned": True}
    assert r.N_nuc == pytest.approx(1.0e5)


def test_acceptance_checks_each_criterion_separately():
    """T6b：逐条判据独立（位置、倾斜、转向各测一次失败）。"""
    from pfwire.clamp_case import judge_acceptance
    a_contact, a_pl = 2.917e-3, 2.389e-3

    # 1) 起裂位置太靠内（远离两个接触边缘）
    r = judge_acceptance(_metrics(x_init=0.0, theta_init=15.0, theta_later=40.0),
                         N_nuc=1.0e5, a_contact=a_contact, a_pl=a_pl)
    assert not r.passed and not r.checks["x_init_in_window"]
    assert r.checks["inclined"] and r.checks["turned"]

    # 2) 不倾斜（|θ_init| < 10°）
    r = judge_acceptance(_metrics(x_init=a_contact, theta_init=5.0, theta_later=40.0),
                         N_nuc=1.0e5, a_contact=a_contact, a_pl=a_pl)
    assert not r.passed and not r.checks["inclined"]
    assert r.checks["x_init_in_window"] and r.checks["turned"]

    # 3) 不转向（|θ_later − θ_init| < 20°）
    r = judge_acceptance(_metrics(x_init=a_contact, theta_init=15.0, theta_later=25.0),
                         N_nuc=1.0e5, a_contact=a_contact, a_pl=a_pl)
    assert not r.passed and not r.checks["turned"]
    assert r.checks["x_init_in_window"] and r.checks["inclined"]


def test_acceptance_fails_when_not_nucleated():
    """T6c：未起裂时三条都不成立，passed=False（不静默当成通过）。"""
    from pfwire.clamp_case import judge_acceptance
    r = judge_acceptance(None, N_nuc=None, a_contact=2.917e-3, a_pl=2.389e-3)
    assert not r.passed
    assert not any(r.checks.values())
    assert r.N_nuc is None
    assert "未起裂" in r.details


def test_acceptance_accepts_either_contact_edge():
    """T6d：裂纹可出现在任一侧接触边缘（±a_contact 都算）。"""
    from pfwire.clamp_case import judge_acceptance
    a_contact, a_pl = 2.917e-3, 2.389e-3
    for x in (a_contact, -a_contact, a_contact - 0.5 * a_pl, -a_contact + 0.5 * a_pl):
        r = judge_acceptance(_metrics(x_init=x, theta_init=15.0, theta_later=40.0),
                             N_nuc=1.0e5, a_contact=a_contact, a_pl=a_pl)
        assert r.checks["x_init_in_window"], "x_init=%g 应落在 ±a_pl 窗口内" % x


# --------------------------------------------------------------------------
# T6 装配：用小网格走通 build_clamp_case（布局 C 的 145 万单元留给验收运行）
# --------------------------------------------------------------------------
def _tiny_config(tmp_path, kappa_sens=None):
    """把 clamp_base.yaml 的网格换成小盒，其余原样。"""
    import yaml
    raw = yaml.safe_load((ROOT / "configs/clamp_base.yaml").read_text(encoding="utf-8"))
    # 小网格：直接给 mesh 段（不走 from_config 的布局 C）
    raw["mesh"] = {"from_config": "configs/clamp_mesh.yaml", "layout": "C",
                   "workdir": str(tmp_path / "mesh"),
                   "tiny": True}
    raw["output"]["dir"] = str(tmp_path / "out")
    if kappa_sens is None:
        raw["contact"].pop("kappa_sens", None)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def test_build_clamp_case_assembles_all_parts(tmp_path):
    """T6e：build_clamp_case 能装出全部部件（网格、接触、载荷、求解器、采样网格）。"""
    from pfwire.clamp_case import build_clamp_case
    case = build_clamp_case(_tiny_config(tmp_path))
    assert case.clamp is not None and case.loads is not None
    assert case.solver is not None and case.state is not None
    assert case.driver is not None and case.grid is not None
    # 塑性 + 接触都挂上了
    assert case.solver.plastic is not None
    assert case.solver.contact is case.clamp
    # cycle-jump 带了 shakedown（带塑性时必填，否则构造即报错）
    assert case.driver.config.shakedown is not None
    # 采样间距必须 <= h_fine（DamageGrid 的校验）
    assert case.config["sampling"]["spacing"] <= 0.5e-6


def test_working_delta_search_reaches_W_target(tmp_path):
    """T6f：割线法在真实接触求解器上搜到目标反力（1% 容差内）。

    ⚠ 目标反力取**本网格在 δ₀ 处实测值的一半**，不用 clamp_base.yaml 的
    W' = 1.75e6：那个值对应布局 C（145 万单元、接触区分辨到 0.5 µm），
    tiny 网格（2059 单元、h = 50 µm）上根本给不出这么大的反力，割线法会
    一路外推出界——那是网格不匹配，不是搜索算法错。布局 C 上的真实验证
    留给 T6 的验收运行。
    """
    from pfwire.clamp_case import build_clamp_case, find_working_delta, solve_contact_at
    case = build_clamp_case(_tiny_config(tmp_path))
    # 先测 δ₀ 处的反力，取它的一半当目标（保证在 W(δ) 的值域内）
    W_at_delta0 = solve_contact_at(case, float(case.config["loading"]["delta0"]))
    W_target = 0.5 * W_at_delta0
    delta = find_working_delta(case, W_target, delta0=17.0e-6, rtol=1.0e-2, max_iter=20)
    W = case.clamp.reaction(case.solver.u)
    assert abs(W - W_target) <= 1.0e-2 * W_target, \
        "W = %.6g 未在 1%% 内达到 W' = %.6g（δ = %.4g）" % (W, W_target, delta)
    assert case.delta == delta


def test_kappa_sensitivity_reports_reaction_change(tmp_path):
    """T6g：κ 敏感性给出接触反力的相对变化（有限值，且两个 κ 的 W 不同）。"""
    from pfwire.clamp_case import (build_clamp_case, find_working_delta,
                                   kappa_sensitivity, solve_contact_at)
    case = build_clamp_case(_tiny_config(tmp_path))
    # 同 T6f：目标反力取本网格实测值的一半（W' = 1.75e6 在 tiny 网格上不可达）
    W0 = solve_contact_at(case, float(case.config["loading"]["delta0"]))
    find_working_delta(case, 0.5 * W0, delta0=17.0e-6, rtol=1.0e-2, max_iter=20)
    s = kappa_sensitivity(case, 1.0)
    assert s["kappa"] == 10.0 and s["kappa_sens"] == 1.0
    assert np.isfinite(s["rel"]) and s["rel"] >= 0.0
    # 换 κ 后反力必须真的变了（否则说明敏感性根本没换接触对象）
    assert s["W"] != s["W_sens"], "换 κ 后接触反力逐位相同，敏感性没有真正重解"


def test_run_clamp_case_produces_result_with_acceptance(tmp_path):
    """T6h：run_clamp_case 能推进并写出含验收判据的 result.json。"""
    import json
    from pfwire.clamp_case import (build_clamp_case, find_working_delta,
                                   run_clamp_case, solve_contact_at)
    case = build_clamp_case(_tiny_config(tmp_path))
    # 小网格上跑不动完整疲劳：把 N_max 压到很小，走几步就停
    case.config["loading"]["cycle"]["N_max"] = 5.0
    case.config["loading"]["cycle"]["dN_max"] = 2.0
    case.driver.config = _replace(case.driver.config, N_max=5.0, dN_max=2.0)
    # 同 T6f：目标反力取本网格实测值的一半
    W0 = solve_contact_at(case, float(case.config["loading"]["delta0"]))
    find_working_delta(case, 0.5 * W0, delta0=17.0e-6, rtol=1.0e-2, max_iter=20)
    result = run_clamp_case(case)
    assert result["status"] in ("completed", "runout", "not_converged", "ratcheting")
    assert "acceptance" in result and "shakedown_tags" in result
    assert result["acceptance"]["checks"]  # 判据齐全
    out = __import__("pathlib").Path(case.config["_outdir"]) / "result.json"
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8"))["case_type"] == "clamp"


def _replace(cfg, **kw):
    """dataclass.replace 的薄封装（frozen dataclass）。"""
    import dataclasses
    return dataclasses.replace(cfg, **kw)
