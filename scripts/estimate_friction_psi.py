"""估算：加上 Coulomb 摩擦切向力后，ψ⁺ 能抬高多少（D-68 问题 20）。

**为什么要估算**：6.5 当前工况是**无摩擦**法向接触（PLAN.md:116-117 的已知偏离，
D-67）。实测塑性安定后掩码内 r_max = 0.9994，全场差 0.06% 够不着起裂门槛
（D-68 问题 19）。微动疲劳的核心驱动是接触斑内的**循环切向微滑**，
它在当前工况里被显式关掉了。本估算回答：把它加回来，ψ⁺ 会变多少？

**⚠ 这是解析量级估算，不是 FE 解。** 目的是判断「值不值得投入做摩擦接触」，
不是给论文的数字。所有结论必须标为估算。

**口径与出处**：

- 法向 Hertz 压力：p(x) = p0 √(1 − x²/a²)，p0 = 2.15 σ_y（D-68/D-50 的饱和压力）
- 摩擦切向（Mindlin 部分滑移）：q(x) 的分布 Johnson, *Contact Mechanics* (1985) §8.4；
  全滑上界 q(x) = μ p(x)。部分滑移时粘着半径 c/a = √(1 − Q/(μP))。
- 表面点的应力状态（接触斑内，平面应变，自由表面面内无约束）：
      σ = [[0, τ], [τ, −p]]，τ = μ p（全滑峰值，x = 0 处）
- 次表面最大剪应力（Hertz 无摩擦经典结果）：τ_max = 0.31 p0 @ z ≈ 0.48a；
  有摩擦时表面 τ = μ p0 成为竞争者（μ = 0.7 时表面更大）。
- ψ⁺ 用项目的 spectral 分解（`energy_split_3d`，D-65）：
      ψ⁺ = ½λ⟨tr ε⟩₊² + G(⟨e₁⟩₊² + ⟨e₂⟩₊² + ⟨e₃⟩₊²)

**为什么要算表面中心点**：它不是 PLAN 预期的起裂位置（那是接触边缘），
但它是**切向牵引力的峰值点**，给出 ψ⁺ 的量级上界。
起裂位置的讨论见输出末尾。

**材料与工况参数全部从 configs / DECISIONS 读**，不手写魔数。

用法：python scripts/estimate_friction_psi.py
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_params():
    mat = yaml.safe_load((ROOT / "configs/material_al1350.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / "configs/clamp_base_c_1um.yaml").read_text(encoding="utf-8"))
    meshcfg = yaml.safe_load((ROOT / cfg["mesh"]["from_config"]).read_text(encoding="utf-8"))
    return mat, cfg, meshcfg


def spectral_psi_plus_2d(sxx, syy, sxy, E, nu):
    """平面应变下的 spectral ψ⁺（与 energy_split_3d 同口径，D-65）。

    σ 为面内 2×2（本估算的接触应力在 x–z 平面，故 dim = 2）。
    平面应变：ε_zz = 0，σ_zz = ν(σ_xx + σ_yy)。
    ε_zz 进谱分解（`energy_split_3d` 用 e3 = ε_zz），但 ε_zz = 0 时它不贡献
    ⟨e₃⟩₊²，只通过 λ⟨tr ε⟩₊² 的 tr 进 —— 而 tr ε = e1 + e2 + 0。
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    # 主应力（面内 2×2）
    tr2 = sxx + syy
    disc = math.sqrt(max(0.0, ((sxx - syy) / 2.0) ** 2 + sxy ** 2))
    s1, s2 = tr2 / 2.0 + disc, tr2 / 2.0 - disc
    szz = nu * (s1 + s2)                     # 平面应变
    # 主应变（Hooke，主轴系）
    e1 = (s1 - nu * (s2 + szz)) / E
    e2 = (s2 - nu * (s1 + szz)) / E
    e3 = 0.0
    tr_eps = e1 + e2 + e3
    psi = 0.5 * lam * max(tr_eps, 0.0) ** 2
    psi += G * (max(e1, 0.0) ** 2 + max(e2, 0.0) ** 2 + max(e3, 0.0) ** 2)
    return psi, (e1, e2, e3), (s1, s2, szz)


def f_alpha(abar, alpha_T, kappa):
    """Carrara 2020 Eq. 38（logarithmic），与 fatigue_degradation 同式。"""
    if abar <= alpha_T:
        return 1.0
    cutoff = alpha_T * 10.0 ** (1.0 / kappa)
    if abar >= cutoff:
        return 0.0
    return (1.0 - kappa * math.log10(abar / alpha_T)) ** 2


def psi_plus_elastic_plastic(sxx, syy, sxy, E, nu, sigma_y, H):
    """弹塑性下的 spectral ψ⁺：J2 各向同性线性硬化的**径向返回映射**。

    ⚠ 为什么需要这个：纯弹性估算在钳口区**完全失效** —— p0 = 2.15σ_y 远超屈服，
    弹性算出的 ε = σ/E 大 2.87 倍，与 FE 实测（r_max = 0.9994）差同量级。
    PLAN §1.4 的「塑性是必需项」正是这个意思。

    算法（与 pfwire.plasticity.return_map 同式，D-63）：
      - 平面应变，ε_zz = 0；塑性不可压 => ε_p,zz = −(ε_p,xx + ε_p,yy)
      - 试算弹性应变 ε_e,tr = ε − ε_p,n；偏量 s_tr，σ_eq,tr = √(1.5)‖s_tr‖
      - 屈服判据 F = σ_eq,tr − (σ_y + H·p_n)；F > 0 时
            Δγ = F / (3G + H)，ε_p ← ε_p,n + Δγ·√1.5·(s_tr/‖s_tr‖)
      - ψ⁺ 用返回映射后的 ε_e（D-65：不用试探应变）
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))

    # 由应力反推总应变（弹性 Hooke，作为**输入**；返回映射再把塑性剥出来）
    # 平面应变：σ_zz = ν(σ_xx + σ_yy)
    szz = nu * (sxx + syy)
    exx = (sxx - nu * (syy + szz)) / E
    eyy = (syy - nu * (sxx + szz)) / E
    exy = sxy / (2.0 * G)
    ezz = 0.0

    # 三维主应力/应变（面内 x–z 平面：这里 (x, y) 对应 (切向, 法向)）
    # 试算：ε_p,n = 0（静力估算，无历史）
    e_tr = np.array([exx, eyy, 0.5 * exy * 2.0 * 0.0])  # 占位，下面直接用张量
    eps = np.array([[exx, 0.5 * exy], [0.5 * exy, eyy]])
    ep_n = np.zeros((2, 2))
    p_n = 0.0

    ee = eps - ep_n
    tr = np.trace(ee)
    I2 = np.eye(2)
    s_tr = 2.0 * G * (ee - tr / 2.0 * I2)
    norm = math.sqrt(max(np.sum(s_tr * s_tr), 1.0e-30))
    sig_eq_tr = math.sqrt(1.5) * norm
    F = sig_eq_tr - (sigma_y + H * p_n)
    if F > 0.0:
        dgamma = F / (3.0 * G + H)
        n = s_tr / norm
        ep_new = ep_n + math.sqrt(1.5) * dgamma * n
        p_new = p_n + dgamma
    else:
        ep_new = ep_n
        p_new = p_n
    ee_new = eps - ep_new

    # 三维谱分解（ε_zz = 0；ε_p,zz = −(ε_p,xx+ε_p,yy) 由 ee_new 的三维嵌入隐含）
    # ee_new 是 2×2，嵌入 3×3：ee3 = [[ee_xx, ee_xy, 0], [ee_xy, ee_yy, 0], [0,0,ee_zz]]
    # ee_zz = −(ee_xx + ee_yy)（塑性不可压 + ε_zz = 0 => ε_e,zz = −ε_p,zz = ε_p,xx+ε_p,yy）
    ee_xx, ee_yy = ee_new[0, 0], ee_new[1, 1]
    ee_xy = ee_new[0, 1]
    ee_zz = -(ep_new[0, 0] + ep_new[1, 1])
    ee3 = np.array([[ee_xx, ee_xy, 0.0], [ee_xy, ee_yy, 0.0], [0.0, 0.0, ee_zz]])

    # 主应变（对称 3×3，块对角：面内 2×2 块 + ee_zz）
    tr2 = ee_xx + ee_yy
    disc = math.sqrt(max(0.0, ((ee_xx - ee_yy) / 2.0) ** 2 + ee_xy ** 2))
    e1, e2 = tr2 / 2.0 + disc, tr2 / 2.0 - disc
    e3 = ee_zz
    tr_eps = e1 + e2 + e3
    psi = 0.5 * lam * max(tr_eps, 0.0) ** 2
    psi += G * (max(e1, 0.0) ** 2 + max(e2, 0.0) ** 2 + max(e3, 0.0) ** 2)
    return psi, (e1, e2, e3), (F > 0.0, p_new)


def main():
    mat, cfg, meshcfg = load_params()
    E = float(mat["elastic"]["E"])
    nu = float(mat["elastic"]["nu"])
    sigma_y = float(mat["plastic"]["sigma_y0"])
    sigma_c = float(mat["fracture"]["sigma_c"])
    ell = float(mat["fracture"]["ell"])
    alpha_T = float(mat["fatigue"]["alpha_T"])
    kappa = float(mat["fatigue"]["kappa"])
    a_contact = float(meshcfg["inputs"]["a_contact"])
    R_clamp = float(cfg["contact"]["R_clamp"])
    sigma_a = float(cfg["loading"]["sigma_a"])
    # 饱和压力：D-50/D-68，p0 = 2.15 σ_y
    H = float(mat["plastic"]["H"])
    p0 = 2.15 * sigma_y
    coef = sigma_c ** 2 / E                       # AT1 门槛 f·σ_c²/E
    abar_now = 4.2859e9                            # D-68 问题 19：N=14352.4 实测
    f_now = f_alpha(abar_now, alpha_T, kappa)
    thresh = f_now * coef

    print("=" * 78)
    print("估算：Coulomb 摩擦切向力能把 ψ⁺ 抬高多少（D-68 问题 20）")
    print("=" * 78)
    print("")
    print("参数（全部来自 configs / DECISIONS，非手写）")
    print("  E = %.4g Pa, ν = %.2f, G = %.4g Pa" % (E, nu, E / (2 * (1 + nu))))
    print("  σ_y = %.4g Pa, σ_c = %.4g Pa, ℓ = %.4g m" % (sigma_y, sigma_c, ell))
    print("  a_contact = %.4g m, R_clamp = %.4g m" % (a_contact, R_clamp))
    print("  σ_a = %.4g Pa（端部弯曲幅）" % sigma_a)
    print("  饱和压力 p0 = 2.15 σ_y = %.4g Pa（D-50/D-68）" % p0)
    print("  塑性硬化 H = %.4g Pa（D-53，派生估计）" % H)
    print("  ᾱ(N=14352.4) = %.6e -> f = %.6f -> 门槛 f σ_c²/E = %.4e Pa"
          % (abar_now, f_now, thresh))
    print("  （对照：实测掩码内 ψ⁺ max = 8.03e4 Pa，r_max = 0.9994）")

    # ---------- 1) 无摩擦的基线（现在的工况）----------
    print("")
    print("-" * 78)
    print("1) 无摩擦基线（当前 6.5 工况）：表面中心只有法向压缩")
    print("-" * 78)
    psi_nf, e_nf, pl_nf = psi_plus_elastic_plastic(0.0, -p0, 0.0, E, nu, sigma_y, H)
    print("  表面中心 σ = [[0, 0], [0, −p0]]（三轴压缩，弹塑性）")
    print("  屈服判据 F > 0？%s；累积等效塑性应变 p = %.4e" % (pl_nf[0], pl_nf[1]))
    print("  主应变 e1 = %.4e, e2 = %.4e, e3 = %.4e" % tuple(e_nf))
    print("  ψ⁺ = %.4e Pa  ->  r = %.4f  %s"
          % (psi_nf, 2 * psi_nf / thresh,
             "(低于门槛)" if 2 * psi_nf / thresh < 1 else "(越过门槛)"))
    print("")
    print("  ⚠ 标定检查：FE 实测（D-68 问题 19）掩码内 ψ⁺ max = 8.03e4 Pa、")
    print("    r_max = 0.9994。本估算给 %.3e Pa、r = %.4f —— 比值 %.2f 倍。"
          % (psi_nf, 2 * psi_nf / thresh, psi_nf / 8.03e4))
    print("    若偏离大，说明「表面中心 + 局部返回映射」这个近似不足以定量，")
    print("    只有**相对抬升**（摩擦 vs 无摩擦）可用。")

    # ---------- 2) 加上摩擦切向 ----------
    print("")
    print("-" * 78)
    print("2) 加上 Coulomb 摩擦：表面中心的切向剪应力 τ = μ p0")
    print("-" * 78)
    print("  μ      τ[MPa]   ψ⁺[Pa]     r = 2ψ⁺/门槛   ψ⁺/ψ⁺_无摩擦   起裂?")
    for mu in (0.1, 0.3, 0.5, 0.7, 1.0):
        tau = mu * p0
        psi_f, e_f, pl_f = psi_plus_elastic_plastic(0.0, -p0, tau, E, nu, sigma_y, H)
        r = 2 * psi_f / thresh
        print("  %4.2f  %7.1f  %.4e  %10.3f  %10.2f     %s"
              % (mu, 1e-6 * tau, psi_f, r, psi_f / psi_nf,
                 "是" if r > 1 else "否"))
    print("")
    print("  PLAN.md:115 取 μ = 0.7；文献区间 0.3~1.0。")

    # ---------- 3) 部分滑移（Mindlin）----------
    print("")
    print("-" * 78)
    print("3) 部分滑移：Q < μP 时切向峰值的折扣（Mindlin）")
    print("-" * 78)
    print("  Q/(μP)  粘着半径 c/a   τ_peak/p0   ψ⁺[Pa]     r")
    mu = 0.7
    for ratio in (1.0, 0.9, 0.7, 0.5, 0.3):
        # 部分滑移的切向峰值：Johnson (1985) Eq. 8.46，q 在 x=c 处达
        # q_max = μ p0 (1 − c²/a²)^{1/2} · (某因子)；近似取 τ_peak = μ p0 · sqrt(ratio)
        # （c/a = sqrt(1−ratio)，滑移区内的切向水平）
        tau = mu * p0 * math.sqrt(ratio)
        psi_f, _, _ = psi_plus_elastic_plastic(0.0, -p0, tau, E, nu, sigma_y, H)
        c_a = math.sqrt(max(0.0, 1.0 - ratio))
        print("  %5.2f   %8.3f   %8.4f  %.4e  %7.3f"
              % (ratio, c_a, mu * math.sqrt(ratio), psi_f, 2 * psi_f / thresh))

    # ---------- 4) 与弯曲叠加 ----------
    print("")
    print("-" * 78)
    print("4) 与端部弯曲 σ_a 叠加（弯曲是轴向 σ_x）")
    print("-" * 78)
    print("  弯曲幅 σ_a = %.4g Pa。弯曲单独：ψ⁺ = σ_a²/(2E) = %.4e Pa（门槛的 %.1f%%）"
          % (sigma_a, sigma_a ** 2 / (2 * E),
             100 * sigma_a ** 2 / (2 * E) / thresh))
    for mu in (0.3, 0.7):
        tau = mu * p0
        # 叠加：面内 σ_x = σ_a（拉），σ_z = −p0，τ = μ p0
        for sxx in (0.0, sigma_a, -sigma_a):
            psi_s, e_s, pl_s = psi_plus_elastic_plastic(sxx, -p0, tau, E, nu, sigma_y, H)
            print("  μ=%.1f, σ_x=%+8.1f MPa: ψ⁺ = %.4e Pa, r = %7.3f"
                  % (mu, 1e-6 * sxx, psi_s, 2 * psi_s / thresh))
        print("")

    # ---------- 5) 结论 ----------
    print("-" * 78)
    print("5) 结论")
    print("-" * 78)
    mu = 0.7
    tau = mu * p0
    psi_f, _, _ = psi_plus_elastic_plastic(0.0, -p0, tau, E, nu, sigma_y, H)
    print("  无摩擦（当前）：ψ⁺ = %.3e Pa，r = %.4f —— 差 %.1f%% 够不着门槛"
          % (psi_nf, 2 * psi_nf / thresh, 100 * (1 - 2 * psi_nf / thresh)))
    print("  μ=0.7 全滑   ：ψ⁺ = %.3e Pa，r = %.3f —— **越过门槛 %.1f 倍**"
          % (psi_f, 2 * psi_f / thresh, 2 * psi_f / thresh))
    print("  抬升倍数     ：%.2f 倍（%.3e -> %.3e Pa）"
          % (psi_f / psi_nf, psi_nf, psi_f))
    print("")
    print("  即：摩擦切向力把 ψ⁺ 抬高约 %.1f 倍，足以越过门槛 %.1f 倍。"
          % (psi_f / psi_nf, 2 * psi_f / thresh))
    print("  对比 6.5 的缺口（0.06%%）：摩擦带来的增量是它的 %.0f 倍量级。"
          % (psi_f / psi_nf / 0.0006))
    print("")
    print("  ⚠ 但这是**表面中心**（切向峰值点），不是 PLAN 预期的起裂位置")
    print("    （接触边缘 ±a = ±%.4g mm）。微动疲劳的经典起裂位置是**粘滑边界**"
          % (1e3 * a_contact))
    print("    x = ±c（Mindlin 的粘着区边缘），那里 q(x) 有平方根奇异性。")
    print("    本估算未解该奇点的应力场，故「越过门槛多少倍」是**量级**而非定位。")
    print("")
    print("  ⚠ 另注：Carrara 累积的是 max(Δψ⁺, 0)。切向力若随弯曲相位反向，")
    print("    每周期的 Δψ⁺ 是峰谷差，这才是累积量 —— 本估算只给峰值 ψ⁺，")
    print("    真实累积率需 FE 解出切向力随相位的历程。")
    print("")
    print("=" * 78)
    print("判断：摩擦缺失足以解释 6.5 的 0.06%% 缺口，且余量巨大（%.0f 倍）。"
          % (psi_f / psi_nf / 0.0006))
    print("      接入 Coulomb 摩擦是 6.5 的必要条件，不是可选优化。")
    print("=" * 78)


if __name__ == "__main__":
    main()
