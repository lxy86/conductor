"""零插桩：F_q 的根到底在哪？纯 numpy 复算源码公式（D-68 问题 33 续 21）。

复算 smooth_clip（与 clamp_contact.smooth_clip 逐行同源）：
    clip_η(x; lim) = x − ½·( g + √(g² + η²) )·sign(x),  g = |x| − lim
复算 F_q（与 clamp_contact.monolithic_residual 同源）：
    F_q = q − clip_η(q − ε_t·s; μp, η)

**结论**（见 DECISIONS.md D-68 问题 33 续 21）：
  F_q 在整个盒内**都不为零**，而且离盒边界越近越大（q=0.99μp 时已到 2.8e6）。
  滑移态 q=±μp 处的地板恒为 η/2。η→0 才严格归零。

  ⇒ 续 19 的「滑移态要求 s 与 q 反号」是**错的推论**：F_q 对 s 的依赖
    只有 2.9e-4 的相对量级，主导项是 η 光滑化偏差。
    真正的机理是 **clip_η 在盒内不恒等**（docstring 自己就写了
    「盒内偏差 O(η²/(4u))，贴近盒面处最大」）—— 贴近盒面正是滑移态。

用法：bash data/_run_probe_fq.sh
"""
import numpy as np


def smooth_clip(x, lim, eta):
    """与 pfwire.clamp_contact.smooth_clip 逐行同源。"""
    g = np.abs(x) - lim
    return x - 0.5 * (g + np.sqrt(g * g + eta * eta)) * np.sign(x)


def F_q(q, s, eps_t, mu_p, eta):
    """与 pfwire.clamp_contact.monolithic_residual 的 F_q 同源。"""
    return q - smooth_clip(q - eps_t * s, mu_p, eta)


MU_P = 1.358e8
EPS_T = 1.0e8
ETA_REL = 0.05


def main() -> None:
    eta = ETA_REL * MU_P
    print("mu_p = %.3e, eta = %.3e (eta_rel=%.2f)" % (MU_P, eta, ETA_REL))
    print()

    print("=== 检验 1：F_q 对 s 的依赖有多强？（滑移态 q=+mu_p）===")
    print("%-12s %-16s" % ("s", "F_q"))
    vals = []
    for s in (-1e-5, -1e-6, 0.0, 1e-6, 1e-5):
        f = F_q(MU_P, s, EPS_T, MU_P, eta)
        vals.append(f)
        print("%-12.1e %-16.6e" % (s, f))
    print("  => 绝对变化 %.3e，相对 %.3e（**极弱**）" % (max(vals) - min(vals),
                                              (max(vals) - min(vals)) / abs(np.mean(vals))))
    print("  => F_q 在滑移态基本**不看 s**；续 19 的「必须 s<0」不成立")
    print()

    print("=== 检验 2：盒内各处 F_q 的值（s=0）—— 是否只有边界附近非零？ ===")
    print("%-14s %-16s %-14s" % ("q/mu_p", "F_q", "距边界"))
    for q_rel in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0):
        q = q_rel * MU_P
        f = F_q(q, 0.0, EPS_T, MU_P, eta)
        print("%-14.3f %-16.6e %-14.3e" % (q_rel, f, MU_P - q))
    print("  => **整个盒内都不归零**，越贴近边界越大（0.99μp 已到 2.8e6）")
    print("     这正是 smooth_clip docstring 写的「盒内偏差 O(η²/(4u))，贴近盒面最大」")
    print()

    print("=== 检验 3：地板是否恒为 eta/2？（q=+mu_p, s=0）===")
    print("%-12s %-16s %-16s %-10s" % ("eta_rel", "F_q", "eta/2", "比值"))
    for eta_rel in (0.05, 0.01, 0.001, 1e-4, 1e-6, 0.0):
        e = eta_rel * MU_P
        f = F_q(MU_P, 0.0, EPS_T, MU_P, e)
        ratio = f / (e / 2) if e > 0 else float("nan")
        print("%-12.1e %-16.6e %-16.6e %-10.6f" % (eta_rel, f, e / 2, ratio))
    print("  => 边界处地板严格 = eta/2；eta=0 时归零")
    print()

    print("=== 结论 ===")
    print("  (1) 滑移态没有根 —— **成立**（续 19 的结论对）")
    print("  (2) 但机理是 **clip_η 在盒内不恒等**（贴边处偏差最大），")
    print("      不是续 19 说的「s 与 q 必须反号、这个组合无解」")
    print("  (3) 修法应从 **η 光滑化** 入手（或换残差定义），不是从符号入手")
    print("  (4) 续 12 的 η 扫描没发现它，是因为只看 ||F|| 总量：")
    print("      地板只作用在少数贴边单元上，被总量稀释（实测 eta_rel=5e-4 时")
    print("      地板 3.4e4 与 ||F|| 4.5e4 同量级，比值 0.755）")


if __name__ == "__main__":
    main()
