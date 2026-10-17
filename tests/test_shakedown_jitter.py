"""decay_jitter 的测试（D-68 问题 17）。

背景：周期塑性应变增量 e_k 是 FEM 在离散网格上量出来的，会带数值抖动。
若序列本意是逐周期精确减半（e_k ≈ 0.5·e_{k-1}），某个 e_k 可能读出 0.502
而不是 0.499 —— 严格判据 `cur <= decay_ratio * prev` 会因这 0.4% 的超标
把整条衰减序列误判成 not_decaying（实测：[..., 1.263e-08, 6.325e-09]，
第三项超出 0.5 阈值 0.16%）。

decay_jitter 只放宽「衰减」的边界判定，不放宽「不衰减」的定义。
"""
from __future__ import annotations

import pytest

from pfwire.cyclejump import ShakedownConfig, classify_plastic_history


def _cfg(jitter):
    return ShakedownConfig(zero_tol=1.0e-10, decay_ratio=0.5, n_check=3,
                           decay_jitter=jitter)


def test_strict_jitter_zero_keeps_false_positive():
    """decay_jitter=0 时，[5.156e-08, 2.527e-08, 1.263e-08, 6.325e-09] 被判 not_decaying。

    这是要修掉的误判：第三项 6.325e-09 超出 0.5*1.263e-08 约 0.16%。
    """
    e = [5.156e-08, 2.527e-08, 1.263e-08, 6.325e-09]
    assert classify_plastic_history(e, _cfg(0.0)) == "not_decaying"


def test_jitter_1pct_accepts_halving_sequence():
    """decay_jitter=0.01 时，同一序列判 decaying。

    第三项比值 = 6.325e-09 / 1.263e-08 = 0.501，在 0.5*(1+0.01) = 0.505 之内。
    """
    e = [5.156e-08, 2.527e-08, 1.263e-08, 6.325e-09]
    assert classify_plastic_history(e, _cfg(0.01)) == "decaying"


def test_jitter_does_not_mask_true_ratchet():
    """jitter 不能把真正的棘轮（e_k ≈ e_{k-1} 或更大）误判成 decaying。"""
    # e_k / e_{k-1} = 1.0 和 1.2，远超 0.5*(1+jitter)
    e = [1.0e-08, 1.0e-08, 1.2e-08]
    assert classify_plastic_history(e, _cfg(0.01)) == "not_decaying"
    e = [1.0e-08, 1.0e-08, 1.0e-08]  # 持平
    assert classify_plastic_history(e, _cfg(0.01)) == "not_decaying"


def test_jitter_does_not_mask_slow_decay():
    """衰减太慢（比值 0.7）的序列也不应被 jitter 放行。"""
    e = [1.0e-08, 7.0e-09, 4.9e-09]  # 比值 0.7、0.7
    assert classify_plastic_history(e, _cfg(0.01)) == "not_decaying"


def test_jitter_bounds():
    """decay_jitter 必须在 [0, 0.5) 内。"""
    with pytest.raises(ValueError, match="decay_jitter"):
        _cfg(-0.01)
    with pytest.raises(ValueError, match="decay_jitter"):
        _cfg(0.5)


def test_jitter_none_is_zero():
    """默认 decay_jitter=0.0，严格判据。"""
    cfg = ShakedownConfig(zero_tol=1.0e-10, decay_ratio=0.5, n_check=3)
    assert cfg.decay_jitter == 0.0
    e = [5.156e-08, 2.527e-08, 1.263e-08, 6.325e-09]
    assert classify_plastic_history(e, cfg) == "not_decaying"
