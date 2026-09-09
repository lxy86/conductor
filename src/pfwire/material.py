"""材料参数与参数换算（PLAN.md Phase 0）。

本模块只做**标量换算**，不 import dolfinx，因此可脱离求解器单测。
弹性张量与 Amor/spectral 拉压分解属 Phase 1.1，届时加在本模块的 UFL 侧。

公式出处
--------
- Lamé 关系、平面应变模量 E/(1-nu^2)：
  Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed., 第 2-3 章。
- 接触等效模量 E* 与等效半径 R：PLAN.md 第 1 节
  （Johnson, *Contact Mechanics*, 1985, 第 4.2 节；线接触约定 a^2 = 4PR/(pi E*)）。
- G_c <-> K_IC（平面应变 Irwin 关系）：PLAN.md 第 1 节。
- AT1 模型 sigma_c = sqrt(3 E G_c / (8 ell))：PLAN.md 第 1/2 节
  （Pham, Amor, Marigo & Maurini 2011；Tanne et al. 2018, JMPS 110:80-99）。
- Amor 分解用的 d 维体积模量 K_d = lambda + 2 mu / d：
  Amor, Marigo & Maurini 2009, JMPS 57:1209-1229（d = 2 为平面应变特化）。

单位：全部 SI（Pa, m, J/m^2, Pa*m^0.5）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

__all__ = [
    "ElasticProperties",
    "FractureProperties",
    "Material",
    "FRACTURE_MODES",
    "lame_parameters",
    "plane_strain_modulus",
    "bulk_modulus",
    "contact_effective_modulus",
    "contact_effective_radius",
    "contact_pair_properties",
    "gc_from_kic",
    "kic_from_gc",
    "at1_gc_from_sigma_c_ell",
    "at1_ell_from_gc_sigma_c",
    "at1_sigma_c_from_gc_ell",
    "resolve_fracture_properties",
    "load_material",
]

#: ``fracture.mode`` 的合法取值。三个断裂参数 (G_c, ell, sigma_c) 由 AT1 关系联系，
#: 只有两个独立；mode 指明「哪两个是输入、哪个被导出」。见 DECISIONS.md D-2。
FRACTURE_MODES: tuple[str, ...] = (
    "from_sigma_c_and_ell",
    "from_gc_and_ell",
    "from_gc_and_sigma_c",
)

Counterpart = Literal["aluminum", "rigid"]


def _coerce(obj: Any, field: str, caster) -> None:
    """就地把 frozen dataclass 的字段转成目标类型。

    必要性：PyYAML 走 YAML 1.1，指数必须带显式符号，`69.0e9` / `500e6` / `1.0e8`
    会被解析成**字符串**而不是 float（PLAN.md 第 5 节的 schema 就是这种写法）。
    验证函数里的 float() 只在算式中生效，不会改写字段本身，因此在此统一 coerce，
    保证任何构造路径下字段都是数值。Phase 1 的工况 config 载入必须复用同样的处理。
    """
    object.__setattr__(obj, field, caster(getattr(obj, field)))


def _require_positive(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError("%s 必须为有限正数，收到 %r" % (name, value))
    return v


def _require_poisson(nu: float) -> float:
    v = float(nu)
    # 平面应变下 nu -> 0.5 使 lambda 发散，故上界取严格小于 0.5
    if not 0.0 <= v < 0.5:
        raise ValueError("nu 必须在 [0, 0.5) 内（平面应变），收到 %r" % (nu,))
    return v


# --------------------------------------------------------------------------
# 弹性常数换算
# --------------------------------------------------------------------------
def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    """返回 (lambda, mu)。三维 Lamé 常数，平面应变直接沿用、不需修正。"""
    E = _require_positive(E, "E")
    nu = _require_poisson(nu)
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lmbda, mu


def plane_strain_modulus(E: float, nu: float) -> float:
    """平面应变等效模量 E' = E/(1-nu^2)。

    用于：梁理论对照（Phase 0 悬臂梁）、Irwin 关系、接触等效模量。
    """
    E = _require_positive(E, "E")
    nu = _require_poisson(nu)
    return E / (1.0 - nu * nu)


def bulk_modulus(E: float, nu: float, dim: int = 2) -> float:
    """Amor 分解所需的 d 维体积模量 K_d = lambda + 2 mu / d。

    dim=2 即平面应变（tr(eps) 为二维迹），dim=3 为通常的 K。Amor et al. 2009。
    """
    if dim not in (2, 3):
        raise ValueError("dim 只能是 2 或 3，收到 %r" % (dim,))
    lmbda, mu = lame_parameters(E, nu)
    return lmbda + 2.0 * mu / dim


# --------------------------------------------------------------------------
# 接触参数换算（PLAN.md 第 1 节）
# --------------------------------------------------------------------------
def contact_effective_modulus(
    E1: float, nu1: float, E2: float | None = None, nu2: float | None = None
) -> float:
    """E* = 1/[(1-nu1^2)/E1 + (1-nu2^2)/E2]。

    E2 为 None 或 inf 表示刚性对偶体，此时 E* = E1/(1-nu1^2)。
    """
    compliance = 1.0 / plane_strain_modulus(E1, nu1)
    if E2 is not None and math.isfinite(float(E2)):
        if nu2 is None:
            raise ValueError("给了 E2 就必须给 nu2")
        compliance += 1.0 / plane_strain_modulus(E2, nu2)
    return 1.0 / compliance


def contact_effective_radius(R_wire: float, counterpart: Counterpart) -> float:
    """等效半径 R：1/R = 1/R1 + 1/R2。

    counterpart="aluminum"：两根同直径铝丝相互接触 -> R = R_wire/2；
    counterpart="rigid"：铝丝对刚性平面 -> R = R_wire。PLAN.md 第 1 节。
    """
    R_wire = _require_positive(R_wire, "R_wire")
    if counterpart == "aluminum":
        return 0.5 * R_wire
    if counterpart == "rigid":
        return R_wire
    raise ValueError(
        "counterpart 只能是 'aluminum' 或 'rigid'，收到 %r" % (counterpart,)
    )


def contact_pair_properties(
    E: float, nu: float, R_wire: float, counterpart: Counterpart
) -> tuple[float, float]:
    """返回接触对的 (E*, R)。counterpart="aluminum" 时对偶体材料与试样相同。

    注：由 Hertz 的 a = sqrt(4PR/(pi E*)) 与 p0 = 2P/(pi a) 消去 P 可得
    a = 2 R p0 / E*，这是 DECISIONS.md D-1 里尺度论证的依据；该恒等式的
    实现与解析对拍属 loads.py（Phase 2.2）。
    """
    R_eff = contact_effective_radius(R_wire, counterpart)
    if counterpart == "aluminum":
        E_star = contact_effective_modulus(E, nu, E, nu)
    else:
        E_star = contact_effective_modulus(E, nu, None, None)
    return E_star, R_eff


# --------------------------------------------------------------------------
# 断裂参数换算：Irwin 与 AT1
# --------------------------------------------------------------------------
def gc_from_kic(K_IC: float, E: float, nu: float) -> float:
    """G_c = K_IC^2 (1-nu^2)/E（平面应变 Irwin）。PLAN.md 第 1 节。"""
    K_IC = _require_positive(K_IC, "K_IC")
    return K_IC * K_IC / plane_strain_modulus(E, nu)


def kic_from_gc(Gc: float, E: float, nu: float) -> float:
    """gc_from_kic 的逆：K_IC = sqrt(G_c E/(1-nu^2))。"""
    Gc = _require_positive(Gc, "Gc")
    return math.sqrt(Gc * plane_strain_modulus(E, nu))


def at1_gc_from_sigma_c_ell(E: float, sigma_c: float, ell: float) -> float:
    """AT1: G_c = 8 sigma_c^2 ell / (3 E)。"""
    E = _require_positive(E, "E")
    sigma_c = _require_positive(sigma_c, "sigma_c")
    ell = _require_positive(ell, "ell")
    return 8.0 * sigma_c * sigma_c * ell / (3.0 * E)


def at1_ell_from_gc_sigma_c(E: float, Gc: float, sigma_c: float) -> float:
    """AT1: ell = 3 E G_c / (8 sigma_c^2)。PLAN.md 第 1 节。"""
    E = _require_positive(E, "E")
    Gc = _require_positive(Gc, "Gc")
    sigma_c = _require_positive(sigma_c, "sigma_c")
    return 3.0 * E * Gc / (8.0 * sigma_c * sigma_c)


def at1_sigma_c_from_gc_ell(E: float, Gc: float, ell: float) -> float:
    """AT1: sigma_c = sqrt(3 E G_c / (8 ell))。Tanne et al. 2018。"""
    E = _require_positive(E, "E")
    Gc = _require_positive(Gc, "Gc")
    ell = _require_positive(ell, "ell")
    return math.sqrt(3.0 * E * Gc / (8.0 * ell))


# --------------------------------------------------------------------------
# 数据类
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ElasticProperties:
    """线弹性常数（平面应变）。"""

    E: float
    nu: float

    def __post_init__(self) -> None:
        _coerce(self, "E", lambda v: _require_positive(v, "E"))
        _coerce(self, "nu", _require_poisson)

    @property
    def lame(self) -> tuple[float, float]:
        return lame_parameters(self.E, self.nu)

    @property
    def lmbda(self) -> float:
        return self.lame[0]

    @property
    def mu(self) -> float:
        return self.lame[1]

    @property
    def E_plane_strain(self) -> float:
        return plane_strain_modulus(self.E, self.nu)

    def bulk(self, dim: int = 2) -> float:
        return bulk_modulus(self.E, self.nu, dim)


@dataclass(frozen=True)
class FractureProperties:
    """AT1 断裂参数。三者由 AT1 关系锁定，只有两个独立。

    ``derived`` 记录哪一个是被导出的；``kic_equivalent`` 是由 G_c 反算的等效
    K_IC，仅用于论文中报告数量级（见 DECISIONS.md D-2）。
    """

    Gc: float
    ell: float
    sigma_c: float
    derived: str
    kic_equivalent: float

    def __post_init__(self) -> None:
        for field in ("Gc", "ell", "sigma_c", "kic_equivalent"):
            _coerce(self, field, lambda v, f=field: _require_positive(v, f))


@dataclass(frozen=True)
class Material:
    """一份材料卡。``alpha_T`` 允许为 None（标定前），取用时须显式 require。"""

    name: str
    elastic: ElasticProperties
    fracture: FractureProperties
    sigma_u: float
    alpha_T: float | None
    sources: dict[str, Any]

    def __post_init__(self) -> None:
        _coerce(self, "sigma_u", lambda v: _require_positive(v, "sigma_u"))
        if self.alpha_T is not None:
            _coerce(self, "alpha_T", lambda v: _require_positive(v, "alpha_T"))

    def require_alpha_T(self) -> float:
        """取 alpha_T，未标定时直接报错（DECISIONS.md D-3：不用默认值蒙）。"""
        if self.alpha_T is None:
            raise ValueError(
                "材料 %r 的 alpha_T 尚未标定（Phase 2.3）。请先运行 "
                "scripts/calibrate_alphaT.py 并把结果写回材料 yaml，"
                "或在工况 yaml 的 fatigue.alpha_T 中显式给值。" % (self.name,)
            )
        return self.alpha_T


# --------------------------------------------------------------------------
# 三角关系求解与 yaml 载入
# --------------------------------------------------------------------------
def resolve_fracture_properties(
    E: float,
    nu: float,
    *,
    mode: str,
    Gc: float | None = None,
    ell: float | None = None,
    sigma_c: float | None = None,
    rtol: float = 1e-6,
) -> FractureProperties:
    """按 mode 用两个输入导出第三个断裂参数。

    若第三个量也给了（非 None），则校验一致性，不一致直接报错——防止 yaml 里
    留下自相矛盾的参数组（PLAN.md 第 1 节原文的 G_c/ell/sigma_c 组合就是过定的，
    见 DECISIONS.md D-2）。
    """
    if mode not in FRACTURE_MODES:
        raise ValueError("fracture.mode 只能取 %r，收到 %r" % (FRACTURE_MODES, mode))

    required = {
        "from_sigma_c_and_ell": ("sigma_c", "ell"),
        "from_gc_and_ell": ("Gc", "ell"),
        "from_gc_and_sigma_c": ("Gc", "sigma_c"),
    }[mode]
    given: dict[str, float | None] = {"Gc": Gc, "ell": ell, "sigma_c": sigma_c}
    missing = [k for k in required if given[k] is None]
    if missing:
        raise ValueError("mode=%r 需要给出 %r，缺少 %r" % (mode, required, missing))

    derived = next(k for k in ("Gc", "ell", "sigma_c") if k not in required)
    if derived == "Gc":
        value = at1_gc_from_sigma_c_ell(E, sigma_c, ell)
    elif derived == "ell":
        value = at1_ell_from_gc_sigma_c(E, Gc, sigma_c)
    else:
        value = at1_sigma_c_from_gc_ell(E, Gc, ell)

    if given[derived] is not None:
        supplied = float(given[derived])
        if abs(supplied - value) > rtol * abs(value):
            raise ValueError(
                "断裂参数过定且不自洽：mode=%r 由 AT1 关系导出 %s=%.6g，"
                "但 yaml 里给的是 %.6g（相对偏差 %.3e > rtol=%.1e）。"
                "请删掉冗余项或改 mode。"
                % (
                    mode,
                    derived,
                    value,
                    supplied,
                    abs(supplied - value) / abs(value),
                    rtol,
                )
            )
    resolved: dict[str, float] = {
        k: float(v) for k, v in given.items() if v is not None
    }
    resolved[derived] = value
    return FractureProperties(
        Gc=resolved["Gc"],
        ell=resolved["ell"],
        sigma_c=resolved["sigma_c"],
        derived=derived,
        kic_equivalent=kic_from_gc(resolved["Gc"], E, nu),
    )


def load_material(path: str | Path) -> Material:
    """读材料 yaml（如 configs/material_al1350.yaml）。

    顶层键：name, elastic{E,nu}, strength{sigma_u}, fracture{mode,...},
    fatigue{alpha_T}, sources{}。缺键即报错，不给默认值（硬约束 3）。
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError("%s 不是 mapping" % (path,))

    for key in ("name", "elastic", "strength", "fracture", "fatigue"):
        if key not in raw:
            raise KeyError("%s 缺少顶层键 %r" % (path, key))

    elastic = ElasticProperties(E=raw["elastic"]["E"], nu=raw["elastic"]["nu"])
    frac_cfg = dict(raw["fracture"])
    mode = frac_cfg.pop("mode", None)
    if mode is None:
        raise KeyError(
            "%s 的 fracture 段缺少 mode（可选：%r）" % (path, FRACTURE_MODES)
        )
    unknown = set(frac_cfg) - {"Gc", "ell", "sigma_c", "rtol"}
    if unknown:
        raise KeyError("%s 的 fracture 段有未知键 %r" % (path, sorted(unknown)))
    fracture = resolve_fracture_properties(
        elastic.E,
        elastic.nu,
        mode=mode,
        Gc=frac_cfg.get("Gc"),
        ell=frac_cfg.get("ell"),
        sigma_c=frac_cfg.get("sigma_c"),
        rtol=float(frac_cfg.get("rtol", 1e-6)),
    )
    alpha_T = raw["fatigue"].get("alpha_T", None)
    return Material(
        name=str(raw["name"]),
        elastic=elastic,
        fracture=fracture,
        sigma_u=_require_positive(raw["strength"]["sigma_u"], "sigma_u"),
        alpha_T=None if alpha_T is None else _require_positive(alpha_T, "alpha_T"),
        sources=dict(raw.get("sources", {})),
    )
