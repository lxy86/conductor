import sys
sys.path.insert(0, "/root/pf-wire")
import numpy as np
from mpi4py import MPI
from pfwire import cyclejump as cj, phasefield as pf
from tests.test_phasefield import UniaxialStrainCase

S, NT = 0.8, 2.0
case = UniaxialStrainCase()
sc = case.material.fracture.sigma_c
E = case.material.elastic.E
psi_max = 0.5 * (S * sc) ** 2 / E
a_T = NT * psi_max / (2.0 / S - 1.0)
print("sigma_c=%.4g E=%.4g psi_max=%.6g a_T=%.6g" % (sc, E, psi_max, a_T))
print("闭式：起裂循环数 =", cj.homogeneous_cycles_to_damage(S, a_T, case.material))
state = cj.FatigueState(case.msh, case.solver.model,
                        alpha_source="undegraded", alpha_T=a_T)
case.solver = pf.PhaseFieldSolver(case.msh, case.solver.model, bcs_u=case.make_bcs,
                                  f_alpha=state.f_alpha, config=case.solver.config)
state.bind_displacement(case.solver.u)
for c in range(1, 7):
    case.u_applied.value = S * case.strain_c * case.L
    case.solver.solve_displacement()
    state.accumulate()
    case.solver.solve_damage()
    case.solver.commit_step()
    print("  仅峰值: cycle=%d  alpha_bar=%.6g (=%.3f psi_max)  max_d=%.6g"
          % (c, state.max_alpha_bar(), state.max_alpha_bar() / psi_max,
             case.solver.max_damage()))
