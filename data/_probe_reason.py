# -*- coding: utf-8 -*-
"""查 PETSc 的 SNESConvergedReason 真实定义（不凭记忆）。"""
from petsc4py import PETSc
for name in dir(PETSc.SNES.ConvergedReason):
    if not name.startswith("_"):
        print("%-28s %d" % (name, getattr(PETSc.SNES.ConvergedReason, name)))
