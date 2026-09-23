import importlib, sys
for name in ("dolfinx", "dolfinx_contact", "dolfinx_mpc", "ufl", "basix"):
    try:
        m = importlib.import_module(name)
        print("OK   %-16s %s" % (name, getattr(m, "__version__", "?")))
    except Exception as e:
        print("MISS %-16s %s" % (name, type(e).__name__))
