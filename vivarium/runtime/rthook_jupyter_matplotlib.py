"""
PyInstaller runtime hook for vivarium-jupyter: fix matplotlib inline backend.

Root cause
----------
ipykernel sets MPLBACKEND='module://matplotlib_inline.backend_inline' in the
process environment before any user code runs.  When the user (or any code)
later does `import matplotlib_inline.backend_inline`, the following chain fires:

  matplotlib_inline/__init__.py:1
    → from .backend_inline import ...          # starts loading backend_inline
  matplotlib_inline/backend_inline.py:6
    → import matplotlib                        # first matplotlib import
  matplotlib/__init__.py:1299
    → rcParams['backend'] = os.environ['MPLBACKEND']   # reads the env var
  matplotlib/__init__.py:774  RcParams.__setitem__
    → validate_backend('module://matplotlib_inline.backend_inline')
    → importlib.import_module('matplotlib_inline.backend_inline')
    → the module is STILL BEING LOADED (circular), so attribute checks fail
    → ValueError

Fix
---
In the runtime hook (which runs before ipykernel and before any user code):

1. Temporarily override MPLBACKEND to 'Agg'.
2. Import matplotlib   → initialises with Agg, no validation error.
3. Import matplotlib_inline.backend_inline → loads fully while matplotlib
   already uses Agg; no re-entrant validation problem.
4. Restore the original MPLBACKEND value.

From this point on, both modules are fully initialised in sys.modules.
When ipykernel later does rcParams['backend'] = 'module://...' (step 4 above),
validate_backend calls importlib.import_module('matplotlib_inline.backend_inline'),
finds the *complete* module in sys.modules, and succeeds.
"""
import os

_prev_mplbackend = os.environ.get('MPLBACKEND')
os.environ['MPLBACKEND'] = 'Agg'

try:
    import matplotlib               # noqa: F401 — initialise with Agg
    import matplotlib_inline.backend_inline  # noqa: F401 — fully load now
except Exception:
    # If the pre-load fails (e.g. a missing transitive dependency), fall back
    # to patching validate_backend so it does not attempt to import module://
    # backends during validation.  The real ImportError will surface later,
    # at the point where the backend is actually activated, with a clearer
    # message than the misleading "not a valid value" ValueError.
    try:
        import matplotlib.rcsetup as _rcsetup

        _orig_validate = _rcsetup.validate_backend

        def _patched_validate_backend(s):
            if isinstance(s, str) and s.lower().startswith('module://'):
                return s
            return _orig_validate(s)

        _rcsetup.validate_backend = _patched_validate_backend
    except Exception:
        pass
finally:
    if _prev_mplbackend is None:
        os.environ.pop('MPLBACKEND', None)
    else:
        os.environ['MPLBACKEND'] = _prev_mplbackend
