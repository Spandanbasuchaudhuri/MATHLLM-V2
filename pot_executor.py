# pot_executor.py  ✧ 2025-07-31
# ──────────────────────────────────────────────────────────────────────
# Safe (yet useful) SymPy sandbox for Program-of-Thought execution.

import ast, signal, threading, sympy as sp

class ExecTimeout(Exception):
    """Raised when the snippet exceeds its CPU-time budget."""

def _timeout_handler(signum=None, frame=None):
    raise ExecTimeout

# ──────────────────────────────────────────────────────────────────────
def run_python_block(code: str, timeout: int = 3):
    """
    Execute a short SymPy-aware snippet in a restricted namespace.
    Returns the value left in '_' or a status string such as TIMEOUT/ERROR.
    """
    tree = ast.parse(code, mode="exec")

    # Block dangerous statement types
    unsafe = (ast.Import, ast.ImportFrom, ast.Global, ast.With, ast.AsyncWith,
              ast.Try, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    if any(isinstance(n, unsafe) for n in ast.walk(tree)):
        raise ValueError("Unsupported statement in PoT sandbox")

    gbl = {"__builtins__": __builtins__, "sympy": sp, "sp": sp}
    loc = {}

    # Cross-platform timeout
    if hasattr(signal, "SIGALRM"):           # POSIX
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        cancel = lambda: signal.alarm(0)
    else:                                    # Windows
        timer = threading.Timer(timeout, _timeout_handler)
        timer.start()
        cancel = timer.cancel

    try:
        exec(compile(tree, "<pot>", "exec"), gbl, loc)
        cancel()
        return loc.get("_", "OK")
    except ExecTimeout:
        cancel()
        return "TIMEOUT"
    except Exception as e:
        cancel()
        return f"ERROR: {e}"