# sympy_solver.py  ✧ 2025-08-01  ▸ MINOR TWEAK (stricter check)
import sympy
from sympy.parsing.sympy_parser import parse_expr

def sympy_check(step_text: str) -> bool:
    """
    Basic sanity check for a single reasoning step.
    Now stricter: if an equality is present, verify it with SymPy.
    """
    # Trivial wrong-answer veto example
    if "2 + 2 = 5" in step_text:
        return False

    # Handle a single equality
    if "=" in step_text:
        parts = step_text.split("=")
        if len(parts) == 2:
            lhs, rhs = (p.strip() for p in parts)
            try:
                lhs_expr = parse_expr(lhs, evaluate=False)
                rhs_expr = parse_expr(rhs, evaluate=False)
                return sympy.simplify(lhs_expr - rhs_expr) == 0
            except Exception:
                return False   # parsing failed → treat as suspect

    # Accept if no explicit equality to check
    return True