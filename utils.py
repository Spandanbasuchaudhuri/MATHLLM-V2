# utils.py  ✧ 2025-07-31
import re, time
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

# ── misc ───────────────────────────────────────────────────────────────
def generate_question_id() -> int:
    return int(time.time())

# ── answer normalisation & equivalence ─────────────────────────────────
_WS = re.compile(r"\s+")

def _canon(expr: str):
    expr = expr.strip()
    try:
        return sp.nsimplify(parse_expr(expr, evaluate=False))
    except Exception:
        # Fallback: maybe vector like "[1, 2, 3]"
        items = [parse_expr(x) for x in re.split(r"[,\s]+", expr.strip("[]"))]
        return sp.ImmutableList(map(sp.nsimplify, items))

def equivalent(raw_a: str | None, raw_b: str | None) -> bool:
    if not raw_a or not raw_b:
        return False

    try:
        return sp.simplify(_canon(raw_a) - _canon(raw_b)) == 0
    except Exception:
        pass

    # fallback string equality ignoring whitespace
    return _WS.sub("", raw_a) == _WS.sub("", raw_b)