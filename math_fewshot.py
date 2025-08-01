# math_fewshot.py  ✧ 2025-07-31
import random, textwrap

FEW_SHOT_EXAMPLES = [
    {"q": "Evaluate ∫_0^1 3x² dx.",                "a": "1"},
    {"q": "Solve 2x + 5 = 9 for x.",               "a": "2"},
    {"q": "Derivative of f(x)=x².",                "a": "2*x"},
    {"q": "Simplify (3/4)/(1/8).",                 "a": "6"},
    {"q": "How many subsets does a 5-element set have?", "a": "32"},
    {"q": "gcd(24, 36).",                          "a": "12"},
    {"q": "Area of a circle with radius 3.",       "a": "9*pi"},
    {"q": "Is 17 prime?",                          "a": "True"},
    {"q": "Find a₁₀ if a_n = 3n – 2.",             "a": "28"},
    {"q": "P(heads) on a fair coin toss.",         "a": "1/2"},
]

def sample_few_shots(k: int = 3):
    """Return k diverse (q, a) dicts."""
    return random.sample(FEW_SHOT_EXAMPLES, k)