# app.py  ✧ 2025-08-01  ▸ UPDATED for higher-accuracy defaults
# Streamlit front-end for the Math-LLM evaluation pipeline
# ──────────────────────────────────────────────────────────
import os, re, json, csv, time, collections
from pathlib import Path
from typing import List, Dict, Tuple

import streamlit as st
from datasets import load_dataset

from math_fewshot  import sample_few_shots
from model_router   import query_model_stream
from reflection     import reflect_and_retry
from pot_executor   import run_python_block
from utils          import equivalent, generate_question_id

st.set_page_config(page_title="Math-LLM Evaluator", layout="wide")

# ─── regex helpers ───────────────────────────────────────────────
# Accept either “FINAL: <ans>” or “FINAL = <ans>”
FINAL_RE = re.compile(r"FINAL\s*[:=]\s*(.+)", re.I)

def parse_final(text: str) -> str | None:
    m = FINAL_RE.search(text)
    return m.group(1).strip() if m else None

def majority_vote(answers: List[str]) -> str:
    return collections.Counter(answers).most_common(1)[0][0]

# ─── prompt builder ──────────────────────────────────────────────
def build_prompt(problem: str,
                 few_shot: bool = True,
                 allow_pot: bool = False) -> str:
    """
    Construct the system/user prompt.

    Key changes:
      • Stronger instructions for symbolic answers
      • Encourage Program-of-Thought explicitly
      • FINAL line uses “=”
    """
    lines = [
        "You are a brilliant mathematician.",
        "Solve the problem step-by-step using clear symbolic reasoning.",
        "If you need calculation, embed it in ```python``` blocks; the code will be executed.",
        "On the last line write exactly:  FINAL = <answer>",
    ]

    if few_shot:
        for ex in sample_few_shots(3):
            lines.append(
                f"\nProblem: {ex['q']}\n"
                f"Solution: {ex['a']}\n"
                f"FINAL = {ex['a']}\n"
            )

    lines.append(f"\nProblem: {problem}\nSolution:")
    return "\n".join(lines)

def run_pot_in_reasoning(reasoning: str) -> str:
    """Execute each ```python ...``` block and insert result lines."""
    def _exec(match: re.Match[str]) -> str:
        code = match.group(1)
        result = run_python_block(code, timeout=3)
        return f"```python\n{code}\n# => {result}\n```"
    return re.sub(r"```python\n(.*?)```", _exec, reasoning, flags=re.S)

# ─── sidebar controls ────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Default model switched to DeepSeek-Math
    model_key = st.selectbox(
        "Model",
        ["deepseekmath:7b", "llama3:8b", "gemma:7b"],
        index=0,
    )

    # Higher self-consistency
    k = st.slider("Self-consistency (k)", 1, 7, 5)

    # Tighter sampling
    temperature = st.slider("Temperature", 0.0, 0.2, 0.05, 0.01)
    top_p       = st.slider("top_p",       0.1, 1.0, 0.9, 0.05)

    # Keep PoT and Reflection on by default
    allow_pot     = st.checkbox("Enable Program-of-Thought", value=True)
    allow_reflect = st.checkbox("Enable Reflection",        value=True)

    st.markdown("---")
    dataset_choice = st.radio(
        "Dataset source",
        ["Built-in MATH-500 (500 Qs)", "Upload custom (.csv or .jsonl)"],
        index=0,
    )
    uploaded = None
    if dataset_choice.startswith("Upload"):
        uploaded = st.file_uploader("Upload dataset", type=["jsonl", "csv"])

    if st.button("Run evaluation"):
        st.session_state.run_eval       = True
        st.session_state.dataset_choice = dataset_choice
        st.session_state.uploaded_file  = uploaded

# ─── main area ───────────────────────────────────────────────────
st.title("📐 Math-LLM Project — Dataset Evaluator")

if "run_eval" not in st.session_state:
    st.info("Choose a dataset and click **Run evaluation**.")
else:
    ds_choice = st.session_state.dataset_choice
    uploaded  = st.session_state.uploaded_file

    # ── load dataset ─────────────────────────────────────────────
    problems: List[Dict[str, str]] = []

    if ds_choice.startswith("Built-in"):
        hf_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        problems = [{"question": r["problem"], "answer": r["answer"]} for r in hf_ds]
    else:
        if uploaded is None:
            st.warning("Please upload a dataset first.")
            st.stop()
        suffix = Path(uploaded.name).suffix
        if suffix == ".csv":
            reader = csv.DictReader(uploaded.getvalue().decode().splitlines())
            problems = [{"question": r["question"], "answer": r["answer"]} for r in reader]
        elif suffix == ".jsonl":
            for line in uploaded:
                row = json.loads(line)
                problems.append({"question": row["question"], "answer": row["answer"]})
        else:
            st.error("Unsupported file type.")
            st.stop()

    total = len(problems)
    st.write(f"### Dataset loaded: **{total}** problems")

    # ── evaluation loop ─────────────────────────────────────────
    correct = 0
    answer_log: List[Tuple[str, str, str]] = []
    progress_bar = st.progress(0, text="Starting…")
    start_time = time.time()

    for idx, row in enumerate(problems, start=1):
        q, gold = row["question"], row["answer"]
        votes = []

        for _ in range(k):
            prompt = build_prompt(q, few_shot=True, allow_pot=allow_pot)
            reasoning = "".join(
                query_model_stream(
                    model_key,
                    prompt,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
            if allow_pot:
                reasoning = run_pot_in_reasoning(reasoning)

            pred = parse_final(reasoning) or ""
            if allow_reflect:
                pred, reasoning = reflect_and_retry(
                    model_key,
                    q,
                    reasoning,
                    pred,
                    lambda pb: build_prompt(pb, True, allow_pot),
                    max_retries=4,          # ⬅️  increased
                )
            votes.append(pred)

        majority = majority_vote(votes)
        if equivalent(majority, gold):
            correct += 1

        answer_log.append((q, gold, majority))
        progress_bar.progress(
            idx / total,
            text=f"Problem {idx}/{total}  •  Current acc: {correct}/{idx}",
        )

    elapsed = time.time() - start_time
    progress_bar.empty()

    # ── results ────────────────────────────────────────────────
    st.success(
        f"**Accuracy: {correct}/{total} = {100*correct/total:.2f}%** "
        f"(elapsed {elapsed/60:.1f} min)"
    )

    with st.expander("Show detailed log"):
        for p, g, a in answer_log:
            st.markdown(
                f"- **Q**: {p}\n"
                f"  - **Gold**: `{g}`\n"
                f"  - **Pred**: `{a}`\n"
                f"  - **Correct**: {equivalent(a, g)}"
            )

    st.session_state.pop("run_eval")  # ready for next run