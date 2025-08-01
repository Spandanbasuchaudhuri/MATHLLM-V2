# reflection.py  ✧ 2025-08-02  ▸ BUG-FIX: safe hint extraction
from model_router import query_model_stream

CRITIC_PROMPT = """
You are a strict math critic.
Respond ONLY with YES if the FINAL answer is certainly correct;
otherwise respond NO and give one-sentence hint.

Problem:
{problem}

Proposed reasoning:
{reasoning}

FINAL = {answer}
""".strip()


def _last_nonblank_line(text: str) -> str | None:
    """Return the last non-empty line or None."""
    for line in reversed(text.strip().splitlines()):
        if line.strip():
            return line.strip()
    return None


def reflect_and_retry(
    model_key: str,
    problem: str,
    reasoning: str,
    answer: str,
    prompt_builder,
    max_retries: int = 4,
):
    """
    Returns (best_answer, combined_reasoning).
    Stops early if the critic gives no usable hint.
    """
    all_reasoning = reasoning

    for attempt in range(max_retries + 1):
        critic = "".join(
            query_model_stream(
                model_key,
                CRITIC_PROMPT.format(
                    problem=problem,
                    reasoning=all_reasoning,
                    answer=answer,
                ),
                temperature=0.0,
            )
        ).strip()

        # ✅ Correct?
        if critic.upper().startswith("YES"):
            return answer, all_reasoning

        # 🔍 Try to extract a hint
        hint = _last_nonblank_line(critic)
        if not hint or hint.upper().startswith("YES"):
            # No hint → give up further retries
            return answer, all_reasoning

        # Retry with the hint
        new_prompt = prompt_builder(problem) + f"\nHINT: {hint}"
        answer = "".join(
            query_model_stream(
                model_key,
                new_prompt,
                temperature=0.5,   # exploration
            )
        )
        all_reasoning += f"\n\n# RETRY {attempt + 1} WITH HINT\n{hint}"

    return answer, all_reasoning