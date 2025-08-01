# model_router.py  ✧ 2025-07-31
import ollama
from typing import Generator

MODEL_MAP = {
    "deepseekmath:7b": "t1c/deepseek-math-7b-rl:latest",
    "llama3:8b":       "llama3:8b",
    "gemma:7b":        "gemma:7b",
}

def _auto_batch() -> int:
    """Heuristic: RTX 40-series laptop ≈ 8 GB → 64 tokens/batch."""
    return 64

def query_model_stream(model_key: str,
                       prompt: str,
                       temperature: float = 0.0,
                       top_p: float = 0.9,
                       n_batch: int | None = None
                       ) -> Generator[str, None, None]:
    slug = MODEL_MAP.get(model_key, model_key)
    stream = ollama.chat(
        model=slug,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options={
            "temperature": temperature,
            "top_p": top_p,
            "n_batch": n_batch or _auto_batch(),
        },
    )
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            yield chunk["message"]["content"]