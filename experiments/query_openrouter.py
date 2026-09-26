"""Utility for querying models via OpenRouter API."""

import json
import time
from pathlib import Path
from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL


def get_client():
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


def query_model(
    client,
    model: str,
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    seed: int | None = None,
) -> dict:
    """Send a single query and return the response with metadata."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    elapsed = time.time() - t0

    choice = response.choices[0]
    return {
        "model": model,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "seed": seed,
        "response": choice.message.content,
        "finish_reason": choice.finish_reason,
        "elapsed_s": round(elapsed, 3),
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }


def save_results(results: list[dict], outpath: str | Path):
    """Append results as JSONL."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
