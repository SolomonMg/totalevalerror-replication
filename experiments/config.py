"""Experiment configuration and shared constants."""

import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SEED = 42

# --- Judge models (these evaluate the responses) ---
JUDGE_MODELS = {
    "frontier": "openai/gpt-4o",
    "cheap": "google/gemini-2.0-flash-001",
    "mid": "anthropic/claude-haiku-4.5",
}

# Approximate per-1M-token pricing (input/output) for cost tracking
MODEL_PRICING = {
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
    "anthropic/claude-haiku-4.5": {"input": 0.80, "output": 4.00},
    "google/gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "openai/gpt-oss-120b": {"input": 0.039, "output": 0.19},
    "arcee-ai/trinity-large-preview:free": {"input": 0.0, "output": 0.0},
    # SUT models (used in generate_sut_responses.py cost tracking)
    "anthropic/claude-opus-4-5": {"input": 15.0, "output": 75.0},
    "openai/gpt-5.1": {"input": 2.0, "output": 8.0},
    "deepseek/deepseek-chat-v3.1": {"input": 0.27, "output": 1.10},
    "x-ai/grok-4.1-fast": {"input": 3.0, "output": 15.0},
    # Tier-matched sensitivity analysis models
    "openai/gpt-5.4": {"input": 2.50, "output": 15.00},
    "google/gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "anthropic/claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "google/gemma-4-31b-it": {"input": 0.10, "output": 0.30},
    "deepseek/deepseek-v3.2": {"input": 0.26, "output": 0.38},
    "qwen/qwen3.5-397b-a17b": {"input": 0.39, "output": 2.34},
}

# Model used for generating prompt paraphrases (not used in ideology experiment)
PARAPHRASE_MODEL = "anthropic/claude-haiku-4.5"

# --- Temperature grid ---
TEMPERATURES = [0.0, 0.7, 1.0]

# --- Stage configurations ---
STAGES = {
    "smoke": {
        "n_items": 10,
        "models": ["google/gemini-2.0-flash-001"],
        "temperatures": [0.7],
        "n_prompt_variants": 1,
        "n_replications": 3,
        "description": "Verify API infrastructure, JSONL format, parsing logic",
    },
    "pilot": {
        "n_items": 30,
        "models": list(JUDGE_MODELS.values()),
        "temperatures": TEMPERATURES,
        "n_prompt_variants": 3,
        "n_replications": 5,
        "description": "Feasibility of variance decomposition; convergence check",
    },
    "full": {
        "n_items": 150,
        "models": list(JUDGE_MODELS.values()),
        "temperatures": TEMPERATURES,
        "n_prompt_variants": 5,
        "n_replications": 8,
        "description": "Publication-quality variance decomposition with CIs",
    },
}

# --- Safety judge models (binary SAFE/UNSAFE classification) ---
SAFETY_JUDGE_MODELS = {
    "gemini-3-flash": "google/gemini-3-flash-preview",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "trinity-large": "arcee-ai/trinity-large-preview:free",
}

# Safety experiment stages
SAFETY_STAGES = {
    "smoke": {
        "n_items": 10,
        "models": ["google/gemini-3-flash-preview"],
        "temperatures": [0.7],
        "n_prompt_variants": 1,
        "n_replications": 3,
        "description": "Verify safety scoring infrastructure",
    },
    "pilot": {
        "n_items": 30,
        "models": list(SAFETY_JUDGE_MODELS.values()),
        "temperatures": TEMPERATURES,
        "n_prompt_variants": 3,
        "n_replications": 5,
        "description": "Safety variance decomposition feasibility check",
    },
    "full": {
        "n_items": 144,
        "models": list(SAFETY_JUDGE_MODELS.values()),
        "temperatures": TEMPERATURES,
        "n_prompt_variants": 5,
        "n_replications": 8,
        "description": "Full safety variance decomposition",
    },
}

# --- Tier-matched sensitivity analysis (safety domain) ---
SENSITIVITY_TIERS = {
    "original": [
        "openai/gpt-4o",
        "google/gemini-2.0-flash-001",
        "anthropic/claude-haiku-4.5",
    ],
    "closed_frontier": [
        "openai/gpt-5.4",
        "google/gemini-3.1-pro-preview",
        "anthropic/claude-opus-4-6",
    ],
    "open_weight": [
        "openai/gpt-oss-120b",
        "google/gemma-4-31b-it",
        "deepseek/deepseek-v3.2",
    ],
}

SENSITIVITY_ALL_MODELS = [m for tier in SENSITIVITY_TIERS.values() for m in tier]

SENSITIVITY_STAGE = {
    "smoke": {
        "n_items": 3,
        "models": SENSITIVITY_ALL_MODELS,
        "temperatures": [0.7],
        "n_prompt_variants": 1,
        "n_replications": 1,
        "description": "Verify all 9 models return valid SAFE/UNSAFE",
    },
    "pilot": {
        "n_items": 30,
        "models": SENSITIVITY_ALL_MODELS,
        "temperatures": TEMPERATURES,
        "n_prompt_variants": 3,
        "n_replications": 5,
        "description": "Tier-matched judge sensitivity analysis (safety)",
    },
}

# --- API settings ---
MAX_TOKENS = 16  # Short responses: single number (Likert) or letter (pairwise)
REQUEST_DELAY_S = 0.2  # Delay between API calls (rate limiting)
MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 2.0  # Exponential backoff base
