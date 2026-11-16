"""Download Terminal-Bench 2.0 per-trial outcomes, the paper's run configs and the reported scores, for R.

Sources (all read 2026-11-08):
  trials       huggingface.co/datasets/yoonholee/terminalbench-trajectories at commit 04e8940 (2026-03-09),
               Apache-2.0. A third-party copy of the trials shown on tbench.ai: one row per trial with task,
               agent, model, reward (0/1), timing and the trajectory text. We drop the trajectory text.
  run configs  github.com/laude-institute/terminal-bench-experiments at commit 0433864 (2026-01-15), Apache-2.0.
               configs/tb2/*.yaml are the Harbor job files of the Terminal-Bench 2.0 paper (Merrill et al., 2026,
               arXiv 2601.11868v1, section 3.4). They name each harness, model and harness version, and set no
               temperature or reasoning effort. The paper (section 3.3) says models ran at the provider's
               default effort.
  leaderboard  the JSON that tbench.ai/leaderboard/terminal-bench/2.0 loads (POST to the site's public
               leaderboard-read function; no license stated). Accuracy and 95% half-width per agent x model.
  paper        Table 2 of arXiv 2601.11868v1, parsed from the PDF with pdftotext.

The copy stores 4,952 trials twice (once with a trial ID and start time, once without), all in cells with ~10
trials per task. The pairs agree on reward and duration, so we keep one row per (agent, model, trial_name).
Accuracy is computed as on the leaderboard (terminal-bench-experiments sql/compute_accuracy.sql): the mean over
the 89 tasks of each task's mean reward. The half-width is 1.96 sqrt(sum_t p_t (1 - p_t) / (n_t - 1)) / 89,
which holds the 89 tasks fixed and counts only trial-to-trial noise.

Design (column paper_design): the agent x model pairs that the paper's six job files list, excluding
openhands-opus-4-5.yaml (a later OpenHands 1.1.0 run). These ran 2025-10-31 to 2025-11-05 where the copy has a
start time. Every design cell must match its leaderboard row to 0.01 points. Table 2 of the paper is reported
beside it; it differs from the leaderboard for all OpenHands 0.60.0 cells and the xAI cells (column paper_diff).

Settings each harness sends by default, from code reading on 2026-11-08 (Harbor 3a5b57e, the last commit before
the runs; OpenHands 0.60.0; mini-SWE-agent 1.14.4; Codex rust-v0.53.0; Gemini CLI v0.11.3; Claude Code 2.0.31 npm
bundle; LiteLLM 1.77.7 - 1.79.x). Harbor's wrappers inject no temperature, effort or thinking setting.
  Terminus 2      temperature 0.7 (dropped for GPT-5 models); no effort or thinking; Anthropic max_tokens 4096
  mini-SWE-agent  temperature 0.0 (dropped for GPT-5 models); no effort or thinking; Anthropic max_tokens 4096
  OpenHands       reasoning_effort "high" for GPT-5, -mini, -nano (config default, llm_config.py L182-185);
                  Gemini 2.5 Pro thinking budget 128 tokens; Gemini 2.5 Flash effort high (budget 4096);
                  Sonnet/Haiku 4.5 temperature dropped (API default 1.0), no thinking; Opus 4.1 temperature 0.0,
                  thinking disabled; Grok and open-weight models temperature 0.0
  Claude Code     API-default temperature; thinking off in print mode (no --max-thinking-tokens); max_tokens 32000
  Codex CLI       effort not sent (API default); reasoning summary "auto"
  Gemini CLI      temperature 0, top_p 1, thinkingBudget -1 (dynamic)
So the paper's "provider default (medium)" holds for OpenAI models under Terminus 2, mini-SWE-agent and Codex
CLI only. Settings match within these groups: GPT-5 family under {Terminus 2, mini-SWE-agent, Codex CLI}; Gemini
2.5 under {mini-SWE-agent, Gemini CLI}; Sonnet and Haiku 4.5 under {OpenHands, Claude Code}; Opus 4.1 and Grok 4
under {mini-SWE-agent, OpenHands}. Three claims re-read in the source (OpenHands effort default and Gemini Pro
budget, Terminus 2 temperature, mini-SWE-agent temperature); the rest rests on one reading.

Outputs: data/processed/tbench2_trials.csv (one row per deduplicated trial, all 109 agent x model cells)
         data/processed/tbench2_cells.csv  (one row per agent x model cell, with leaderboard and paper scores)
Run from the project root: experiments/.venv/bin/python -I experiments/fetch_terminalbench.py
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import urllib.request

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external"
OUT = ROOT / "data/processed"

HF_REPO = "yoonholee/terminalbench-trajectories"
HF_SHA = "04e8940f5b6736a7ce8d22224fe2f2af74163ed2"
HF_FILES = {
    "train-00000-of-00002.parquet": "e95bc8dc9f3e1b2f3c6c383d0a060064205d87e93e9ac9541847d1c511effdfa",
    "train-00001-of-00002.parquet": "00d61614054bedcbadda9974cef30fe1e07d1701e7126edfa210f99840ae99b2",
}
EXP_REPO = "laude-institute/terminal-bench-experiments"
EXP_SHA = "043386442be68526403431b2024f50d3080abb72"
EXP_CONFIGS = ["mini-swe-agent", "openhands", "proprietary", "terminus-closed", "terminus-open-slim",
               "terminus-open-fireworks"]
LB_URL = "https://ofhuhcpkvzjlejydnvyd.supabase.co/functions/v1/leaderboard-read"
LB_BODY = {"package": "terminal-bench/terminal-bench-2", "name": "2-0"}
LB_DATE = "2026-11-08"
ARXIV = "2601.11868v1"
N_TASKS = 89
COLS = ["task_name", "agent", "model", "reward", "duration_seconds", "trial_name", "trial_id", "started_at",
        "ended_at"]
# Table 2 names -> the copy's agent and model names
T2_AGENT = {"Codex CLI": "codex", "Terminus 2": "terminus-2", "Claude Code": "claude-code",
            "OpenHands": "openhands", "Mini-SWE-Agent": "mini-swe-agent", "Gemini CLI": "gemini-cli"}
T2_MODEL = {
    "GPT-5.2": "gpt-5.2@openai", "GPT-5": "gpt-5@openai", "GPT-5-Mini": "gpt-5-mini@openai",
    "GPT-5-Nano": "gpt-5-nano@openai", "Claude Opus 4.5": "claude-opus-4-5-20251101@anthropic",
    "Claude Sonnet 4.5": "claude-sonnet-4-5-20250929@anthropic", "Claude Opus 4.1": "claude-opus-4-1-20250805@anthropic",
    "Claude Haiku 4.5": "claude-haiku-4-5-20251001@anthropic", "Gemini 3 Pro": "gemini-3-pro-preview@gemini",
    "Gemini 3 Flash": "gemini-3-flash-preview@gemini", "Gemini 2.5 Pro": "gemini-2.5-pro@gemini",
    "Gemini 2.5 Flash": "gemini-2.5-flash@gemini", "Grok 4": "grok-4-0709@xai", "Grok Code Fast 1": "grok-code-fast-1@xai",
    "Kimi K2 Thinking": "moonshotai/Kimi-K2-Thinking@together_ai",
    "Kimi K2 Instruct": "moonshotai/Kimi-K2-Instruct-0905@together_ai",
    "Qwen 3 Coder 480B": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8@together_ai",
    "GPT-OSS-120B": "openai/gpt-oss-120b@together_ai", "GPT-OSS-20B": "openai/gpt-oss-20b@together_ai",
    "MiniMax M2": "accounts/fireworks/models/minimax-m2@fireworks_ai",
    "GLM 4.6": "accounts/fireworks/models/glm-4p6@fireworks_ai",
}


def fetch(url, dest, data=None):
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "research-script",
                                                          **({"Content-Type": "application/json"} if data else {})})
    with urllib.request.urlopen(req, timeout=300) as r:
        dest.write_bytes(r.read())
    return dest


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def config_model(name):
    """Harbor model name (provider/model) -> the copy's model@provider."""
    provider, model = name.split("/", 1)
    return f"{model}@{provider}"


# ---- download ----
hf_dir = RAW / "terminalbench_trajectories" / HF_SHA[:7]
for f, digest in HF_FILES.items():
    p = fetch(f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_SHA}/data/{f}", hf_dir / f)
    assert sha256(p) == digest, f
exp_dir = RAW / "terminal_bench_experiments" / EXP_SHA[:7]
for c in EXP_CONFIGS:
    fetch(f"https://raw.githubusercontent.com/{EXP_REPO}/{EXP_SHA}/configs/tb2/{c}.yaml", exp_dir / f"{c}.yaml")
lb_path = fetch(LB_URL, RAW / "terminalbench_leaderboard" / LB_DATE / "lb_2_0.json", json.dumps(LB_BODY).encode())
pdf = fetch(f"https://arxiv.org/pdf/{ARXIV}", RAW / "terminalbench_paper" / f"{ARXIV}.pdf")

# ---- trials ----
tr = pd.concat([pq.read_table(hf_dir / f, columns=COLS).to_pandas() for f in HF_FILES], ignore_index=True)
assert len(tr) == 52104
key = ["agent", "model", "trial_name"]
dup = tr[tr.duplicated(key, keep=False)]
assert (dup.groupby(key)[["reward", "duration_seconds"]].nunique() == 1).all().all()
tr["has_id"] = tr.trial_id != ""
tr = tr.sort_values(key + ["has_id"], ascending=[True, True, True, False]).drop_duplicates(key)
assert len(tr) == 52104 - 4952 and tr.reward.isin([0, 1]).all()
assert tr.task_name.nunique() == N_TASKS
tr["started_at"] = tr.started_at.replace("", None)

# ---- paper design from the job files ----
design = []
for c in EXP_CONFIGS:
    job = yaml.safe_load((exp_dir / f"{c}.yaml").read_text())
    assert job["n_attempts"] == 5
    for a in job["agents"]:
        kw = a.get("kwargs") or {}
        assert not ({"temperature", "reasoning_effort"} & set(kw)), (c, a)
        design.append(dict(agent=a["name"], model=config_model(a["model_name"]), config=c,
                           agent_version=str(kw.get("version", ""))))
design = pd.DataFrame(design)
assert not design.duplicated(["agent", "model"]).any()

# ---- cells ----
tk = tr.groupby(["agent", "model", "task_name"]).reward.agg(p="mean", n="size").reset_index()
tk["v"] = np.where(tk.n > 1, tk.p * (1 - tk.p) / (tk.n - 1), np.nan)
cells = tk.groupby(["agent", "model"]).agg(n_tasks=("task_name", "size"), n_trials=("n", "sum"),
                                           min_trials=("n", "min"), max_trials=("n", "max"),
                                           acc=("p", "mean"), sv=("v", "sum")).reset_index()
cells["acc"] *= 100
cells["pm95"] = 1.96 * np.sqrt(cells.sv) / N_TASKS * 100
dates = tr.groupby(["agent", "model"]).agg(first_start=("started_at", "min"), last_start=("started_at", "max"),
                                          share_dated=("started_at", lambda s: s.notna().mean())).reset_index()
for c in ["first_start", "last_start"]:
    dates[c] = dates[c].str[:10]
cells = cells.drop(columns="sv").merge(dates, on=["agent", "model"])
cells = cells.merge(design, on=["agent", "model"], how="left")
cells["paper_design"] = cells.config.notna()

# ---- leaderboard ----
lb = json.loads(lb_path.read_text())
assert lb["leaderboard"]["title"] == "Terminal-Bench 2.0"
lbr = pd.DataFrame([dict(agent=r["metadata"]["agent_name"], lb_agent_version=r["metadata"]["agent_version"],
                         lb_label=(r["metadata"]["agent_display"] or {}).get("label"),
                         model=f'{r["metadata"]["model_names"][0]}@{r["metadata"]["model_providers"][0].lower()}',
                         lb_n_models=len(r["metadata"]["model_names"]), lb_acc=r["metrics"]["accuracy"],
                         lb_pm95=r["metrics"].get("accuracy_ci95_half_width"))
                    for r in lb["rows"]])
lbr = lbr[lbr.lb_n_models == 1].drop(columns="lb_n_models")
d_lb = cells[cells.paper_design].merge(lbr, on=["agent", "model"], how="left")
assert d_lb.groupby(["agent", "model"]).size().eq(1).all() and d_lb.lb_acc.notna().all(), \
    d_lb[d_lb.lb_acc.isna()][["agent", "model"]]
assert (d_lb.acc - d_lb.lb_acc).abs().max() < 0.01
v = d_lb[d_lb.agent_version != ""]
assert (v.agent_version == v.lb_agent_version).all()
cells = cells.merge(d_lb[["agent", "model", "lb_acc", "lb_pm95", "lb_agent_version"]], on=["agent", "model"],
                    how="left")

# ---- paper Table 2 ----
txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
t2 = [m.groups() for m in re.finditer(r"^\s*(\S.*?)\s{2,}(Codex CLI|Terminus 2|Claude Code|OpenHands|Mini-SWE-Agent|"
                                      r"Gemini CLI)\s+([\d.]+)% ± ([\d.]+)%", txt, re.M)]
t2 = pd.DataFrame(t2, columns=["model", "agent", "paper_acc", "paper_pm95"])
assert len(t2) == 55
t2 = t2.assign(agent=t2.agent.map(T2_AGENT), model=t2.model.map(T2_MODEL),
               paper_acc=t2.paper_acc.astype(float), paper_pm95=t2.paper_pm95.astype(float))
assert t2.notna().all().all()
cells = cells.merge(t2, on=["agent", "model"], how="left")
assert cells.paper_acc.notna().sum() == 55
cells["paper_diff"] = cells.acc - cells.paper_acc

# ---- write ----
tr["paper_design"] = tr.set_index(["agent", "model"]).index.isin(
    cells[cells.paper_design].set_index(["agent", "model"]).index)
tr[["agent", "model", "task_name", "trial_name", "reward", "duration_seconds", "started_at", "paper_design"]] \
    .sort_values(["agent", "model", "task_name", "trial_name"]).to_csv(OUT / "tbench2_trials.csv", index=False)
cells.sort_values("acc", ascending=False).round(4).to_csv(OUT / "tbench2_cells.csv", index=False)

dc = cells[cells.paper_design]
print(f"{len(tr)} trials in {len(cells)} cells; paper design {len(dc)} cells, {int(dc.n_trials.sum())} trials, "
      f"{dc.model.nunique()} models x {dc.agent.nunique()} harnesses")
print(f"design cells vs leaderboard: max |diff| {(dc.acc - dc.lb_acc).abs().max():.4f} points")
pd.set_option("display.width", 200)
print("Table 2 rows off by > 0.1 points from the copy:")
print(cells[cells.paper_diff.abs() > 0.1][["agent", "model", "acc", "paper_acc", "paper_diff", "pm95", "paper_pm95"]]
      .round(2).to_string(index=False))
print(dc.pivot(index="model", columns="agent", values="acc").round(1).to_string())
