# Total Evaluation Error: replication materials

Code, processed data, and paper source for *Total Evaluation Error* (NeurIPS 2026). The paper
decomposes the uncertainty in LLM evaluation pipelines (items, prompt variants, judge or SUT models,
temperatures, replications, and their interactions) and uses D-study projections to choose designs
that shrink it.

The R package that implements the method on your own data is at
<https://github.com/SolomonMg/totalevalerror>. This repository reproduces the paper.

## Contents

| Folder | What it holds |
|---|---|
| `experiments/` | Python scripts that query models through OpenRouter and write raw JSONL |
| `analysis/` | R scripts: cleaning, variance decomposition (G-study), D-study, simulations, figures |
| `data/processed/` | Cleaned CSVs and every intermediate result the paper cites |
| `data/*.csv` | Item sets (AILuminate demo prompts, MMLU items, ideology items) |
| `rebuttal_neurips2026/` | Prompt-variant generation, construct-equivalence screen, and the re-collections behind the SI breadth analysis (folder name kept because scripts use these paths) |
| `figures/` | Every figure, as PDF and PNG |
| `paper/` | LaTeX source of the arXiv and NeurIPS camera-ready versions |
| `tests/` | `testthat` checks that every number cited in the paper matches the CSVs |

## Reproducing the paper

Requirements: R 4.5 with `tidyverse`, `lme4`, `patchwork`, `testthat`, `rprojroot`, `jsonlite`,
`data.table`; Python 3.12 (see `experiments/requirements.txt`); TeX Live; Ghostscript.

```bash
bash run_all.sh               # clean -> decompose -> simulate -> figures -> verify
Rscript tests/run_tests.R     # check every cited number against data/processed
make -C paper arxiv neurips   # build both PDFs
```

The analysis stages run from the processed CSVs and need no API access. Monte Carlo stages cache
their results in `data/processed/`; delete a cache CSV to recompute it. Re-collecting model
responses requires an OpenRouter key (copy `.env.example` to `.env`).

## Raw data

Raw API responses (JSONL, several GB) are archived on Harvard Dataverse; the DOI will be added here.
Everything the paper reports can be reproduced from `data/processed/` without them.

## Notes

- MMLU answers are parsed with a strict parser (`analysis/lib_parse_mmlu.R`) that never infers a
  letter from prose; one prompt variant without a letter-only instruction is excluded
  (SI, "Answer parsing and variant exclusion").
- A Chatbot Arena user prompt in the public `lmarena-ai/arena-human-preference-100k` dataset
  contains what appears to be an AWS credential. It is redacted in
  `data/processed/arena_classifier/test_predictions*.csv`.
- Model responses to the AILuminate hazard prompts (`data/sut_responses_safety.csv`, which the
  safety judging scripts read) are in the Dataverse deposit only, because some are unsafe by design.

## License

Code: MIT (`LICENSE`). Data and figures: CC BY 4.0 (`LICENSE-DATA.md`), except where third-party
sources set their own terms.
