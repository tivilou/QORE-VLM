# Reproduction Results

This folder collects **independent reproduction results** for QORE from collaborators
and external reproducers. The goal is to verify that the numbers reported in the paper
hold across different hardware, software stacks, and runs.

> If you are reproducing our results, please upload your run here via a Pull Request
> (or push directly if you have write access). One subfolder per reproducer.

## How to contribute a reproduction

1. Create a subfolder named after yourself / your group and the date:
   `reproduction/<name-or-org>_<YYYY-MM-DD>/`
   e.g. `reproduction/zhang-lab_2026-06-15/`
2. Put your results inside that subfolder following the layout below.
3. Open a Pull Request titled `repro: <name> — <application/benchmark>`.

## Expected layout per reproduction

```
reproduction/<name>_<date>/
├── README.md          # who you are, what you ran, headline numbers
├── env.md             # hardware (GPU/QPU), OS, key package versions
├── config/            # exact configs / scripts / commands used
├── logs/              # raw eval / solver logs
├── results/           # metrics tables (csv / json) + any plots
└── notes.md           # (optional) deviations, issues, observations
```

## What to report

Please include at minimum:

- **Application**: KV-Cache eviction or RAG context selection (or both)
- **Setup**: model (LLaMA-3-8B / Mistral-7B / etc.), benchmark, budget K,
  QORE solver backend (simulated annealing / QAOA / quantum kernel)
- **Metrics**:
  - KV-Cache: accuracy/F1 on benchmark, cache size K, reduction %, perplexity, latency
  - RAG: answer accuracy/F1, recall@K of gold passages, redundancy ratio
- **Comparison**: the corresponding paper / baseline number you are checking
  against, and the delta
- **Reproducibility**: random seeds, number of runs, and variance if available

A short results table is enough for the headline; raw logs go in `logs/`.

## Notes

- Keep large artifacts (checkpoints, datasets, index files) **out** of git — link to
  external storage instead. The repo `.gitignore` already excludes `*.pt`,
  `*.safetensors`, `datasets/`, etc.
- Metrics files (`*.csv`, `*.json`) and small plots are fine to commit.
- One subfolder per reproduction run keeps history clean and attributable.
