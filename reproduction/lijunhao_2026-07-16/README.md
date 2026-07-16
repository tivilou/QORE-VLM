# Reproduction — lijunhao, 2026-07-16

## Headline result

| Application | Model | Benchmark | QORE solver | Budget K | Accuracy/F1 | Redundancy ratio | Paper number | Δ |
|-------------|-------|-----------|-------------|----------|-------------|-----------------|--------------|---|
| KV-Cache eviction | Llama 3 8B | LongBench (6 QA tasks, 1,150 samples) | Simulated annealing (`dwave-neal`) | 1,024 tokens | Macro F1 0.1949 | N/A for KV-Cache | Not recorded | N/A |

QORE retained 15.29% of the theoretical full-cache tokens (1,024 of an average
6,698), saving approximately 709.2 MB of KV cache per sample. Its macro F1 was
0.1949, compared with 0.3671 for the uncompressed full-cache baseline.

## Baseline comparison

| Policy | Macro F1 | Micro F1 | Average cache (MB) | Average latency (ms) |
|--------|---------:|---------:|-------------------:|---------------------:|
| Full cache | 0.3671 | 0.3635 | 838.34 | 2,893.61 |
| QORE | 0.1949 | 0.1928 | 129.09 | 18,023.05 |
| Window | 0.1480 | 0.1467 | 131.75 | 3,378.24 |
| SnapKV-style | 0.1412 | 0.1400 | 133.16 | 4,712.54 |
| Random | 0.1379 | 0.1361 | 133.53 | 3,694.44 |
| PyramidKV-style | 0.1329 | 0.1308 | 136.06 | 6,366.02 |
| H2O-style | 0.1127 | 0.1122 | 136.27 | 5,400.22 |

## Summary

- **Who**: lijunhao (`3321796364@qq.com`)
- **What you ran**: A single-seed evaluation of QORE and six comparison policies
  on all 1,150 LongBench QA samples, using a local Llama 3 8B model and a
  1,024-token cache budget.
- **Verdict**: Partially reproduced. The run completed and includes raw
  per-sample outputs, but the original paper target number was not recorded, so
  an exact paper-to-run delta cannot be reported.

## Artifacts

- `results/summary.csv`: comparison table for all policies.
- `results/<policy>.json`: aggregate metrics and configuration.
- `results/<policy>.samples.json`: raw per-sample evaluation output.
- `logs/`: captured evaluation logs.
- `config/`: normalized configuration and the command used to launch the run.
