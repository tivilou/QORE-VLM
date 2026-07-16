# Notes

- The evaluation used seed 42 and one run per policy; variance and confidence
  intervals were not measured.
- QORE used attention quality, cosine redundancy, and simulated annealing.
- The run covered NarrativeQA (200), Qasper (200), MultiFieldQA-en (150),
  HotpotQA (200), 2WikiMQA (200), and MuSiQue (200).
- H2O, SnapKV, and PyramidKV are simplified style-compatible implementations,
  not line-by-line reproductions of their original papers. This is recorded in
  each aggregate JSON file.
- The captured `run.log` ends while H2O is loading. `remaining.log` contains the
  continuation used to finish the remaining policies. The result JSON files and
  summary contain completed outputs for all seven policies.
- Hardware identity and exact installed package versions were not preserved in
  the logs, so `env.md` reports only directly verifiable information.
