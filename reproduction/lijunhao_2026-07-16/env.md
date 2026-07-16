# Environment

## Hardware

- GPU(s): Exact model not recorded; maximum measured process GPU memory was
  35,588.1 MB.
- QPU / quantum backend: No physical QPU; TensorCircuit simulation with
  simulated annealing.
- CPU / RAM: Not recorded.

## Software

- OS: Not recorded.
- Python: Exact version not recorded.
- PyTorch: Exact installed version not recorded; repository constraint
  `>=2.0,<2.6`.
- transformers: Exact installed version not recorded; repository constraint
  `>=4.40,<4.54`.
- datasets: Exact installed version not recorded; repository constraint
  `>=2.14,<3.0`.
- QUBO solver: `dwave-neal==0.6.0`, as locked by `requirements.txt`.
- Quantum library: `tensorcircuit==0.12.0`, as locked by `requirements.txt`.

## Commands

```bash
MODEL_PATH=models/llama3-8b \
MAX_CAPACITY=1024 \
TRIGGER_EVERY=128 \
MAX_SAMPLES=0 \
MAX_INPUT_LENGTH=7900 \
OUTPUT_DIR=results/kv_cache/longbench_full \
bash scripts/kv_cache/run_longbench.sh llama3
```

See `config/run.sh` for the saved launch command.
