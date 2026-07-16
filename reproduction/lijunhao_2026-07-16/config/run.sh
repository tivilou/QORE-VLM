#!/bin/bash

MODEL_PATH=models/llama3-8b \
MAX_CAPACITY=1024 \
TRIGGER_EVERY=128 \
MAX_SAMPLES=0 \
MAX_INPUT_LENGTH=7900 \
OUTPUT_DIR=results/kv_cache/longbench_full \
bash scripts/kv_cache/run_longbench.sh llama3
