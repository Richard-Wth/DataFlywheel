#!/usr/bin/env bash
set -euo pipefail

# This script is meant to be runnable from anywhere (no hard-coded repo paths).
# You can override any path via environment variables below.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATAFLYWHEEL_ROOT="${DATAFLYWHEEL_ROOT:-${REPO_ROOT}/DataFlywheel}"
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-${REPO_ROOT}/LlamaFactory}"

PIPELINE_PYTHON="${PIPELINE_PYTHON:-python}"
INFERENCE_PYTHON="${INFERENCE_PYTHON:-/home/test/anaconda3/envs/vllm/bin/python}"

BASE_MODEL="${BASE_MODEL:-/home/test/My_codes/West/ID/models/Qwen3_8B}"
TRAIN_DATA="${TRAIN_DATA:-/home/test/My_codes/West/ID/DataFlywheel/data/training/openthoughts_math_1h.json}"
BENCHMARK_PATH="${BENCHMARK_PATH:-${DATAFLYWHEEL_ROOT}/data/benchmark/benchmark.json}"
LLAMAFACTORY_CONFIG="${LLAMAFACTORY_CONFIG:-${LLAMAFACTORY_ROOT}/examples/train_full/qwen3_full_sft.yaml}"

SAVE_DIR="${SAVE_DIR:-${DATAFLYWHEEL_ROOT}/saves}"
DATA_DIR="${DATA_DIR:-${DATAFLYWHEEL_ROOT}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATAFLYWHEEL_ROOT}/output}"
# Run name controls subfolders under saves/ and output/ (logs, inference, judge, etc.)
RUN_NAME="${RUN_NAME:-qwen3-8b}"

ITERATIONS="${ITERATIONS:-5}"
MODE="${MODE:-sample}"
NUM_PER_CASE="${NUM_PER_CASE:-3}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
INFER_LIMIT="${INFER_LIMIT:-20}"
JUDGE_K="${JUDGE_K:-4}"

"${PIPELINE_PYTHON}" "${DATAFLYWHEEL_ROOT}/src/pipeline.py" \
  --base-model "${BASE_MODEL}" \
  --train-data "${TRAIN_DATA}" \
  --benchmark-path "${BENCHMARK_PATH}" \
  --llamafactory-config "${LLAMAFACTORY_CONFIG}" \
  --llamafactory-root "${LLAMAFACTORY_ROOT}" \
  --iterations "${ITERATIONS}" \
  --mode "${MODE}" \
  --num-per-case "${NUM_PER_CASE}" \
  --save-dir "${SAVE_DIR}" \
  --run-name "${RUN_NAME}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --inference-python "${INFERENCE_PYTHON}" \
  --inference-backend "${INFER_BACKEND}" \
  --num-samples "${NUM_SAMPLES}" \
  --infer-limit "${INFER_LIMIT}" \
  --judge-k "${JUDGE_K}" \
  --judge-no-resume \
  --bad-attr-model "gpt-5.2" \
  --gen-model "gpt-5.2"