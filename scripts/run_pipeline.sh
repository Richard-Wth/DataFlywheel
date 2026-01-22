#!/usr/bin/env bash
set -euo pipefail

# This script is meant to be runnable from anywhere (no hard-coded repo paths).
# You can override any path via environment variables below.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATAFLYWHEEL_ROOT="${DATAFLYWHEEL_ROOT:-${REPO_ROOT}/DataFlywheel}"
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-${REPO_ROOT}/LlamaFactory}"

PIPELINE_PYTHON="${PIPELINE_PYTHON:-python}"
INFERENCE_PYTHON="${INFERENCE_PYTHON:-python}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
LLAMAFACTORY_CONFIG="${LLAMAFACTORY_CONFIG:-${LLAMAFACTORY_ROOT}/examples/train_full/qwen3_full_sft.yaml}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"

SAVE_DIR="${SAVE_DIR:-${DATAFLYWHEEL_ROOT}/saves}"
DATA_DIR="${DATA_DIR:-${DATAFLYWHEEL_ROOT}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATAFLYWHEEL_ROOT}/output}"
TRAINING_JSON="${TRAINING_JSON:-${DATA_DIR}/training/numina_training.json}"
SEED_TRAIN_DATA="${SEED_TRAIN_DATA:-${DATA_DIR}/training/numina_train_10000.json}"
BENCHMARK_PATH="${BENCHMARK_PATH:-${DATA_DIR}/benchmark/numina_test.json}"
# Run name controls subfolders under saves/ and output/ (logs, inference, judge, etc.)
RUN_NAME="${RUN_NAME:-qwen3-0.6b-numina}"

TRAIN_DATASET_ID="${TRAIN_DATASET_ID:-AI-MO/NuminaMath-CoT}"
TRAIN_DATASET_CONFIG="${TRAIN_DATASET_CONFIG:-}"
TRAIN_DATASET_SPLIT="${TRAIN_DATASET_SPLIT:-train}"
TRAIN_DATASET_LIMIT="${TRAIN_DATASET_LIMIT:-10000}"
BENCHMARK_DATASET_ID="${BENCHMARK_DATASET_ID:-AI-MO/NuminaMath-CoT}"
BENCHMARK_DATASET_CONFIG="${BENCHMARK_DATASET_CONFIG:-}"
BENCHMARK_SPLIT="${BENCHMARK_SPLIT:-test}"

ITERATIONS="${ITERATIONS:-5}"
MODE="${MODE:-full}"
NUM_PER_CASE="${NUM_PER_CASE:-5}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
INFER_LIMIT="${INFER_LIMIT:-50}"
JUDGE_K="${JUDGE_K:-1}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-50}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TRAIN_FROM_PREV="${TRAIN_FROM_PREV:-1}"
TRAIN_BASE="${TRAIN_BASE:-1}"
REBUILD_TRAIN_DATA="${REBUILD_TRAIN_DATA:-0}"
RESET_TRAINING_JSON="${RESET_TRAINING_JSON:-0}"

EXTRA_TRAIN_DATA_ARGS=()
if [[ -n "${SEED_TRAIN_DATA}" ]]; then
  EXTRA_TRAIN_DATA_ARGS+=(--train-data "${SEED_TRAIN_DATA}")
fi

EXTRA_PIPELINE_ARGS=()
if [[ "${TRAIN_FROM_PREV}" == "1" ]]; then
  EXTRA_PIPELINE_ARGS+=(--train-from-prev)
fi
if [[ "${TRAIN_BASE}" == "1" ]]; then
  EXTRA_PIPELINE_ARGS+=(--train-base)
fi
if [[ "${REBUILD_TRAIN_DATA}" == "1" ]]; then
  EXTRA_PIPELINE_ARGS+=(--rebuild-train-data)
fi
if [[ "${RESET_TRAINING_JSON}" == "1" ]]; then
  EXTRA_PIPELINE_ARGS+=(--reset-training-json)
fi
if [[ -n "${TRAIN_DATASET_CONFIG}" ]]; then
  EXTRA_PIPELINE_ARGS+=(--train-dataset-config "${TRAIN_DATASET_CONFIG}")
fi
if [[ -n "${BENCHMARK_DATASET_CONFIG}" ]]; then
  EXTRA_PIPELINE_ARGS+=(--benchmark-dataset-config "${BENCHMARK_DATASET_CONFIG}")
fi

"${PIPELINE_PYTHON}" "${DATAFLYWHEEL_ROOT}/src/pipeline.py" \
  --base-model "${BASE_MODEL}" \
  "${EXTRA_TRAIN_DATA_ARGS[@]}" \
  --training-json "${TRAINING_JSON}" \
  --benchmark-path "${BENCHMARK_PATH}" \
  --benchmark-dataset-id "${BENCHMARK_DATASET_ID}" \
  --benchmark-split "${BENCHMARK_SPLIT}" \
  --train-dataset-id "${TRAIN_DATASET_ID}" \
  --train-dataset-split "${TRAIN_DATASET_SPLIT}" \
  --train-dataset-limit "${TRAIN_DATASET_LIMIT}" \
  --llamafactory-config "${LLAMAFACTORY_CONFIG}" \
  --llamafactory-root "${LLAMAFACTORY_ROOT}" \
  --llamafactory-cli "${LLAMAFACTORY_CLI}" \
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
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --max_tokens "${MAX_TOKENS}" \
  --max_model_len "${MAX_MODEL_LEN}" \
  --infer-limit "${INFER_LIMIT}" \
  --judge-k "${JUDGE_K}" \
  "${EXTRA_PIPELINE_ARGS[@]}" \
  --judge-no-resume \
  --bad-attr-model "gpt-5.2" \
  --gen-model "gpt-5.2"

# export LLAMAFACTORY_CLI=/home/test/anaconda3/envs/lf_train/bin/llamafactory-cli
# bash /home/test/My_codes/West/ID/DataFlywheel/scripts/run_pipeline.sh
