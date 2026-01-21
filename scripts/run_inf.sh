#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-50}"

python3 "${ROOT_DIR}/src/inference.py" \
  --model "/home/test/My_codes/West/ID/LlamaFactory/saves/qwen3-8b/full_sft/ds_math_thinking_xmltags/checkpoint-200" \
  --input "${ROOT_DIR}/data/benchmark/benchmark.json" \
  --output "${ROOT_DIR}/output/inference/qwen3_8b/qwen3_8b_openthoughts_milestone_5k.json" \
  --tp 8 \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_tokens 32768 \
  --limit 1000 \
  --use_xml_tags \
  --decode_skip_special_tokens