#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

python3 "${ROOT_DIR}/src/judge_inference.py" \
  --input "${ROOT_DIR}/output/inference/qwen3_8b/qwen3_8b_openthoughts_milestone_5k.json" \
  --summary "${ROOT_DIR}/output/judge/qwen3_8b/openthoughts_milestone_5k/qwen3_8b.summary.json" \
  --statistic "${ROOT_DIR}/output/judge/qwen3_8b/openthoughts_milestone_5k/qwen3_8b.stat.json" \
  --bad-cases "${ROOT_DIR}/output/judge/qwen3_8b/openthoughts_milestone_5k/bad_cases.json"

