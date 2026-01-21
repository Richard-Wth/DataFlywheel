#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

python3 "${ROOT_DIR}/src/gen_cot.py" \
  --input "${ROOT_DIR}/data/training/aime.json" \
  --output "/home/test/My_codes/West/ID/DataFlywheel/data/training/aime_thinking.json" \
  --prompt "${ROOT_DIR}/src/prompts/gen_cot.jinja" \
  --model "gpt-5.2" \
  --system "You are a helpful assistant." \
  --limit 2
  
