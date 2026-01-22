#!/usr/bin/env python3
import argparse
import json
import re
import sys
from typing import Any, Dict, List


FINAL_ANSWER_PREFIX = "**Final Answer:**"


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []
        if text.startswith("["):
            data = json.loads(text)
            return data if isinstance(data, list) else []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _validate_item(
    item: Dict[str, Any],
    idx: int,
    require_think: bool,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(item, dict):
        return [f"[{idx}] item is not a dict"]

    instruction = item.get("instruction", "")
    output = item.get("output", "")

    if not isinstance(instruction, str) or not instruction.strip():
        errors.append(f"[{idx}] missing or empty instruction")
    else:
        # Instruction should be the NEW question text (not a training prefix).
        if instruction.strip().lower().startswith("return your final response within"):
            errors.append(f"[{idx}] instruction still contains legacy training prefix")
        if instruction.strip().lower().startswith("new question:"):
            errors.append(f"[{idx}] instruction should not start with 'New question:' header")
        if len(instruction.strip()) < 20:
            errors.append(f"[{idx}] instruction too short to be a complete question")
    if not isinstance(output, str) or not output.strip():
        errors.append(f"[{idx}] missing or empty output")
        return errors

    # Output should start with a <think> block (since instruction is the question).
    if not _first_nonempty_line(output).lower().startswith("<think>"):
        errors.append(f"[{idx}] output should start with <think>...</think>")

    if require_think:
        think_count = output.lower().count("<think>")
        end_count = output.lower().count("</think>")
        if think_count != 1 or end_count != 1:
            errors.append(f"[{idx}] expected exactly one <think>...</think> block")
        else:
            low = output.lower()
            start = low.find("<think>")
            end = low.find("</think>")
            if end < start:
                errors.append(f"[{idx}] </think> appears before <think>")
            else:
                think_body = output[start + len("<think>") : end].strip()
                if not think_body:
                    errors.append(f"[{idx}] empty <think> content")
    else:
        if "<think>" in output or "</think>" in output:
            errors.append(f"[{idx}] unexpected <think> tag in stripped mode")

    if "</think>" in output.lower():
        # split in a case-insensitive way
        m = re.search(r"(?is)</think>", output)
        tail = output[m.end() :] if m else ""
    else:
        tail = output
    tail = tail.strip()
    if not tail:
        errors.append(f"[{idx}] missing solution content after <think>")
    else:
        if "\\boxed{" in tail:
        # Must end with "**Final Answer:** ..."
        lines = [ln.rstrip() for ln in tail.splitlines() if ln.strip()]
        if not lines:
            errors.append(f"[{idx}] missing content after </think>")
        else:
            last = lines[-1].strip()
            if not last.startswith(FINAL_ANSWER_PREFIX):
                errors.append(f"[{idx}] missing final '{FINAL_ANSWER_PREFIX} ...' line")
            else:
                val = last[len(FINAL_ANSWER_PREFIX) :].strip()
                if not val:
                    errors.append(f"[{idx}] empty final answer after '{FINAL_ANSWER_PREFIX}'")
                else:
                    # Require boxed final answer.
                    if "\\boxed{" not in val:
                        errors.append(f"[{idx}] final answer must be wrapped in \\\\boxed{{...}}")
                    # Ensure the last non-empty line ends with the boxed token (no trailing text).
                    if not re.search(r"\\boxed\\{[\\s\\S]*\\}\\s*$", val):
                        errors.append(f"[{idx}] boxed final answer is not properly closed at end of line")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate data_gen output format.")
    parser.add_argument("--input", required=True, help="Path to generated JSON/JSONL")
    parser.add_argument(
        "--mode",
        default="raw",
        choices=["raw", "stripped"],
        help="raw requires <think> block; stripped forbids it",
    )
    parser.add_argument("--max-report", type=int, default=20, help="Max error lines to print")
    args = parser.parse_args()

    items = _read_json_or_jsonl(args.input)
    if not items:
        print("No items loaded.")
        sys.exit(1)

    require_think = args.mode == "raw"
    all_errors: List[str] = []
    for idx, item in enumerate(items):
        all_errors.extend(_validate_item(item, idx, require_think=require_think))

    total = len(items)
    bad = len({e.split("]")[0] for e in all_errors}) if all_errors else 0
    print(f"Checked {total} items. Bad items: {bad}. Error lines: {len(all_errors)}.")
    for err in all_errors[: max(0, int(args.max_report))]:
        print(err)

    if all_errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
