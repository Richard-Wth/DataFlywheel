#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple


TRAINING_PREFIX = "Return your final response within \\\\boxed{}."
NEW_QUESTION_PREFIX = "New question:"


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


def _extract_last_boxed(text: str) -> str:
    if not isinstance(text, str) or "\\boxed{" not in text:
        return ""
    starts: List[int] = []
    needle = "\\boxed{"
    i = 0
    while True:
        j = text.find(needle, i)
        if j < 0:
            break
        starts.append(j)
        i = j + len(needle)
    for start in reversed(starts):
        i = start + len(needle)
        depth = 1
        out_chars: List[str] = []
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
                out_chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(out_chars).strip()
                out_chars.append(ch)
            else:
                out_chars.append(ch)
            i += 1
    return ""


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _extract_original_question(instruction: str) -> str:
    if not isinstance(instruction, str):
        return ""
    inst = instruction.strip()
    if inst.startswith(TRAINING_PREFIX):
        return inst[len(TRAINING_PREFIX) :].strip()
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
    if not isinstance(output, str) or not output.strip():
        errors.append(f"[{idx}] missing or empty output")
        return errors

    first_line = _first_nonempty_line(output)
    if not first_line.startswith(NEW_QUESTION_PREFIX):
        errors.append(f"[{idx}] missing 'New question:' line at top")
        new_question = ""
    else:
        new_question = first_line[len(NEW_QUESTION_PREFIX) :].strip()
        if not new_question:
            errors.append(f"[{idx}] empty new question text")

    if require_think:
        think_count = output.count("<think>")
        end_count = output.count("</think>")
        if think_count != 1 or end_count != 1:
            errors.append(f"[{idx}] expected exactly one <think>...</think> block")
        else:
            start = output.find("<think>")
            end = output.find("</think>")
            if end < start:
                errors.append(f"[{idx}] </think> appears before <think>")
            else:
                think_body = output[start + len("<think>") : end].strip()
                if not think_body:
                    errors.append(f"[{idx}] empty <think> content")
    else:
        if "<think>" in output or "</think>" in output:
            errors.append(f"[{idx}] unexpected <think> tag in stripped mode")

    if "</think>" in output:
        tail = output.split("</think>", 1)[1]
    else:
        tail = output
    tail = tail.strip()
    if not tail:
        errors.append(f"[{idx}] missing solution content after <think>")
    else:
        last_boxed = _extract_last_boxed(tail)
        if not last_boxed:
            errors.append(f"[{idx}] missing final \\\\boxed{{...}} in solution")
        else:
            boxed_token = f"\\\\boxed{{{last_boxed}}}"
            if not tail.rstrip().endswith(boxed_token):
                errors.append(f"[{idx}] final \\\\boxed{{...}} is not the last output")

    original_question = _extract_original_question(instruction)
    if original_question and new_question:
        if _normalize_text(original_question) == _normalize_text(new_question):
            errors.append(f"[{idx}] new question matches original question")

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
