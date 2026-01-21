import argparse
import json
import os
import re
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

DATASET_ID = "HuggingFaceH4/MATH-500"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

FINAL_ANSWER_INSTRUCTION = (
    "Please reason step by step, and put your final answer in the last line within \\boxed{} "
    "(do not write anything after the boxed answer)."
)


def extract_last_boxed(text: str) -> str:
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


_HASH_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)


def extract_final_answer(text: str) -> str:
    assert isinstance(text, str), f"text must be a string, but got {type(text)}"
    boxed = extract_last_boxed(text)
    if boxed:
        return boxed.strip()
    m = _HASH_RE.search(text)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def normalize_final_answer(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().replace("$", "")
    s = re.sub(r"[\s\.\,;:，。；：]+$", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def _get_question(example: Dict[str, Any]) -> str:
    q = example.get("problem") or example.get("question") or ""
    return q.strip() if isinstance(q, str) else str(q)


def _get_gt_final(example: Dict[str, Any]) -> str:
    ans = example.get("answer", "")
    if isinstance(ans, dict):
        return str(ans.get("value", "")).strip()
    return str(ans).strip()


def build_prompt(tokenizer: Any, question: str) -> str:
    user_content = f"{question.strip()}\n\n{FINAL_ANSWER_INSTRUCTION}".strip()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_content


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_out_dir = os.path.join(repo_root, "output", "inference")
    default_hf_home = os.path.join(repo_root, "output", "hf_home")

    parser = argparse.ArgumentParser(description="vLLM inference on MATH-500 (pass@1)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF model id or local path")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=default_out_dir)
    parser.add_argument("--hf_home", type=str, default=default_hf_home)
    parser.add_argument("--output_name", type=str, default="math500_qwen3_0_6b_vllm_tp8.jsonl")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.output_name)
    summary_path = out_path.replace(".jsonl", ".summary.json")

    if args.hf_home:
        os.environ.setdefault("HF_HOME", args.hf_home)
        os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    ds = load_dataset(DATASET_ID, split=args.split)
    if args.limit is not None:
        ds = ds.select(range(int(args.limit)))

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=1,
    )

    total = len(ds)
    correct = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for start in tqdm(range(0, total, args.batch_size), desc="infer"):
            end = min(total, start + args.batch_size)
            batch_indices = list(range(start, end))
            batch_examples = [ds[i] for i in batch_indices]
            batch_questions = [_get_question(ex) for ex in batch_examples]
            batch_gt = [_get_gt_final(ex) for ex in batch_examples]
            batch_prompts = [build_prompt(tokenizer, q) for q in batch_questions]

            batch_outputs = llm.generate(batch_prompts, sampling_params)

            for idx, question, gt, out in zip(batch_indices, batch_questions, batch_gt, batch_outputs):
                pred_text = out.outputs[0].text if out.outputs else ""
                pred_final = extract_final_answer(pred_text)
                gt_final = extract_final_answer(gt)
                ok = bool(normalize_final_answer(pred_final)) and (
                    normalize_final_answer(pred_final) == normalize_final_answer(gt_final)
                )
                correct += int(ok)
                row = {
                    "dataset_id": DATASET_ID,
                    "sample_index": idx,
                    "question": question,
                    "gt_final": gt_final,
                    "prediction": pred_text,
                    "pred_final": pred_final,
                    "correct": ok,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pass1 = correct / total if total else 0.0
    summary = {
        "dataset_id": DATASET_ID,
        "split": args.split,
        "model": args.model,
        "tp": args.tp,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "total": total,
        "correct": correct,
        "pass@1": pass1,
        "output": out_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
