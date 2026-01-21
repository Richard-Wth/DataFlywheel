import argparse
import concurrent.futures
import json
import os
import random
import re
import time
import sys
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR and _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from utils import extract_last_boxed, load_existing_jsonl, read_json, write_json


_BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
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


def _normalize_final_answer(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"[\s\.\,;:，。；：]+$", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def judge_one(item: Dict[str, Any]) -> Dict[str, Any]:
    prediction = item.get("prediction", "")
    answer = item.get("answer", "")
    pred_final = extract_final_answer(prediction)
    gt_final = extract_final_answer(answer)
    final_correct = bool(_normalize_final_answer(pred_final)) and (
        _normalize_final_answer(pred_final) == _normalize_final_answer(gt_final)
    )
    return {
        "correct": final_correct,
        "final_correct": final_correct,
        "pred_final": pred_final,
        "gt_final": gt_final,
    }


def _get_predictions(item: Dict[str, Any]) -> List[str]:
    preds = item.get("predictions")
    if isinstance(preds, list) and preds:
        # New format: list[dict] with {"output": "...", "final_answer": "..."}
        if isinstance(preds[0], dict):
            out: List[str] = []
            for p in preds:
                if not isinstance(p, dict):
                    out.append("")
                    continue
                text = p.get("output", "")
                out.append(text if isinstance(text, str) else "")
            return out
        # Old format: list[str]
        return [p if isinstance(p, str) else "" for p in preds]
    pred = item.get("prediction", "")
    return [pred] if isinstance(pred, str) and pred else [""]


def judge_many(item: Dict[str, Any], k_override: Optional[int] = None) -> Dict[str, Any]:
    """Judge all candidates for one item.

    Returns:
      - candidates: list of per-candidate judge dicts (same schema as judge_one)
      - pass@1: whether the first candidate is correct
      - pass@k: whether any of the first k candidates is correct
      - k: used k
      - n_candidates: total candidates
    """
    preds = _get_predictions(item)
    answer = item.get("answer", "")
    # Judge each candidate using the same logic as judge_one.
    candidates: List[Dict[str, Any]] = []
    for p in preds:
        candidates.append(judge_one({"prediction": p, "answer": answer}))

    if k_override is None:
        k = int(item.get("num_samples") or len(preds) or 1)
    else:
        k = int(k_override)
    k = max(1, min(k, len(candidates)))
    pass1 = bool(candidates[0].get("final_correct")) if candidates else False
    passk = any(c.get("final_correct") for c in candidates[:k]) if candidates else False
    return {
        "candidates": candidates,
        "pass@1": pass1,
        f"pass@{k}": passk,
        "k": k,
        "n_candidates": len(candidates),
    }


def build_judge_key(item: Dict[str, Any]) -> str:
    # Support inference outputs without dataset_id.
    dataset_key = item.get("dataset_id", "") or item.get("dataset", "") or ""
    return f"{dataset_key}::{item.get('sample_index','')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge predictions using OpenAI compatible interface")
    parser.add_argument("--input", type=str, default=None, help="Input prediction results JSON (list)")
    parser.add_argument("--output", type=str, default=None, help="Output judge results JSONL (recommended, supports resume)")
    parser.add_argument("--summary", type=str, default=None, help="Output summary statistics JSON")
    parser.add_argument("--statistic", type=str, default=None, help="Output pass@1 dataset statistics JSON")
    parser.add_argument("--bad-cases", type=str, default=None, help="Output bad cases JSON")
    parser.add_argument("--k", type=int, default=8, help="Compute pass@k using top-k candidates (default: 8).")
    parser.add_argument("--max-workers", type=int, default=16, help="Number of concurrent workers")
    parser.add_argument("--limit", type=int, default=None, help="Only judge the first N questions (for debugging)")
    parser.add_argument("--no-resume", action="store_true", help="Do not use resume, overwrite output")
    args = parser.parse_args()

    for p in (args.output, args.summary, args.statistic, args.bad_cases):
        if not p:
            continue
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

    if args.no_resume and args.output and args.output.endswith(".jsonl"):
        if os.path.exists(args.output):
            os.remove(args.output)

    print(f"读取输入: {args.input}")
    data = read_json(args.input)
    if not isinstance(data, list):
        raise SystemExit("输入文件必须是 JSON 列表")
    if args.limit:
        data = data[: args.limit]
        print(f"限制评测前 {args.limit} 条")

    existing: Dict[str, Dict[str, Any]] = {}
    if (not args.no_resume) and args.output and args.output.endswith(".jsonl"):
        existing = load_existing_jsonl(args.output, key_field="_judge_key")
        if existing:
            print(f"断点续跑：已存在 {len(existing)} 条结果，将跳过这些样本")

    rows_out: List[Dict[str, Any]] = []
    if existing:
        rows_out.extend(list(existing.values()))

    to_judge: List[Dict[str, Any]] = []
    for item in data:
        key = build_judge_key(item)
        if key and key in existing:
            continue
        item = dict(item)
        item["_judge_key"] = key
        to_judge.append(item)

    print(f"待评测: {len(to_judge)} / 总计: {len(data)}")

    def _work(it: Dict[str, Any]) -> Dict[str, Any]:
        preds = _get_predictions(it)
        inferred_k = max(1, len(preds)) if preds else 1
        k_eff = int(args.k) if args.k is not None else inferred_k
        if k_eff != inferred_k:
            print(f"[warn] judge-k={k_eff} 与推理次数({inferred_k})不一致")
        verdict_multi = judge_many(it, k_override=k_eff)
        # For backward-compatibility: keep a single-judge view (first candidate).
        verdict_1 = verdict_multi["candidates"][0] if verdict_multi.get("candidates") else judge_one(it)
        passk = any(v.get("final_correct") for v in verdict_multi.get("candidates", [])[:k_eff])
        out = {
            "_judge_key": it.get("_judge_key", build_judge_key(it)),
            "dataset": it.get("dataset", ""),
            "dataset_id": it.get("dataset_id", ""),
            "sample_index": it.get("sample_index", ""),
            "question": it.get("question", ""),
            "answer": it.get("answer", ""),
            "prediction": it.get("prediction", ""),
            "predictions": it.get("predictions", None),
            "judge": verdict_1,
            "judge_multi": {
                "k": k_eff,
                "pass@1": bool(verdict_multi.get("pass@1")),
                "pass@k": bool(passk),
                f"pass@{k_eff}": bool(passk),
                "candidates": verdict_multi.get("candidates", []),
            },
        }
        return out

    if to_judge:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(_work, it) for it in to_judge]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="评测进度"):
                row = fut.result()
                rows_out.append(row)
                # 增量落盘（jsonl）
                if args.output and args.output.endswith(".jsonl"):
                    with open(args.output, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.output and args.output.endswith(".jsonl") and os.path.exists(args.output):
        rows_out = []
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows_out.append(row)

        # 去重：避免断点续跑或重复落盘导致样本被计数多次
        dedup: Dict[str, Dict[str, Any]] = {}
        for row in rows_out:
            key = row.get("_judge_key")
            if not isinstance(key, str) or not key:
                key = build_judge_key(row)
            if not isinstance(key, str) or not key:
                key = f"{row.get('dataset','')}::{row.get('sample_index','')}"
            dedup[key] = row
        rows_out = list(dedup.values())

    judged = [r for r in rows_out if isinstance(r.get("judge"), dict)]
    n = len(judged)
    pass1 = sum(1 for r in judged if r["judge"].get("final_correct")) / n if n else 0.0
    ks = [int((r.get("judge_multi", {}) or {}).get("k", 1)) for r in judged]
    k = ks[0] if ks and len(set(ks)) == 1 else (max(ks) if ks else 1)
    if ks and len(set(ks)) != 1:
        print(f"[warn] 推理次数不一致: {sorted(set(ks))}，统计使用 k={k}")
    passk = sum(1 for r in judged if (r.get("judge_multi", {}) or {}).get("pass@k")) / n if n else 0.0

    summary = {
        "count": n,
        "pass@1": pass1,
        f"pass@{k}": passk,
        "input": args.input,
        "output": args.output,
    }
    write_json(args.summary, summary)
    print(f"完成。汇总已写入: {args.summary}")

    by_dataset: Dict[str, Dict[str, Any]] = {}

    def _acc_update(bucket: Dict[str, Dict[str, Any]], key: str, row: Dict[str, Any]) -> None:
        if key is None:
            key = ""
        key = str(key)
        if key not in bucket:
            bucket[key] = {
                "count": 0,
                "pass@1": 0.0,
                f"pass@{k}": 0.0,
            }
        b = bucket[key]
        b["count"] += 1
        b["pass@1"] += 1.0 if row["judge"].get("final_correct") else 0.0
        b[f"pass@{k}"] += 1.0 if (row.get("judge_multi", {}) or {}).get("pass@k") else 0.0

    for r in judged:
        _acc_update(by_dataset, r.get("dataset", ""), r)

    def _finalize(bucket: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for bucket_key, v in bucket.items():
            c = int(v.get("count", 0)) or 0
            if c <= 0:
                out[bucket_key] = {
                    "count": 0,
                    "pass@1": 0.0,
                    f"pass@{k}": 0.0,
                }
                continue
            out[bucket_key] = {
                "count": c,
                "pass@1": float(v["pass@1"]) / c,
                f"pass@{k}": float(v[f"pass@{k}"]) / c,
            }
        return out

    statistic = {
        "overall": {
            "count": n,
            "pass@1": pass1,
            f"pass@{k}": passk,
        },
        "by_dataset": _finalize(by_dataset),
    }
    write_json(args.statistic, statistic)
    print(f"统计已写入: {args.statistic}")

    if args.output and not args.output.endswith(".jsonl"):
        write_json(args.output, rows_out)

    if args.bad_cases:
        bad_rows: List[Dict[str, Any]] = []
        for r in rows_out:
            judge_multi = r.get("judge_multi", {}) or {}
            if not judge_multi.get(f"pass@{k}"):
                bad_rows.append(
                    {
                        "question": r.get("question", ""),
                        "answer": r.get("answer", ""),
                        "dataset": r.get("dataset", ""),
                        "sample_index": r.get("sample_index", ""),
                        "predictions": r.get("predictions", []),
                    }
                )
        write_json(args.bad_cases, bad_rows)


if __name__ == "__main__":
    main()

