import argparse
import concurrent.futures
import json
import random
import os
import re
from typing import Any, Dict, List, Tuple, Optional

from jinja2 import Template
from tqdm import tqdm

from utils import extract_last_boxed, init_openai_client, write_json


DEFAULT_SYSTEM_PROMPT = (
    "You are a data generator for math contest model training.\n"
    "Given an original question and its failure attribution (error analysis), you must create a NEW math question "
    "that is clearly different from the original (not a paraphrase or trivial number swap), but targets a similar "
    "topic and difficulty, then solve YOUR new question.\n"
    "\n"
    "Hard rules:\n"
    "- Do NOT output JSON.\n"
    "- Do NOT restate or paraphrase the original question.\n"
    "- The new question must be unambiguous and have a single correct final answer.\n"
    "- The reasoning must be correct and consistent with the final answer.\n"
    "\n"
    "Output format (must follow strictly):\n"
    "1) A line starting with `New question:` followed by your new problem.\n"
    "2) A single `<think>...</think>` block containing the reasoning steps.\n"
    "3) After `</think>`, write the solution (can be short), and end with a line containing `**Final Answer:** ...`.\n"
    "4) The `**Final Answer:** ...` line must be the last thing you write.\n"
)


_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)
_TAG_BLOCK_RE = re.compile(
    r"<\|begin_of_thought\|>[\s\S]*?<\|end_of_thought\|>\s*"
    r"|<\|begin_of_solution\|>[\s\S]*?<\|end_of_solution\|>\s*",
    re.IGNORECASE,
)

NEW_QUESTION_PREFIX = "New question:"
FINAL_ANSWER_PREFIX = "**Final Answer:**"
_BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _load_template(path: str) -> Template:
    with open(path, "r", encoding="utf-8") as f:
        return Template(f.read())


def _extract_response_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = getattr(resp, "output", None)
    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                ctype = getattr(content, "type", None)
                if ctype in ("output_text", "text"):
                    part = getattr(content, "text", None) or getattr(content, "output_text", None)
                    if isinstance(part, str) and part.strip():
                        parts.append(part.strip())
        if parts:
            return "\n".join(parts)
    return ""


def _call_model(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> str:
    if hasattr(client, "responses"):
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        text = _extract_response_text(resp)
        if text:
            return text

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    return content.strip() if isinstance(content, str) else ""

def _strip_unwanted_tags(text: str) -> str:
    """Best-effort cleanup for special training tags (keep <think> for new format)."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Remove the special training tags if they appear (some models emit these).
    text = _TAG_BLOCK_RE.sub("", text)
    return text.strip()

def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize(s: str) -> List[str]:
    # Simple tokenizer for similarity heuristics (ASCII letters/digits + individual symbols).
    return re.findall(r"[a-z]+|\d+|[^\s]", (s or "").lower())

def _too_similar(a: str, b: str) -> bool:
    """Heuristic: reject exact match or near-duplicate questions."""
    na = _normalize_text(a)
    nb = _normalize_text(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ta = _tokenize(na)
    tb = _tokenize(nb)
    if len(ta) < 12 or len(tb) < 12:
        return False
    sa, sb = set(ta), set(tb)
    if not sa or not sb:
        return False
    jacc = len(sa & sb) / max(1, len(sa | sb))
    # Only apply strict jaccard threshold for moderately long questions.
    if min(len(na), len(nb)) >= 60 and jacc >= 0.92:
        return True
    return False

def _debox(text: str) -> str:
    """Remove a single outer \\boxed{...} wrapper if present."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if "\\boxed{" not in t:
        return t
    # If the entire string is one boxed expression, unwrap it.
    m = _BOX_RE.fullmatch(t)
    if m:
        return (m.group(1) or "").strip()
    return t

def _strip_latex_wrappers(s: str) -> str:
    """Strip common LaTeX math wrappers like \\( ... \\), $$...$$, $...$."""
    if not isinstance(s, str):
        return ""
    t = s.strip()
    # Remove surrounding \(...\) repeatedly
    while t.startswith("\\(") and t.endswith("\\)"):
        t = t[2:-2].strip()
    # Remove surrounding $$...$$
    while t.startswith("$$") and t.endswith("$$") and len(t) >= 4:
        t = t[2:-2].strip()
    # Remove surrounding $...$
    while t.startswith("$") and t.endswith("$") and len(t) >= 2:
        t = t[1:-1].strip()
    return t

def _clean_final_answer(raw: str) -> str:
    """Normalize final answer string and return an unboxed value/expression."""
    if not isinstance(raw, str):
        return ""
    t = _strip_latex_wrappers(raw.strip())
    # If there's any boxed anywhere, prefer the last boxed content.
    boxed_any = extract_last_boxed(t)
    if boxed_any:
        t = boxed_any.strip()
    else:
        # Fallback: unwrap if the whole string is boxed.
        t = _debox(t).strip()
    t = _strip_latex_wrappers(t)
    # Avoid leaving a boxed token around.
    if "\\boxed{" in t:
        # If still present, try once more.
        boxed_any2 = extract_last_boxed(t)
        if boxed_any2:
            t = boxed_any2.strip()
        else:
            t = t.replace("\\boxed{", "").replace("}", "").strip()
    return t.strip()

def _ensure_boxed(ans: str) -> str:
    """Wrap ans in \\boxed{...} unless it already contains a boxed expression (use the last one)."""
    if not isinstance(ans, str):
        return ""
    a = ans.strip()
    if not a:
        return ""
    boxed = extract_last_boxed(a)
    if boxed:
        inner = boxed.strip()
        return f"\\boxed{{{inner}}}" if inner else ""
    inner = _clean_final_answer(a)
    return f"\\boxed{{{inner}}}" if inner else ""

def _first_nonempty_line(text: str) -> str:
    for ln in (text or "").splitlines():
        if ln.strip():
            return ln.strip()
    return ""

def _parse_new_format(model_output: str) -> Optional[Tuple[str, str]]:
    """
    Parse model output in the NEW format:
      New question: ...
      <think>...</think>
      ...
      **Final Answer:** ...
    Returns (new_question, answer_output) where answer_output excludes the New question line.
    """
    text = _strip_unwanted_tags(model_output)
    if not text:
        return None

    # Require the first non-empty line to be the New question line.
    first = _first_nonempty_line(text)
    if not first.startswith(NEW_QUESTION_PREFIX):
        return None
    new_question_inline = first[len(NEW_QUESTION_PREFIX) :].strip()

    # Require exactly one think block (case-insensitive).
    think_matches = list(re.finditer(r"(?is)<think>[\s\S]*?</think>", text))
    if len(think_matches) != 1:
        return None
    think_raw = think_matches[0].group(0)
    m_think = re.match(r"(?is)<think>([\s\S]*?)</think>", think_raw)
    if not m_think:
        return None
    think_body = (m_think.group(1) or "").strip()
    if not think_body:
        return None
    # Normalize tag casing for consistency in training data.
    think_block = f"<think>{think_body}</think>"

    # Always collect question content from the region between the "New question:" line and <think>.
    # This supports multi-line questions even when the first line already has some text.
    start_pos = text.find(first)
    q_lines: List[str] = []
    if new_question_inline:
        q_lines.append(new_question_inline)
    if start_pos >= 0:
        region = text[start_pos + len(first) : think_matches[0].start()]
        for ln in region.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith(NEW_QUESTION_PREFIX):
                ln = ln[len(NEW_QUESTION_PREFIX) :].strip()
            if ln:
                q_lines.append(ln)
    new_question = " ".join(q_lines).strip()
    if not new_question:
        return None
    # Basic completeness check: avoid extremely short or fragmentary "questions".
    if len(new_question) < 20:
        return None

    tail = text[think_matches[0].end() :].lstrip()
    if not tail.strip():
        return None

    tail_lines = tail.splitlines()
    last_idx = None
    for i in range(len(tail_lines) - 1, -1, -1):
        if tail_lines[i].strip():
            last_idx = i
            break
    if last_idx is None:
        return None

    last_line = tail_lines[last_idx].strip()
    if not last_line.startswith(FINAL_ANSWER_PREFIX):
        return None
    final_val = last_line[len(FINAL_ANSWER_PREFIX) :].strip()
    boxed_val = _ensure_boxed(final_val)
    if not boxed_val:
        return None
    tail_lines[last_idx] = f"{FINAL_ANSWER_PREFIX} {boxed_val}".rstrip()

    # Ensure nothing after the final answer except whitespace.
    after = "\n".join(tail_lines[last_idx + 1 :]).strip()
    if after:
        return None

    tail_clean = "\n".join(tail_lines).strip()
    answer_out = f"{think_block}\n\n{tail_clean}".strip()
    return new_question, answer_out


def _build_training_dict(
    original_question: str,
    model_output: str,
) -> Dict[str, Any]:
    """
    Post-process raw model text into a training dict (alpaca) for NEW question solving:
      - instruction: the NEW question text (not the original bad-case question)
      - output: <think>...</think> + solution + last line **Final Answer:** ...
    """
    original_question = original_question if isinstance(original_question, str) else ""
    original_question = original_question.strip()

    parsed = _parse_new_format(model_output)
    if not parsed:
        return {}
    new_question, answer_out = parsed

    if original_question and _too_similar(new_question, original_question):
        return {}

    # Guardrail: discourage leaking the "New question:" header into the training output.
    if answer_out.lstrip().startswith(NEW_QUESTION_PREFIX):
        return {}

    return {"instruction": new_question, "input": "", "output": answer_out}


def main() -> None:
    parser = argparse.ArgumentParser(description="基于归因结果生成训练数据")
    parser.add_argument("--input", type=str, required=True, help="归因 JSON 文件")
    parser.add_argument("--output", type=str, required=True, help="输出训练数据 JSON 文件")
    parser.add_argument(
        "--prompt",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "prompts", "data_gen.jinja"),
        help="Jinja 提示词模板路径",
    )
    parser.add_argument("--model", type=str, default="gpt-5.2", help="模型名称")
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM_PROMPT, help="system 提示词")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条")
    parser.add_argument("--max-workers", type=int, default=16, help="并发数")
    parser.add_argument("--temperature", type=float, default=0.2, help="生成温度")
    parser.add_argument("--num-per-case", type=int, default=1, help="每条归因生成样本数")
    parser.add_argument("--max-retries", type=int, default=4, help="单个样本解析/校验失败时的最大重试次数")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("输入必须是 JSON 列表")

    if args.limit:
        data = data[: args.limit]

    template = _load_template(args.prompt)
    num_per_case = max(1, int(args.num_per_case))

    def _build_one(idx_item: Tuple[int, Dict[str, Any]]) -> Tuple[int, List[Dict[str, Any]]]:
        idx, item = idx_item
        if not isinstance(item, dict):
            return idx, []
        question = item.get("question", "")
        answer = item.get("answer", "")
        if not isinstance(question, str) or not question.strip():
            question = json.dumps(item, ensure_ascii=False)
        question = question.strip()
        answer = answer if isinstance(answer, str) else ""
        answer = answer.strip()

        attr = item.get("attribution", {})
        prompt = template.render(
            attribution=attr,
            question=question,
            answer=answer,
        )
        client = init_openai_client()

        def _do_call() -> str:
            return _call_model(
                client=client,
                model=args.model,
                system_prompt=args.system,
                user_prompt=prompt,
                temperature=float(args.temperature),
            )

        samples: List[Dict[str, Any]] = []
        for _ in range(num_per_case):
            max_retries = max(1, int(getattr(args, "max_retries", 4) or 4))
            cand: Dict[str, Any] = {}
            for _attempt in range(max_retries):
                raw = _do_call()
                cand = _build_training_dict(
                    original_question=question,
                    model_output=raw,
                )
                if cand:
                    break
            if cand:
                samples.append(cand)
        return idx, samples

    results: List[Tuple[int, List[Dict[str, Any]]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
        futures = [executor.submit(_build_one, (idx, item)) for idx, item in enumerate(data)]
        with tqdm(total=len(futures), desc="生成中") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                try:
                    idx, rows = fut.result()
                    if rows:
                        results.append((idx, rows))
                except Exception as exc:
                    tqdm.write(f"[warn] one task failed: {exc!r}")
                finally:
                    pbar.update(1)

    results_sorted: List[Dict[str, Any]] = []
    for _, rows in sorted(results, key=lambda x: x[0]):
        results_sorted.extend(rows)

    random.shuffle(results_sorted)
    write_json(args.output, results_sorted)
    tqdm.write(f"完成：生成 {len(results_sorted)} 条，已写入 {args.output}")


if __name__ == "__main__":
    main()
