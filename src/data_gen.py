import argparse
import concurrent.futures
import json
import random
import os
import re
from typing import Any, Dict, List, Tuple, Optional

from jinja2 import Template
from tqdm import tqdm

from utils import init_openai_client, write_json


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
    "- Do NOT output <think> or </think>.\n"
    "\n"
    "Output format (must follow strictly):\n"
    "Question:\n"
    "<your new question>\n"
    "Solution:\n"
    "<your solution>\n"
)


QUESTION_HEADERS = ("Question:", "New question:", "Problem:")
SOLUTION_HEADERS = ("Solution:", "Answer:", "Output:")


def _load_template(path: str) -> Template:
    with open(path, "r", encoding="utf-8") as f:
        return Template(f.read())


def _load_few_shot_example(path: Optional[str], index: Optional[int]) -> Optional[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None

    def _coerce(item: Dict[str, Any]) -> Optional[Dict[str, str]]:
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        user_input = item.get("input", "")
        if not isinstance(instruction, str) or not instruction.strip():
            return None
        if not isinstance(output, str) or not output.strip():
            return None
        return {
            "instruction": instruction.strip(),
            "input": user_input.strip() if isinstance(user_input, str) else "",
            "output": output.strip(),
        }

    if index is not None:
        if 0 <= index < len(data) and isinstance(data[index], dict):
            return _coerce(data[index])
        return None

    for item in data:
        if isinstance(item, dict):
            example = _coerce(item)
            if example:
                return example
    return None


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
    """Best-effort cleanup for special training tags and <think> blocks."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Remove special training tags if they appear (some models emit these).
    text = re.sub(r"(?is)<\|begin_of_thought\|>([\s\S]*?)<\|end_of_thought\|>", r"\1", text)
    text = re.sub(r"(?is)<\|begin_of_solution\|>([\s\S]*?)<\|end_of_solution\|>", r"\1", text)
    text = re.sub(r"(?is)<think>([\s\S]*?)</think>", r"\1", text)
    text = text.replace("<think>", "").replace("</think>", "")
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

def _find_header_positions(text: str, headers: Tuple[str, ...]) -> List[Tuple[int, str]]:
    lower = text.lower()
    positions: List[Tuple[int, str]] = []
    for header in headers:
        h_low = header.lower()
        start = 0
        while True:
            idx = lower.find(h_low, start)
            if idx == -1:
                break
            positions.append((idx, header))
            start = idx + len(h_low)
    return sorted(positions, key=lambda x: x[0])


def _parse_question_solution_format(model_output: str) -> Optional[Tuple[str, str]]:
    """
    Parse model output in the format:
      Question:
      ...
      Solution:
      ...
    Returns (new_question, answer_output).
    """
    text = _strip_unwanted_tags(model_output)
    if not text:
        return None

    q_positions = _find_header_positions(text, QUESTION_HEADERS)
    s_positions = _find_header_positions(text, SOLUTION_HEADERS)
    if not q_positions or not s_positions:
        return None

    for q_idx, q_header in reversed(q_positions):
        s_after = next((s for s in s_positions if s[0] > q_idx), None)
        if not s_after:
            continue
        s_idx, s_header = s_after
        question = text[q_idx + len(q_header) : s_idx].strip()
        answer = text[s_idx + len(s_header) :].strip()
        if question and answer:
            return question, answer
    return None


def _build_training_dict(
    original_question: str,
    model_output: str,
) -> Dict[str, Any]:
    """
    Post-process raw model text into a training dict (alpaca) for NEW question solving:
      - instruction: the NEW question text (not the original bad-case question)
      - output: solution text (no <think> tags)
    """
    original_question = original_question if isinstance(original_question, str) else ""
    original_question = original_question.strip()

    parsed = _parse_question_solution_format(model_output)
    if not parsed:
        return {}
    new_question, answer_out = parsed

    if original_question and _too_similar(new_question, original_question):
        return {}

    # Guardrail: discourage leaking the prompt headers into training output.
    lower_out = answer_out.lstrip().lower()
    if lower_out.startswith("question:") or lower_out.startswith("new question:"):
        return {}
    if lower_out.startswith("solution:") or lower_out.startswith("answer:"):
        answer_out = answer_out.split(":", 1)[-1].strip()

    if len(new_question.strip()) < 20 or not answer_out.strip():
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
    parser.add_argument("--few-shot-path", type=str, default=None, help="few-shot 示例 JSON 文件路径")
    parser.add_argument("--few-shot-index", type=int, default=None, help="few-shot 示例索引（默认取首条）")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("输入必须是 JSON 列表")

    if args.limit:
        data = data[: args.limit]

    template = _load_template(args.prompt)
    few_shot = _load_few_shot_example(args.few_shot_path, args.few_shot_index)
    if args.few_shot_path and not few_shot:
        print(f"[warn] few-shot 示例加载失败: {args.few_shot_path}")
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
            few_shot=few_shot,
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
