import argparse
import concurrent.futures
import json
import os
import re
from typing import Any, Dict, List, Tuple

from jinja2 import Template
from tqdm import tqdm

from utils import init_openai_client, write_json


DEFAULT_SYSTEM_PROMPT = (
    "You are a data generator for math contest model training. "
    "Given an original question and its failure attribution, write a NEW, detailed, fully correct solution. "
    "Do NOT output JSON. Do NOT output any special tags. "
    "The final answer MUST be in \\boxed{...} and MUST be the last thing you write. "
    "Do NOT include any policy/disclaimer text. "
)


TRAINING_USER_PREFIX = "Return your final response within \\\\boxed{}."

_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)
_TAG_BLOCK_RE = re.compile(
    r"<\|begin_of_thought\|>[\s\S]*?<\|end_of_thought\|>\s*"
    r"|<\|begin_of_solution\|>[\s\S]*?<\|end_of_solution\|>\s*",
    re.IGNORECASE,
)


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
    """Best-effort cleanup if model still outputs think/tags."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _THINK_BLOCK_RE.sub("", text)
    text = text.replace("<think>", "").replace("</think>", "")
    # Remove the special training tags if they appear.
    text = _TAG_BLOCK_RE.sub("", text)
    return text.strip()


def _build_training_dict(
    question: str,
    model_output: str,
) -> Dict[str, Any]:
    """Post-process raw model text into a training dict with 3 keys."""
    question = question if isinstance(question, str) else ""
    question = question.strip()

    out = _strip_unwanted_tags(model_output)
    if not out:
        return {}

    intro = f"{TRAINING_USER_PREFIX} {question}".strip() if question else TRAINING_USER_PREFIX
    return {
        "introduction": intro,
        "input": "",
        "output": out,
    }


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
            raw = _do_call()
            cand = _build_training_dict(
                question=question,
                model_output=raw,
            )
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

    write_json(args.output, results_sorted)
    tqdm.write(f"完成：生成 {len(results_sorted)} 条，已写入 {args.output}")


if __name__ == "__main__":
    main()
