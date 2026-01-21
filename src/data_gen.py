import argparse
import concurrent.futures
import json
import os
from typing import Any, Dict, List, Tuple

from jinja2 import Template
from tqdm import tqdm

from utils import init_openai_client, robust_json_parse, write_json


DEFAULT_SYSTEM_PROMPT = (
    "You are a data generator for math contest model training. "
    "Generate high-quality, diverse training samples based on the given failure attribution. "
    "You MUST include a detailed chain-of-thought inside <|begin_of_thought|>...<|end_of_thought|>, then a clean solution inside "
    "<|begin_of_solution|>...<|end_of_solution|>. "
    "Return JSON only."
)

TRAINING_SYSTEM_PROMPT = (
    "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. "
    "This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. "
    "Please structure your response into two main sections: Thought and Solution. "
    "In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> "
    "Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. "
    "In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. "
    "The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> "
    "Now, try to solve the following question through the above guidelines:"
)


def _canonicalize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce fields required by training format, regardless of model quirks."""
    if not isinstance(sample, dict):
        return {}
    s = dict(sample)
    # Force training system prompt to match the base training data.
    s["system"] = TRAINING_SYSTEM_PROMPT
    if "input" not in s or not isinstance(s.get("input"), str):
        s["input"] = ""
    return s


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


def _normalize_samples(obj: Any) -> List[Dict[str, Any]]:
    """Accept either a single sample dict or a list of sample dicts."""
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return [s for s in obj if isinstance(s, dict)]
    return []


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
    parser.add_argument("--max-regen", type=int, default=6, help="生成结果不合格时的重采样次数（每条样本）")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("输入必须是 JSON 列表")

    if args.limit:
        data = data[: args.limit]

    template = _load_template(args.prompt)
    num_per_case = max(1, int(args.num_per_case))
    max_regen = max(1, int(args.max_regen))

    def _build_one(idx_item: Tuple[int, Dict[str, Any]]) -> Tuple[int, List[Dict[str, Any]]]:
        idx, item = idx_item
        if not isinstance(item, dict):
            return idx, []
        question = item.get("question", "")
        if not isinstance(question, str) or not question.strip():
            question = json.dumps(item, ensure_ascii=False)
        question = question.strip()

        attr = item.get("attribution", {})
        prompt = template.render(
            attribution=attr,
            question=question,
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
            last_reason = "not_generated"
            for _regen in range(max_regen):
                raw = _do_call()
                parsed = robust_json_parse(raw)
                if not parsed:
                    last_reason = "json_parse_failed"
                    continue
                cand_list = _normalize_samples(parsed)
                if not cand_list:
                    last_reason = "no_candidate"
                    continue
                cand = _canonicalize_sample(cand_list[0])
                # Minimal sanity check only: keep pipeline-compatible keys.
                inst = cand.get("instruction")
                out = cand.get("output")
                if not (isinstance(inst, str) and inst.strip()):
                    last_reason = "missing_or_empty_instruction"
                    continue
                if not (isinstance(out, str) and out.strip()):
                    last_reason = "missing_or_empty_output"
                    continue
                samples.append(cand)
                break
            else:
                tqdm.write(f"[warn] skip one sample idx={idx}: {last_reason}")
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
