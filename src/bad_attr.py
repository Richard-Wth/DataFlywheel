import argparse
import concurrent.futures
import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

from jinja2 import Template
from tqdm import tqdm

from utils import init_openai_client, robust_json_parse, write_json


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert math tutor and evaluator. "
    "Analyze why a model answered a problem incorrectly, "
    "and provide actionable, concise attributions. "
    "Do not reveal chain-of-thought. Return JSON only."
)


def _load_template(path: str) -> Template:
    with open(path, "r", encoding="utf-8") as f:
        return Template(f.read())


def _render_prompt(template: Template, question: str, answer: str, predictions: List[str]) -> str:
    preds = [p for p in predictions if isinstance(p, str)]
    return template.render(question=question, answer=answer, predictions=preds)


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
    temperature: float = 0.0,
) -> str:
    # Gemini preview models often only support chat.completions in this setup.
    if isinstance(model, str) and model.startswith("gemini"):
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


def _with_retries(fn, max_retries: int = 3, base_sleep: float = 1.0) -> str:
    last_err: Exception = None  # type: ignore[assignment]
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_err = exc
            if attempt >= max_retries:
                break
            sleep_s = base_sleep * (2**attempt) * (0.8 + 0.4 * random.random())
            time.sleep(sleep_s)
    raise last_err  # type: ignore[misc]


def _safe_predictions(item: Dict[str, Any]) -> List[str]:
    preds = item.get("predictions", [])
    if isinstance(preds, list):
        if preds and isinstance(preds[0], dict):
            return [p.get("output", "") for p in preds if isinstance(p, dict)]
        return [p for p in preds if isinstance(p, str)]
    pred = item.get("prediction", "")
    return [pred] if isinstance(pred, str) and pred else []


def main() -> None:
    parser = argparse.ArgumentParser(description="对 bad cases 进行归因分析")
    parser.add_argument("--input", type=str, required=True, help="bad cases JSON 文件")
    parser.add_argument("--output", type=str, required=True, help="归因输出 JSON 文件")
    parser.add_argument(
        "--prompt",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "prompts", "bad_attr.jinja"),
        help="Jinja 提示词模板路径",
    )
    parser.add_argument("--model", type=str, default="gpt-5.2", help="模型名称")
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM_PROMPT, help="system 提示词")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条")
    parser.add_argument("--max-workers", type=int, default=16, help="并发数")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度")
    parser.add_argument("--retries", type=int, default=3, help="失败重试次数")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("输入必须是 JSON 列表")

    if args.limit:
        data = data[: args.limit]

    template = _load_template(args.prompt)

    def _build_one(idx_item: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        idx, item = idx_item
        if not isinstance(item, dict):
            return idx, {}
        question = item.get("question", "")
        answer = item.get("answer", "")
        predictions = _safe_predictions(item)
        if not question:
            question = json.dumps(item, ensure_ascii=False)

        prompt = _render_prompt(template, question=question, answer=answer, predictions=predictions)
        client = init_openai_client()

        def _do_call() -> str:
            return _call_model(
                client=client,
                model=args.model,
                system_prompt=args.system,
                user_prompt=prompt,
                temperature=float(args.temperature),
            )

        output = _with_retries(_do_call, max_retries=int(args.retries))
        parsed = robust_json_parse(output) or {"raw_text": output}
        row = {
            "question": question,
            "answer": answer,
            "predictions": predictions,
            "attribution": parsed,
        }
        return idx, row

    results: List[Tuple[int, Dict[str, Any]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
        futures = [executor.submit(_build_one, (idx, item)) for idx, item in enumerate(data)]
        with tqdm(total=len(futures), desc="归因中") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                try:
                    idx, row = fut.result()
                    if row:
                        results.append((idx, row))
                except Exception as exc:
                    tqdm.write(f"[warn] one task failed: {exc!r}")
                finally:
                    pbar.update(1)

    results_sorted = [row for _, row in sorted(results, key=lambda x: x[0])]
    write_json(args.output, results_sorted)
    tqdm.write(f"完成：归因 {len(results_sorted)} 条，已写入 {args.output}")


if __name__ == "__main__":
    main()
