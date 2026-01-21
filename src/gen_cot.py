import argparse
import concurrent.futures
import json
import os
from typing import Any, Dict, List, Tuple

from jinja2 import Template
from tqdm import tqdm

from utils import init_openai_client

def _load_template(path: str) -> Template:
    with open(path, "r", encoding="utf-8") as f:
        return Template(f.read())


def _render_prompt(template: Template, question: str, answer: str) -> str:
    return template.render(question=question, answer=answer)


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
) -> str:
    # Prefer Responses API when available.
    if hasattr(client, "responses"):
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = _extract_response_text(resp)
        if text:
            return text

    # Fallback to Chat Completions API.
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content
    return content.strip() if isinstance(content, str) else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="用 gpt-5.2 生成 AIME 推理过程")
    parser.add_argument("--input", type=str, required=True, help="输入 JSON 文件")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "prompts", "gen_cot.jinja"),
        help="Jinja 提示词模板路径",
    )
    parser.add_argument("--model", type=str, default="gpt-5.2", help="模型名称")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.", help="system 提示词")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条")
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
        if not question:
            question = json.dumps(item, ensure_ascii=False)
        prompt = _render_prompt(template, question=question, answer=answer)
        client = init_openai_client()
        output = _call_model(
            client=client,
            model=args.model,
            system_prompt=args.system,
            user_prompt=prompt,
        )
        return idx, {
            "instruction": question,
            "input": "",
            "output": output,
            "system": args.system,
        }

    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(_build_one, (idx, item)) for idx, item in enumerate(data)]
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="生成中",
        ):
            idx, row = fut.result()
            if row:
                results.append((idx, row))

    results = [row for _, row in sorted(results, key=lambda x: x[0])]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"完成：生成 {len(results)} 条，已写入 {args.output}")


if __name__ == "__main__":
    main()
