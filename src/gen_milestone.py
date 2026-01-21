import argparse
import concurrent.futures
import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

from jinja2 import Template
from tqdm import tqdm

from utils import init_openai_client


DEFAULT_MILESTONE_SYSTEM_PROMPT = (
    "Role: You are a math professor demonstrating perfect reasoning to a student.\n"
    "\n"
    "Task: Solve the given math problem using a strictly structured approach with alternating Plan and Solve steps.\n"
    "\n"
    "Format requirements (must follow strictly):\n"
    "1) Do NOT include or restate the problem in the output.\n"
    "2) Output only alternating <Plan> and <Solve> sections for each step, then <Answer>.\n"
    "   - Step 1: <Plan> then <Solve>\n"
    "   - Step 2: <Plan> then <Solve>\n"
    "   - Continue until the solution is complete.\n"
    "3) Each <Plan> contains exactly one high-level logical step (no calculations).\n"
    "4) Each <Solve> must execute the immediately preceding <Plan> step.\n"
    "   - CRITICAL: be extremely verbose in calculations.\n"
    "   - Do NOT jump from equation A to result B. Show the algebra in between.\n"
    "   - Include every single intermediate calculation step.\n"
    "   - Do not simplify equations in one go. Treat this as a scratchpad. Validate each arithmetic operation.\n"
    "5) <Answer>:\n"
    "   - Output only the final result, boxed.\n"
    "\n"
    "Language requirements:\n"
    "- Write everything in English, even if the problem statement or the original solution contains other languages.\n"
    "\n"
    "Do NOT output any other text besides alternating <Plan>/<Solve> steps and the final <Answer>.\n"
)


def _load_template(path: str) -> Template:
    with open(path, "r", encoding="utf-8") as f:
        return Template(f.read())


def _render_prompt(template: Template, question: str, solution: str) -> str:
    return template.render(question=question, solution=solution)


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
    # Prefer Responses API when available.
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

    # Fallback to Chat Completions API.
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


def _concat_instruction_input(instruction: str, inp: str) -> str:
    instruction = instruction if isinstance(instruction, str) else ""
    inp = inp if isinstance(inp, str) else ""
    instruction = instruction.strip()
    inp = inp.strip()
    if instruction and inp:
        return f"{instruction}\n\n{inp}"
    return instruction or inp


def _with_retries(fn, max_retries: int = 3, base_sleep: float = 1.0) -> str:
    last_err: Exception = None  # type: ignore[assignment]
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_err = exc
            if attempt >= max_retries:
                break
            # jitter + exponential backoff
            sleep_s = base_sleep * (2**attempt) * (0.8 + 0.4 * random.random())
            time.sleep(sleep_s)
    raise last_err  # type: ignore[misc]


def main() -> None:
    parser = argparse.ArgumentParser(description="将 Long CoT 重写为结构化 induction（Plan/Solve/Answer）")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "training", "openthoughts_math_5k.json"),
        help="输入 JSON 文件（list）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "training",
            "openthoughts_math_5k_milestone.json",
        ),
        help="输出 JSON 文件（list）",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "prompts", "milestone.jinja"),
        help="Jinja 提示词模板路径",
    )
    parser.add_argument("--model", type=str, default="gpt-5.2", help="模型名称")
    parser.add_argument("--system", type=str, default=DEFAULT_MILESTONE_SYSTEM_PROMPT, help="system 提示词")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条")
    parser.add_argument("--max-workers", type=int, default=32, help="并发数（默认 32）")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度（默认 0）")
    parser.add_argument("--retries", type=int, default=3, help="失败重试次数（默认 3）")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("输入必须是 JSON 列表")
    if args.limit:
        data = data[: args.limit]
        print(f"限制处理前 {args.limit} 条")
    print(f"共加载 {len(data)} 条样本")

    template = _load_template(args.prompt)

    def _build_one(idx_item: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        idx, item = idx_item
        if not isinstance(item, dict):
            return idx, {}

        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        question = _concat_instruction_input(instruction, inp)
        if not question:
            question = json.dumps(item, ensure_ascii=False)

        # 原始解答（通常包含长 CoT + solution）
        solution = item.get("output", "")
        solution = solution if isinstance(solution, str) else ""

        user_prompt = _render_prompt(template, question=question, solution=solution)

        client = init_openai_client()

        def _do_call() -> str:
            return _call_model(
                client=client,
                model=args.model,
                system_prompt=args.system,
                user_prompt=user_prompt,
                temperature=float(args.temperature),
            )

        output = _with_retries(_do_call, max_retries=int(args.retries))
        return idx, {
            "instruction": instruction,
            "input": inp if isinstance(inp, str) else "",
            "output": output,
            "system": args.system,
        }

    results: List[Tuple[int, Dict[str, Any]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
        futures = [executor.submit(_build_one, (idx, item)) for idx, item in enumerate(data)]
        with tqdm(total=len(futures), desc="生成中") as pbar:
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
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)

    tqdm.write(f"完成：生成 {len(results_sorted)} 条，已写入 {args.output}")


if __name__ == "__main__":
    main()

