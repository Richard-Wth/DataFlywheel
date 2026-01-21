import json
import os
import argparse
import re
from typing import Dict, Any, List
from tqdm import tqdm

QUESTION_KEYS = ("problem", "question", "prompt", "query", "input", "text", "statement")
FINAL_ANSWER_INSTRUCTION = (
    "Please reason step by step, and put your final answer in the last line within \\boxed{} "
    "(do not write anything after the boxed answer)."
)

TRAINING_USER_PREFIX = "Return your final response within \\\\boxed{}."

XML_TAGS_INSTRUCTION = (
    "Output format requirements (must follow strictly):\n"
    "- Your entire output must consist of exactly three sections, in this order: <Plan>, <Solve>, <Answer>.\n"
    "- <Plan>: list the key theorems/strategies only; do not perform calculations.\n"
    "- <Solve>: concise derivation/calculation steps; keep essential equations.\n"
    "- <Answer>: output only the final answer; if the problem requests \\\\boxed{}, wrap it in \\\\boxed{}.\n"
    "- Do not output any other text before, between, or after these three sections.\n"
    "- Do NOT output <think> or </think>.\n"
    "- Write everything in English.\n"
    "\n"
    "You must use the tags exactly:\n"
    "<Plan>...</Plan>\n"
    "<Solve>...</Solve>\n"
    "<Answer>...</Answer>"
)

TRAINING_LONG_SYSTEM_PROMPT = (
    "Your role as an assistant is to rewrite a given math solution into a structured, teachable format.\n"
    "You must be systematic and careful: analyze the problem, identify the key ideas, and ensure the final result is correct.\n"
    "However, you must NOT reveal any hidden chain-of-thought. Do not output internal deliberations.\n"
    "\n"
    "Output format requirements (must follow strictly):\n"
    "- Your entire output must consist of exactly three sections, in this order: <Plan>, <Solve>, <Answer>.\n"
    "- <Plan>: list the key theorems/strategies only; do not perform calculations.\n"
    "- <Solve>: follow the Plan and present the most concise derivation/calculation steps; keep essential equations, skip tedious arithmetic.\n"
    "- <Answer>: output only the final answer. If the problem requests \\\\boxed{}, wrap the final answer in \\\\boxed{}.\n"
    "- Do not output any other text before, between, or after these three sections.\n"
    "\n"
    "Language requirements:\n"
    "- Write everything in English, even if the problem statement or the original solution contains other languages.\n"
)

_BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
_HASH_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)

_CKPT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


def _read_json_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _has_any_weights_files(model_dir: str) -> bool:
    if not model_dir or not os.path.isdir(model_dir):
        return False
    for name in os.listdir(model_dir):
        low = name.lower()
        if low.endswith(".safetensors") or low.endswith(".bin"):
            # ignore pure index json
            if low.endswith(".index.json"):
                continue
            return True
    return False


def _index_is_satisfied(model_dir: str) -> bool:
    """If an index json exists, ensure all referenced shard files are present."""
    if not model_dir or not os.path.isdir(model_dir):
        return True
    # common hf index filenames
    idx_candidates = [
        os.path.join(model_dir, "model.safetensors.index.json"),
        os.path.join(model_dir, "pytorch_model.bin.index.json"),
    ]
    idx_path = next((p for p in idx_candidates if os.path.isfile(p)), None)
    if not idx_path:
        return True

    data = _read_json_file(idx_path)
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return True
    shard_files = sorted({v for v in weight_map.values() if isinstance(v, str) and v.strip()})
    # If index exists but it references nothing, treat as satisfied.
    if not shard_files:
        return True
    for shard in shard_files:
        if not os.path.isfile(os.path.join(model_dir, shard)):
            return False
    return True


def _is_usable_model_dir(model_dir: str) -> bool:
    return _has_any_weights_files(model_dir) and _index_is_satisfied(model_dir)


def _find_latest_checkpoint_dir(train_output_dir: str) -> str:
    """Pick latest checkpoint-* directory by step number (highest)."""
    if not train_output_dir or not os.path.isdir(train_output_dir):
        return ""
    best_step = None
    best_path = ""
    for name in os.listdir(train_output_dir):
        m = _CKPT_DIR_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(train_output_dir, name)
        if not os.path.isdir(path):
            continue
        if best_step is None or step > best_step:
            best_step = step
            best_path = path
    return best_path


def _resolve_model_dir(model_dir: str, ckpt: int | None) -> str:
    """Resolve to a directory that actually contains loadable weights for vLLM/HF."""
    if not model_dir or not os.path.isdir(model_dir):
        return model_dir

    # Explicit checkpoint wins if present.
    if ckpt is not None:
        ckpt_dir = os.path.join(model_dir, f"checkpoint-{ckpt}")
        if os.path.isdir(ckpt_dir):
            return ckpt_dir

    # If current dir looks good, keep it.
    if _is_usable_model_dir(model_dir):
        return model_dir

    # Common failure mode: output dir contains an index json referencing missing shards.
    latest = _find_latest_checkpoint_dir(model_dir)
    if latest and _is_usable_model_dir(latest):
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.isfile(idx_path) and not _index_is_satisfied(model_dir):
            print(
                f"[WARN] 检测到 `{idx_path}` 引用了缺失的分片权重，"
                f"自动改用最新 checkpoint: {latest}"
            )
        else:
            print(f"[INFO] 当前目录未找到可加载权重，自动改用最新 checkpoint: {latest}")
        return latest

    # Fallback: try any checkpoint dir that is usable.
    for name in sorted(os.listdir(model_dir), reverse=True):
        m = _CKPT_DIR_RE.match(name)
        if not m:
            continue
        path = os.path.join(model_dir, name)
        if _is_usable_model_dir(path):
            print(f"[INFO] 当前目录未找到可加载权重，自动改用 checkpoint: {path}")
            return path

    return model_dir


def _get_text_field(problem: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = problem.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

def get_problem_question(problem: Dict[str, Any]) -> str:
    dataset_id = problem.get("_dataset_id")
    if dataset_id == "openai/gsm8k":
        question = _get_text_field(problem, ["question"])
    elif dataset_id == "HuggingFaceH4/MATH-500":
        question = _get_text_field(problem, ["problem", "question"])
    elif dataset_id == "Hwilner/imo-answerbench":
        question = _get_text_field(problem, ["problem", "question", "Problem"])
    elif dataset_id in {"MathArena/hmmt_nov_2025", "MathArena/hmmt_feb_2025"}:
        question = _get_text_field(problem, ["problem", "question"])
    elif dataset_id == "TIGER-Lab/AIME25":
        question = _get_text_field(problem, ["question"])
    else:
        question = _get_text_field(problem, list(QUESTION_KEYS))
    
    return question or json.dumps(problem, ensure_ascii=False)

def _augment_question(question: str) -> str:
    if FINAL_ANSWER_INSTRUCTION in question:
        return question
    trimmed = question.rstrip()
    if trimmed:
        return f"{trimmed}\n\n{FINAL_ANSWER_INSTRUCTION}"
    return FINAL_ANSWER_INSTRUCTION


def _augment_question_with_xml_tags(question: str) -> str:
    """Augment question with an instruction to output <Plan>/<Solve>/<Answer> tags."""
    if XML_TAGS_INSTRUCTION in question:
        return question
    trimmed = question.rstrip()
    if trimmed:
        return f"{trimmed}\n\n{XML_TAGS_INSTRUCTION}"
    return XML_TAGS_INSTRUCTION


def _format_user_prompt_like_training(problem: Dict[str, Any], question_text: str) -> str:
    """Format user prompt to match training data style (alpaca: instruction + optional input).

    Training example (`openthoughts_math_5k_milestone.json`) uses:
    - instruction: starts with "Return your final response within \\boxed{}." + problem statement
    - input: usually empty
    """
    # Prefer explicit alpaca-style fields if present.
    instruction = _get_text_field(problem, ["instruction"])
    user_input = _get_text_field(problem, ["input"])

    if not instruction:
        instruction = question_text.strip() if isinstance(question_text, str) else ""

    # Ensure the boxed prefix exists (match training style).
    if instruction and not instruction.lstrip().startswith(TRAINING_USER_PREFIX):
        instruction = f"{TRAINING_USER_PREFIX} {instruction}"

    if user_input:
        return f"{instruction}\n\n{user_input}".strip()
    return instruction.strip()


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks and stray think tags."""
    if not isinstance(text, str) or not text:
        return ""
    text = _THINK_BLOCK_RE.sub("", text)
    text = text.replace("<think>", "").replace("</think>", "")
    return text


def normalize_plan_solve_answer(text: str) -> str:
    """Normalize model output to exactly three sections: <Plan>, <Solve>, <Answer>.

    This fixes common issues:
    - extra <think> blocks
    - missing </Plan> or </Solve>
    - extra text outside the three sections
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _strip_think(text).strip()

    def _find(tag: str) -> int:
        return text.find(tag)

    p0, s0, a0 = _find("<Plan>"), _find("<Solve>"), _find("<Answer>")
    if p0 == -1 and s0 == -1 and a0 == -1:
        return text

    # Boundaries: prefer explicit closing tags; otherwise use next opening tag.
    def _section(open_tag: str, close_tag: str, next_open_tags: list[str]) -> str:
        start = text.find(open_tag)
        if start == -1:
            return ""
        start += len(open_tag)
        end = text.find(close_tag, start)
        if end == -1:
            # use earliest next opening tag as fallback boundary
            candidates = [text.find(t, start) for t in next_open_tags]
            candidates = [c for c in candidates if c != -1]
            end = min(candidates) if candidates else len(text)
        return text[start:end].strip()

    plan = _section("<Plan>", "</Plan>", ["<Solve>", "<Answer>"])
    solve = _section("<Solve>", "</Solve>", ["<Answer>"])
    answer = _section("<Answer>", "</Answer>", [])
    # If </Answer> is missing, take tail after <Answer>.
    if a0 != -1 and not answer:
        answer = text[a0 + len("<Answer>") :].strip()

    plan = _strip_think(plan).strip()
    solve = _strip_think(solve).strip()
    answer = _strip_think(answer).strip()

    return (
        "<Plan>\n"
        + plan
        + "\n</Plan>\n\n"
        + "<Solve>\n"
        + solve
        + "\n</Solve>\n\n"
        + "<Answer>\n"
        + answer
        + "\n</Answer>"
    ).strip()


def extract_final_answer(text: str) -> str:
    """Extract final answer from model output.

    Priority:
    1) last \\boxed{...}
    2) last '#### ...' line (some datasets use this)
    3) last non-empty line
    """
    if not isinstance(text, str):
        return ""
    boxed = _BOX_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    m = _HASH_RE.search(text)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""

def main():
    parser = argparse.ArgumentParser(description="Inference with vLLM")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/test/My_codes/West/ID/LlamaFactory/saves/qwen3-8b/full_sft/ds_math_thinking_xmltags",
        help="Model path (can be a checkpoint dir like .../checkpoint-471).",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=None,
        help="If set, and --model points to the training output dir, use checkpoint-<ckpt> under it.",
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tp", type=int, default=8, help="Tensor Parallel Size")
    parser.add_argument("--temperature", type=float, default=0.7, help="Inference temperature (higher -> more diverse).")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling top_p (math: 0.9~0.98).")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling (math: 0~100, -1 disables).")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Maximum generation length")
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples per question (pass@k).")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of questions to infer")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "transformers"],
        help="Inference backend: vllm (default) or transformers fallback.",
    )
    parser.add_argument(
        "--use_xml_tags",
        action="store_true",
        help="If set, add an instruction to force outputs to contain <Plan>/<Solve>/<Answer> tags.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="<Plan>,</Plan>,<Solve>,</Solve>,<Answer>,</Answer>",
        help="Comma-separated tags to ensure tokenizer knows them (optional).",
    )
    parser.add_argument(
        "--decode_skip_special_tokens",
        action="store_true",
        help="If set, decode with skip_special_tokens=True (default: False, keeps tags).",
    )
    parser.add_argument(
        "--normalize_xml_output",
        action="store_true",
        help="If set, post-process outputs into exactly <Plan>/<Solve>/<Answer> (recommended).",
    )
    
    args = parser.parse_args()

    llm = None
    sampling_params = None
    use_vllm = args.backend == "vllm"

    if use_vllm:
        try:
            # Lazy import heavy deps so `--help` works even if vLLM stack isn't installed.
            from vllm import LLM, SamplingParams
        except Exception as exc:
            print(f"[warn] vLLM not available, falling back to transformers: {exc}")
            use_vllm = False

    # Resolve a loadable model directory (fix common index/shard mismatches).
    if args.model and os.path.isdir(args.model):
        args.model = _resolve_model_dir(args.model, args.ckpt)

    with open(args.input, "r", encoding="utf-8") as f:
        problems = json.load(f)
    if args.limit:
        problems = problems[:args.limit]
        print(f"限制处理前 {args.limit} 个问题")
    print(f"共加载 {len(problems)} 个问题")

    if use_vllm:
        llm_kwargs = dict(
            model=args.model,
            tensor_parallel_size=args.tp,
            trust_remote_code=True,
        )
        if args.max_model_len is not None:
            llm_kwargs["max_model_len"] = args.max_model_len
        llm = LLM(**llm_kwargs)

        sampling_kwargs = dict(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            n=args.num_samples,
        )
        # When we want XML tags, stop exactly at </Answer> and suppress <think>.
        if args.use_xml_tags:
            sampling_kwargs.update(
                dict(
                    stop="</Answer>",
                    include_stop_str_in_output=True,
                )
            )
        sampling_params = SamplingParams(**sampling_kwargs)

    formatted_prompts = []
    question_texts = []
    question_with_instructions = []
    if use_vllm:
        tokenizer = llm.get_tokenizer()
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        import importlib.util

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Loading with device_map="auto" requires `accelerate`. Provide a clear error
        # instead of a long ValueError stacktrace.
        if importlib.util.find_spec("accelerate") is None:
            raise RuntimeError(
                "Transformers backend requires `accelerate` when using device_map=\"auto\". "
                "Please install it (pip install accelerate) or run with --backend vllm."
            )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )

    # Ensure tokenizer contains tags (<Plan> etc.) so decode shows them.
    # This is safe even if the tokenizer already has these tokens (it will add 0 new tokens).
    if isinstance(args.tags, str) and args.tags.strip():
        tag_list = [t.strip() for t in args.tags.split(",") if t.strip()]
        try:
            tokenizer.add_tokens(tag_list, special_tokens=False)
        except Exception:
            # Some tokenizers might not expose add_tokens in vLLM wrapper; ignore silently.
            pass
    
    for problem in tqdm(problems, desc="格式化数据"):
        question_text = get_problem_question(problem)
        # Match training data: when using XML tags, put only instruction/input in the USER message.
        # All formatting constraints live in the (restored) long SYSTEM prompt.
        user_prompt = (
            _format_user_prompt_like_training(problem, question_text)
            if args.use_xml_tags
            else _augment_question(question_text)
        )
        question_texts.append(question_text)
        question_with_instructions.append(user_prompt)
        
        # Restore the training-time long system prompt when XML tags are enabled.
        system_prompt = "You are a helpful assistant."
        if args.use_xml_tags:
            system_prompt = TRAINING_LONG_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(text)

    print(f"开始推理...")
    outputs = None
    if use_vllm:
        outputs = llm.generate(formatted_prompts, sampling_params)

    results = []
    if use_vllm:
        for problem, output, question_text, user_prompt in zip(problems, outputs, question_texts, question_with_instructions):
            # Prefer decoding from token ids to control skip_special_tokens behavior (so tags are preserved).
            predictions_texts = []
            for o in output.outputs:
                token_ids = getattr(o, "token_ids", None)
                if token_ids is not None:
                    txt = tokenizer.decode(token_ids, skip_special_tokens=args.decode_skip_special_tokens).strip()
                else:
                    txt = (o.text or "").strip()
                if args.use_xml_tags or args.normalize_xml_output:
                    txt = normalize_plan_solve_answer(txt)
                predictions_texts.append(txt)
            predictions = [{"output": p, "final_answer": extract_final_answer(p)} for p in predictions_texts]
            result = {
                "question": question_text,
                "answer": problem.get("answer", ""),
                "dataset": problem.get("_dataset", ""),
                "sample_index": problem.get("_sample_index", ""),
                "predictions": predictions,
            }
            results.append(result)
    else:
        from transformers import GenerationConfig
        import torch

        gen_config = GenerationConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_tokens,
            do_sample=True if args.temperature and args.temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for problem, question_text, user_prompt in zip(problems, question_texts, question_with_instructions):
            predictions_texts = []
            for _ in range(int(args.num_samples)):
                inputs = tokenizer(user_prompt, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    output_ids = model.generate(**inputs, generation_config=gen_config)
                generated = output_ids[0, inputs["input_ids"].shape[1]:]
                txt = tokenizer.decode(generated, skip_special_tokens=args.decode_skip_special_tokens).strip()
                if args.use_xml_tags or args.normalize_xml_output:
                    txt = normalize_plan_solve_answer(txt)
                predictions_texts.append(txt)
            predictions = [{"output": p, "final_answer": extract_final_answer(p)} for p in predictions_texts]
            result = {
                "question": question_text,
                "answer": problem.get("answer", ""),
                "dataset": problem.get("_dataset", ""),
                "sample_index": problem.get("_sample_index", ""),
                "predictions": predictions,
            }
            results.append(result)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"推理完成，结果已保存到: {args.output}")
    print(f"成功推理 {len(results)} 个问题")

if __name__ == "__main__":
    main()
