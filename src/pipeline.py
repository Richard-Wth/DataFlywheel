import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import random
import sys
import atexit

from utils import extract_last_boxed, read_json, write_json

MATH500_DATASET_ID = "HuggingFaceH4/MATH-500"
NUMINA_DATASET_ID = "AI-MO/NuminaMath-CoT"
QUESTION_FIELD_CANDIDATES = (
    "problem",
    "question",
    "prompt",
    "query",
    "input",
    "text",
    "statement",
    "instruction",
)
SOLUTION_FIELD_CANDIDATES = (
    "solution",
    "output",
    "response",
    "completion",
    "analysis",
    "answer",
)
FINAL_FIELD_CANDIDATES = ("final_answer", "final", "result", "answer")
_HASH_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _run_cmd(cmd: List[str], dry_run: bool = False, env: Optional[Dict[str, str]] = None) -> None:
    print("运行命令:", " ".join(cmd))
    if dry_run:
        return
    if _LOG_FH is not None:
        subprocess.run(cmd, check=True, env=env, stdout=_LOG_FH, stderr=_LOG_FH)
    else:
        subprocess.run(cmd, check=True, env=env)


class _Tee:
    def __init__(self, *streams):
        self.streams = [s for s in streams if s]

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


_LOG_FH = None


def _format_cmd(template: str, **kwargs) -> List[str]:
    formatted = template.format(**kwargs)
    return shlex.split(formatted)


def _get_text_field(example: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_answer_value(value: Any) -> str:
    if isinstance(value, dict):
        if "value" in value:
            value = value.get("value")
        else:
            return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        if len(value) == 1:
            value = value[0]
        else:
            return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value).strip()


def _ensure_boxed_final(final: str) -> str:
    if not isinstance(final, str):
        return ""
    final = final.strip()
    if not final:
        return ""
    if "\\boxed{" in final:
        return final
    return f"\\boxed{{{final}}}"


def _dataset_name_from_id(dataset_id: str) -> str:
    if not dataset_id:
        return "dataset"
    return dataset_id.split("/")[-1].lower()


def _load_dataset_split(
    dataset_id: str,
    split: Optional[str],
    config: Optional[str],
):
    from datasets import load_dataset

    if split:
        try:
            if config:
                return load_dataset(dataset_id, config, split=split)
            return load_dataset(dataset_id, split=split)
        except Exception:
            pass

    if config:
        dataset_obj = load_dataset(dataset_id, config)
    else:
        dataset_obj = load_dataset(dataset_id)

    if hasattr(dataset_obj, "keys"):
        if split and split in dataset_obj:
            return dataset_obj[split]
        for split_name in ("test", "validation", "train"):
            if split_name in dataset_obj:
                return dataset_obj[split_name]
        first_split = next(iter(dataset_obj.keys()))
        return dataset_obj[first_split]
    return dataset_obj


def _extract_solution_text(example: Dict[str, Any]) -> str:
    text = _get_text_field(example, SOLUTION_FIELD_CANDIDATES)
    if text:
        return text
    for key in FINAL_FIELD_CANDIDATES:
        if key in example:
            return _normalize_answer_value(example.get(key))
    return ""


def _extract_final_answer(example: Dict[str, Any], solution_text: str) -> str:
    final = _get_text_field(example, FINAL_FIELD_CANDIDATES)
    if final:
        return final
    if solution_text:
        boxed = extract_last_boxed(solution_text)
        if boxed:
            return boxed
        m = _HASH_RE.search(solution_text)
        if m:
            return m.group(1).strip()
    return ""


def _merge_json_lists(paths: List[str], output_path: str, shuffle: bool = False, seed: int = 42) -> int:
    merged = []
    for p in paths:
        if not p:
            continue
        data = read_json(p)
        if isinstance(data, list):
            merged.extend(data)
    if shuffle and len(merged) > 1:
        rng = random.Random(seed)
        rng.shuffle(merged)
    write_json(output_path, merged)
    return len(merged)


def _count_json_list(path: str) -> int:
    data = read_json(path)
    return len(data) if isinstance(data, list) else 0


def _compare_stats(stat_path_a: str, stat_path_b: str, output_path: str) -> None:
    def _load(path: str) -> Dict[str, float]:
        data = read_json(path)
        overall = (data or {}).get("overall", {}) if isinstance(data, dict) else {}
        return {
            k: float(v) for k, v in overall.items() if isinstance(v, (int, float))
        }

    a = _load(stat_path_a)
    b = _load(stat_path_b)
    keys = sorted(set(a.keys()) | set(b.keys()))
    rows = []
    for k in keys:
        rows.append(
            {
                "metric": k,
                "iter_a": a.get(k),
                "iter_b": b.get(k),
                "delta": (b.get(k) - a.get(k)) if (k in a and k in b) else None,
            }
        )
    out = {
        "iter_a_stat": stat_path_a,
        "iter_b_stat": stat_path_b,
        "metrics": rows,
    }
    write_json(output_path, out)


def select_benchmark(output_path: str, mode: str) -> str:
    from data_select import DataSelector

    selector = DataSelector(mode=mode)
    data = selector.select_and_sample_data()
    selector.save_data(data, output_path)
    selector.print_statistics(data)
    return output_path


def _set_hf_home(hf_home: str) -> None:
    if not hf_home:
        return
    hf_home = os.path.abspath(hf_home)
    os.environ.setdefault("HF_HOME", hf_home)
    os.makedirs(hf_home, exist_ok=True)


def _set_offline(offline: bool) -> None:
    if not offline:
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def build_math500_benchmark(
    output_path: str,
    split: str = "test",
    limit: Optional[int] = None,
) -> str:
    return build_hf_benchmark(
        output_path=output_path,
        dataset_id=MATH500_DATASET_ID,
        split=split,
        limit=limit,
        config=None,
        dataset_name="math500",
    )


def build_hf_benchmark(
    output_path: str,
    dataset_id: str,
    split: str = "test",
    limit: Optional[int] = None,
    config: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> str:
    ds = _load_dataset_split(dataset_id=dataset_id, split=split, config=config)
    if limit is not None:
        if hasattr(ds, "select"):
            ds = ds.select(range(int(limit)))
        else:
            ds = list(ds)[: int(limit)]

    name = dataset_name or _dataset_name_from_id(dataset_id)
    rows: List[Dict[str, Any]] = []
    difficulty_default = "hard" if dataset_id == MATH500_DATASET_ID else "unknown"
    for i, ex in enumerate(ds):
        ex = dict(ex)
        question = _get_text_field(ex, QUESTION_FIELD_CANDIDATES)
        if not question:
            question = json.dumps(ex, ensure_ascii=False)
        solution = _extract_solution_text(ex)
        final = _extract_final_answer(ex, solution)
        answer = solution
        if final:
            if not answer:
                answer = final
            else:
                if not extract_last_boxed(answer) and not _HASH_RE.search(answer):
                    answer = f"{answer}\n\n{_ensure_boxed_final(final)}"

        row = {
            "question": question,
            "answer": answer,
            "_dataset": name,
            "_dataset_id": dataset_id,
            "_difficulty": ex.get("_difficulty", difficulty_default),
            "_sample_index": i,
        }
        rows.append(row)

    _ensure_dir(os.path.dirname(output_path))
    write_json(output_path, rows)
    print(f"已构建 benchmark: {output_path} ({len(rows)} 条, {dataset_id}::{split})")
    return output_path


def build_hf_training_data(
    output_path: str,
    dataset_id: str,
    split: str = "train",
    limit: Optional[int] = None,
    config: Optional[str] = None,
) -> str:
    ds = _load_dataset_split(dataset_id=dataset_id, split=split, config=config)
    if limit is not None:
        if hasattr(ds, "select"):
            ds = ds.select(range(int(limit)))
        else:
            ds = list(ds)[: int(limit)]

    rows: List[Dict[str, Any]] = []
    for ex in ds:
        ex = dict(ex)
        question = _get_text_field(ex, QUESTION_FIELD_CANDIDATES)
        if not question:
            question = json.dumps(ex, ensure_ascii=False)
        solution = _extract_solution_text(ex)
        if not solution:
            solution = _normalize_answer_value(ex.get("final_answer"))
        if not solution:
            continue
        rows.append({"instruction": question, "input": "", "output": solution})

    _ensure_dir(os.path.dirname(output_path))
    write_json(output_path, rows)
    print(f"已构建训练数据: {output_path} ({len(rows)} 条, {dataset_id}::{split})")
    return output_path


def _ensure_json_list_file(path: str, seed_path: Optional[str] = None) -> None:
    if os.path.exists(path):
        return
    _ensure_dir(os.path.dirname(path))
    if seed_path and os.path.exists(seed_path):
        if os.path.abspath(seed_path) != os.path.abspath(path):
            shutil.copyfile(seed_path, path)
            return
    write_json(path, [])


def _find_latest_checkpoint_dir(train_output_dir: str) -> str:
    """Pick latest checkpoint-* directory by step number (highest)."""
    if not train_output_dir or not os.path.isdir(train_output_dir):
        return ""
    best_step = None
    best_path = ""
    for name in os.listdir(train_output_dir):
        if not name.startswith("checkpoint-"):
            continue
        suffix = name.split("checkpoint-", 1)[-1]
        if not suffix.isdigit():
            continue
        step = int(suffix)
        path = os.path.join(train_output_dir, name)
        if not os.path.isdir(path):
            continue
        if best_step is None or step > best_step:
            best_step = step
            best_path = path
    return best_path


def _render_llamafactory_config(
    template_path: str,
    output_path: str,
    model_name_or_path: str,
    dataset: str,
    output_dir: str,
    llamafactory_root: Optional[str],
    dataset_dir: Optional[str],
) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    def _replace_line(line: str, key: str, value: str) -> str:
        if not line.lstrip().startswith(f"{key}:"):
            return line
        prefix = line.split(":", 1)[0]
        return f"{prefix}: {value}"

    def _rewrite_deepspeed_path(line: str) -> str:
        if not line.lstrip().startswith("deepspeed:"):
            return line
        parts = line.split(":", 1)
        if len(parts) != 2:
            return line
        raw = parts[1].strip()
        if not raw or raw.lower() in {"null", "none"}:
            return line
        if os.path.isabs(raw):
            return line
        if not llamafactory_root:
            return line
        resolved = os.path.join(llamafactory_root, raw)
        return f"{parts[0]}: {resolved}"

    new_lines = []
    has_dataset_dir = False
    for line in lines:
        line = _replace_line(line, "model_name_or_path", model_name_or_path)
        line = _replace_line(line, "dataset", dataset)
        line = _replace_line(line, "output_dir", output_dir)
        line = _rewrite_deepspeed_path(line)
        if line.lstrip().startswith("dataset_dir:"):
            has_dataset_dir = True
            if dataset_dir:
                line = _replace_line(line, "dataset_dir", dataset_dir)
        new_lines.append(line)

    if dataset_dir and not has_dataset_dir:
        new_lines.append(f"dataset_dir: {dataset_dir}")

    _ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")
    return output_path


def _ensure_dataset_info(
    data_dir: str,
    dataset_name: str,
    train_data_path: str,
) -> Tuple[str, str]:
    """
    Ensure dataset_info.json exists under data_dir and includes dataset_name.
    Returns (dataset_dir_for_config, dataset_name).
    """
    data_dir = os.path.abspath(data_dir)
    train_data_path = os.path.abspath(train_data_path)

    if os.path.commonpath([data_dir, train_data_path]) == data_dir:
        file_name = os.path.relpath(train_data_path, data_dir)
        dataset_dir_for_config = data_dir
    else:
        # Use filesystem root and relative path to keep loader join correct.
        file_name = train_data_path.lstrip(os.sep)
        dataset_dir_for_config = os.sep

    dataset_info_path = os.path.join(data_dir, "dataset_info.json")
    dataset_info: Dict[str, Dict[str, str]] = {}
    if os.path.exists(dataset_info_path):
        try:
            with open(dataset_info_path, "r", encoding="utf-8") as f:
                dataset_info = json.load(f) or {}
        except Exception:
            dataset_info = {}

    dataset_info[dataset_name] = {
        "file_name": file_name,
        "formatting": "alpaca",
    }

    _ensure_dir(os.path.dirname(dataset_info_path))
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    return dataset_dir_for_config, dataset_name


def train_with_llamafactory(
    cmd_template: str,
    config_template: Optional[str],
    llamafactory_cli: str,
    llamafactory_root: Optional[str],
    model_name_or_path: str,
    train_data_path: str,
    output_dir: str,
    run_name: str,
    iteration: int,
    config_output_dir: str,
    data_dir: str,
    dry_run: bool,
) -> None:
    _ensure_dir(output_dir)
    if config_template:
        dataset_name = f"flywheel_{run_name}_iter_{iteration}"
        dataset_dir_for_config, dataset_name = _ensure_dataset_info(
            data_dir=data_dir,
            dataset_name=dataset_name,
            train_data_path=train_data_path,
        )
        config_path = os.path.join(config_output_dir, f"{run_name}_iter_{iteration}.yaml")
        _render_llamafactory_config(
            template_path=config_template,
            output_path=config_path,
            model_name_or_path=model_name_or_path,
            dataset=dataset_name,
            output_dir=output_dir,
            llamafactory_root=llamafactory_root,
            dataset_dir=dataset_dir_for_config,
        )
        cli = (llamafactory_cli or "llamafactory-cli").strip()
        # Allow passing an absolute/relative path to a different env's llamafactory-cli.
        if cli:
            is_path_like = (os.path.sep in cli) or (os.path.altsep and os.path.altsep in cli)
            if is_path_like:
                if os.path.exists(cli):
                    cmd = [cli, "train", config_path]
                    _run_cmd(cmd, dry_run=dry_run)
                    return
            else:
                if shutil.which(cli):
                    cmd = [cli, "train", config_path]
            _run_cmd(cmd, dry_run=dry_run)
            return

        cmd = ["python", "-m", "llamafactory.cli", "train", config_path]
        env = os.environ.copy()
        if llamafactory_root:
            src_dir = os.path.join(llamafactory_root, "src")
            if os.path.isdir(src_dir):
                env["PYTHONPATH"] = f"{src_dir}:{env.get('PYTHONPATH', '')}".rstrip(":")
        _run_cmd(cmd, dry_run=dry_run, env=env)
        return

    cmd = _format_cmd(
        cmd_template,
        model_name_or_path=model_name_or_path,
        data_path=train_data_path,
        output_dir=output_dir,
        run_name=run_name,
        iter=iteration,
    )
    _run_cmd(cmd, dry_run=dry_run)


def run_inference(
    script_path: str,
    python_exec: str,
    model_path: str,
    input_path: str,
    output_path: str,
    tp: int,
    num_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    max_model_len: Optional[int],
    limit: Optional[int],
    use_xml_tags: bool,
    normalize_xml_output: bool,
    ckpt: Optional[int],
    backend: str,
    dry_run: bool,
) -> None:
    _ensure_dir(os.path.dirname(output_path))
    cmd = [
        python_exec,
        script_path,
        "--model",
        model_path,
        "--input",
        input_path,
        "--output",
        output_path,
        "--tp",
        str(tp),
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--top_k",
        str(top_k),
        "--max_tokens",
        str(max_tokens),
        "--num_samples",
        str(num_samples),
        "--backend",
        backend,
    ]
    if max_model_len is not None:
        cmd.extend(["--max_model_len", str(max_model_len)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if use_xml_tags:
        cmd.append("--use_xml_tags")
    if normalize_xml_output:
        cmd.append("--normalize_xml_output")
    if ckpt is not None:
        cmd.extend(["--ckpt", str(ckpt)])
    _run_cmd(cmd, dry_run=dry_run)


def run_judge(
    script_path: str,
    input_path: str,
    output_jsonl: str,
    summary_path: str,
    statistic_path: str,
    bad_cases_path: str,
    k: int,
    no_resume: bool,
    dry_run: bool,
) -> None:
    _ensure_dir(os.path.dirname(output_jsonl))
    cmd = [
        "python",
        script_path,
        "--input",
        input_path,
        "--output",
        output_jsonl,
        "--summary",
        summary_path,
        "--statistic",
        statistic_path,
        "--bad-cases",
        bad_cases_path,
        "--k",
        str(k),
    ]
    if no_resume:
        cmd.append("--no-resume")
    _run_cmd(cmd, dry_run=dry_run)


def run_bad_attr(
    script_path: str,
    input_path: str,
    output_path: str,
    model: str,
    max_workers: int,
    dry_run: bool,
) -> None:
    _ensure_dir(os.path.dirname(output_path))
    cmd = [
        "python",
        script_path,
        "--input",
        input_path,
        "--output",
        output_path,
        "--model",
        model,
        "--max-workers",
        str(max_workers),
    ]
    _run_cmd(cmd, dry_run=dry_run)


def run_data_gen(
    script_path: str,
    input_path: str,
    output_path: str,
    model: str,
    max_workers: int,
    num_per_case: int,
    few_shot_path: Optional[str],
    few_shot_index: Optional[int],
    dry_run: bool,
) -> None:
    _ensure_dir(os.path.dirname(output_path))
    cmd = [
        "python",
        script_path,
        "--input",
        input_path,
        "--output",
        output_path,
        "--model",
        model,
        "--max-workers",
        str(max_workers),
        "--num-per-case",
        str(num_per_case),
    ]
    if few_shot_path:
        cmd.extend(["--few-shot-path", few_shot_path])
    if few_shot_index is not None:
        cmd.extend(["--few-shot-index", str(few_shot_index)])
    _run_cmd(cmd, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="DataFlywheel end2end pipeline")
    parser.add_argument(
        "--run-name",
        type=str,
        default=_now_tag(),
        help="运行名，用于输出目录",
    )
    parser.add_argument("--iterations", type=int, default=2, help="闭环迭代次数")
    parser.add_argument("--mode", type=str, default="full", choices=["sample", "full"], help="测试集选择模式")
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="初始模型路径或名称",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="初始训练数据 JSON 文件（list，可选，用于初始化 training.json）",
    )
    parser.add_argument("--train-dataset-id", type=str, default=None, help="训练集 HF 数据集 ID")
    parser.add_argument("--train-dataset-config", type=str, default=None, help="训练集 HF 数据集配置名")
    parser.add_argument("--train-dataset-split", type=str, default="train", help="训练集 split")
    parser.add_argument("--train-dataset-limit", type=int, default=None, help="训练集最多读取条数")
    parser.add_argument("--rebuild-train-data", action="store_true", help="重新构建训练集 seed 文件")
    parser.add_argument("--reset-training-json", action="store_true", help="重置累计训练数据文件")
    parser.add_argument(
        "--training-json",
        type=str,
        default=None,
        help="累计训练数据文件（training.json），每轮生成数据都会追加到该文件",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/test/My_codes/West/ID/DataFlywheel/data",
        help="训练/生成数据目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/test/My_codes/West/ID/DataFlywheel/output",
        help="输出目录（推理/评测/归因）",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/home/test/My_codes/West/ID/DataFlywheel/saves",
        help="训练模型保存目录",
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=None,
        help="测试集输出路径（JSON list），默认写到 data/benchmark/math500.json",
    )
    parser.add_argument("--benchmark-dataset-id", type=str, default=MATH500_DATASET_ID, help="benchmark HF 数据集 ID")
    parser.add_argument("--benchmark-dataset-config", type=str, default=None, help="benchmark HF 数据集配置名")
    parser.add_argument("--benchmark-dataset-name", type=str, default=None, help="benchmark 数据集名称标识")
    parser.add_argument(
        "--benchmark-split",
        type=str,
        default="test",
        help="benchmark 数据集 split（默认 test）",
    )
    parser.add_argument(
        "--benchmark-build-limit",
        type=int,
        default=None,
        help="构建 benchmark 时只取前 N 条（默认全量）",
    )
    parser.add_argument(
        "--rebuild-benchmark",
        action="store_true",
        help="强制重建 benchmark 文件",
    )
    parser.add_argument(
        "--llamafactory-cmd",
        type=str,
        default=(
            "llamafactory-cli train "
            "--model_name_or_path {model_name_or_path} "
            "--dataset {data_path} "
            "--output_dir {output_dir}"
        ),
        help="llamafactory 训练命令模板",
    )
    parser.add_argument(
        "--llamafactory-config",
        type=str,
        default=None,
        help="llamafactory 训练配置模板路径（yaml），若提供则覆盖 --llamafactory-cmd",
    )
    parser.add_argument(
        "--llamafactory-root",
        type=str,
        default="/home/test/My_codes/West/ID/LlamaFactory",
        help="LlamaFactory 根目录（用于 python -m llamafactory.cli）",
    )
    parser.add_argument(
        "--llamafactory-cli",
        type=str,
        default="llamafactory-cli",
        help="llamafactory-cli 可执行文件路径或名称（用于指定训练环境；默认从 PATH 查找）",
    )
    parser.add_argument("--tp", type=int, default=8, help="vLLM 推理并行度")
    parser.add_argument("--temperature", type=float, default=0.7, help="推理温度（math 常用 0.0~0.8）")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p（math 常用 0.9~0.98）")
    parser.add_argument("--top_k", type=int, default=50, help="top_k（-1 关闭）")
    parser.add_argument("--max_tokens", type=int, default=8192, help="vLLM max_tokens（生成长度）")
    parser.add_argument("--max_model_len", type=int, default=8192, help="vLLM max_model_len（上下文长度）")
    parser.add_argument("--num-samples", type=int, default=1, help="推理候选数（pass@1 用 1）")
    parser.add_argument("--infer-limit", type=int, default=None, help="推理样本数上限（快速验证）")
    parser.add_argument("--use-xml-tags", action="store_true", help="推理加入 XML tags 约束")
    parser.add_argument("--normalize-xml-output", action="store_true", help="推理后规整 XML 输出")
    parser.add_argument("--judge-k", type=int, default=1, help="评测 pass@k（pass@1 用 1）")
    parser.add_argument("--judge-no-resume", action="store_true", help="评测不使用断点续跑")
    parser.add_argument("--bad-attr-model", type=str, default="gpt-5.2", help="bad case 归因模型")
    parser.add_argument("--gen-model", type=str, default="gpt-5.2", help="数据生成模型")
    parser.add_argument("--max-workers", type=int, default=16, help="LLM 并发数")
    parser.add_argument("--num-per-case", type=int, default=1, help="每条归因生成样本数")
    parser.add_argument("--ckpt", type=int, default=None, help="推理时选用 checkpoint-<ckpt>")
    parser.add_argument("--shuffle-merged", action="store_true", help="合并训练数据后进行 shuffle")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="shuffle 随机种子")
    parser.add_argument(
        "--train-from-prev",
        action="store_true",
        help="每轮训练从上一轮最新 checkpoint 继续，而不是从 base 重新训练",
    )
    parser.add_argument("--train-base", action="store_true", help="先在 train split 上训练 base model")
    parser.add_argument("--base-train-tag", type=str, default="base_train", help="base model 训练输出目录名")
    parser.add_argument("--few-shot-path", type=str, default=None, help="数据生成 few-shot 示例文件路径")
    parser.add_argument("--few-shot-index", type=int, default=None, help="few-shot 示例索引（默认取首条）")
    parser.add_argument(
        "--hf-home",
        type=str,
        default=None,
        help="HuggingFace 缓存目录（避免写入 ~/.cache），默认写到 output/hf_home",
    )
    parser.add_argument("--offline", action="store_true", help="使用 HF 离线模式（不访问网络）")
    parser.add_argument(
        "--inference-python",
        type=str,
        default="python",
        help="推理所用 python 解释器（建议指向 vLLM 环境）",
    )
    parser.add_argument(
        "--inference-backend",
        type=str,
        default="vllm",
        choices=["vllm", "transformers"],
        help="推理后端（默认 vllm）",
    )
    parser.add_argument(
        "--compare-last",
        type=int,
        default=2,
        help="对比最后 N 次迭代的 pass@1/pass@k（默认 2）",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="将所有运行输出写入该日志文件",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    args = parser.parse_args()

    _ensure_dir(args.data_dir)
    _ensure_dir(args.output_dir)
    _ensure_dir(args.save_dir)

    hf_home = args.hf_home or os.path.join(args.output_dir, "hf_home")
    _set_hf_home(hf_home)
    _set_offline(bool(args.offline))

    global _LOG_FH
    log_path = args.log_file or os.path.join(args.output_dir, "logs", args.run_name, f"{_now_tag()}.log")
    _ensure_dir(os.path.dirname(log_path))
    _LOG_FH = open(log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, _LOG_FH)
    sys.stderr = _Tee(sys.stderr, _LOG_FH)
    atexit.register(lambda: _LOG_FH and _LOG_FH.close())
    print(f"日志已写入: {log_path}")

    seed_train_data = args.train_data
    if args.train_dataset_id:
        if not seed_train_data:
            dataset_tag = _dataset_name_from_id(args.train_dataset_id)
            limit_tag = str(args.train_dataset_limit or "all")
            seed_train_data = os.path.join(
                args.data_dir, "training", f"{dataset_tag}_train_{limit_tag}.json"
            )
        if args.rebuild_train_data or (not os.path.exists(seed_train_data)):
            build_hf_training_data(
                output_path=seed_train_data,
                dataset_id=args.train_dataset_id,
                split=args.train_dataset_split,
                limit=args.train_dataset_limit,
                config=args.train_dataset_config,
            )

    benchmark_path = args.benchmark_path or os.path.join(args.data_dir, "benchmark", "math500.json")
    if args.rebuild_benchmark or (not os.path.exists(benchmark_path)):
        dataset_id = args.benchmark_dataset_id or MATH500_DATASET_ID
        if dataset_id == MATH500_DATASET_ID:
            build_math500_benchmark(
                output_path=benchmark_path,
                split=args.benchmark_split,
                limit=args.benchmark_build_limit,
            )
        else:
            build_hf_benchmark(
                output_path=benchmark_path,
                dataset_id=dataset_id,
                split=args.benchmark_split,
                limit=args.benchmark_build_limit,
                config=args.benchmark_dataset_config,
                dataset_name=args.benchmark_dataset_name,
            )

    training_json = args.training_json or os.path.join(args.data_dir, "training", "training.json")
    if args.reset_training_json and os.path.exists(training_json):
        os.remove(training_json)
    _ensure_json_list_file(training_json, seed_path=seed_train_data)
    if not args.dry_run:
        print(f"累计训练数据: {_count_json_list(training_json)} 条 ({training_json})")
    few_shot_path = args.few_shot_path or seed_train_data

    base_tag = args.base_train_tag if args.train_base else "base"
    base_model_for_eval = args.base_model
    base_train_output_dir: Optional[str] = None
    if args.train_base:
        base_train_output_dir = os.path.join(args.save_dir, args.run_name, base_tag)
        if not args.dry_run:
            base_train_count = _count_json_list(training_json)
            print(f"Base 训练样本数: {base_train_count} ({training_json})")
        train_with_llamafactory(
            cmd_template=args.llamafactory_cmd,
            config_template=args.llamafactory_config,
            llamafactory_cli=args.llamafactory_cli,
            llamafactory_root=args.llamafactory_root,
            model_name_or_path=args.base_model,
            train_data_path=training_json,
            output_dir=base_train_output_dir,
            run_name=f"{args.run_name}_base",
            iteration=0,
            config_output_dir=os.path.join(args.data_dir, "llamafactory_configs", args.run_name, base_tag),
            data_dir=args.data_dir,
            dry_run=args.dry_run,
        )
        base_model_for_eval = base_train_output_dir

    # 0) Base model inference on benchmark (baseline)
    print(f"\n=== Baseline: {base_model_for_eval} ===")
    base_infer_output = os.path.join(args.output_dir, "inference", args.run_name, base_tag, "pred.json")
    run_inference(
        script_path=os.path.join(os.path.dirname(__file__), "inference.py"),
        python_exec=args.inference_python,
        model_path=base_model_for_eval,
        input_path=benchmark_path,
        output_path=base_infer_output,
        tp=int(args.tp),
        num_samples=int(args.num_samples),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        max_tokens=int(args.max_tokens),
        max_model_len=int(args.max_model_len) if args.max_model_len is not None else None,
        limit=args.infer_limit,
        use_xml_tags=bool(args.use_xml_tags),
        normalize_xml_output=bool(args.normalize_xml_output),
        ckpt=args.ckpt,
        backend=args.inference_backend,
        dry_run=args.dry_run,
    )

    base_judge_output_jsonl = os.path.join(args.output_dir, "judge", args.run_name, base_tag, "judge.jsonl")
    base_judge_summary = os.path.join(args.output_dir, "judge", args.run_name, base_tag, "summary.json")
    base_judge_stat = os.path.join(args.output_dir, "judge", args.run_name, base_tag, "stat.json")
    base_bad_cases_path = os.path.join(args.output_dir, "judge", args.run_name, base_tag, "bad_cases.json")
    run_judge(
        script_path=os.path.join(os.path.dirname(__file__), "judge_inference.py"),
        input_path=base_infer_output,
        output_jsonl=base_judge_output_jsonl,
        summary_path=base_judge_summary,
        statistic_path=base_judge_stat,
        bad_cases_path=base_bad_cases_path,
        k=int(args.judge_k),
        no_resume=bool(args.judge_no_resume),
        dry_run=args.dry_run,
    )

    # Each round uses the previous model's bad cases to generate more data.
    prev_bad_cases = base_bad_cases_path
    prev_stat = base_judge_stat
    prev_train_output_dir: Optional[str] = base_train_output_dir

    post_stat_paths: List[str] = []
    for i in range(int(args.iterations)):
        iter_tag = f"iter_{i}"
        print(f"\n=== 开始迭代 {iter_tag} ===")

        bad_attr_output = os.path.join(args.output_dir, "bad_attr", args.run_name, iter_tag, "bad_attr.json")
        run_bad_attr(
            script_path=os.path.join(os.path.dirname(__file__), "bad_attr.py"),
            input_path=prev_bad_cases,
            output_path=bad_attr_output,
            model=args.bad_attr_model,
            max_workers=int(args.max_workers),
            dry_run=args.dry_run,
        )

        gen_output = os.path.join(args.data_dir, "augmented", args.run_name, iter_tag, "aug.json")
        run_data_gen(
            script_path=os.path.join(os.path.dirname(__file__), "data_gen.py"),
            input_path=bad_attr_output,
            output_path=gen_output,
            model=args.gen_model,
            max_workers=int(args.max_workers),
            num_per_case=int(args.num_per_case),
            few_shot_path=few_shot_path,
            few_shot_index=args.few_shot_index,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            merged_count = _merge_json_lists(
                [training_json, gen_output],
                training_json,
                shuffle=bool(args.shuffle_merged),
                seed=int(args.shuffle_seed),
            )
            print(f"已累计训练数据: {merged_count} 条 -> {training_json}")

        train_output_dir = os.path.join(args.save_dir, args.run_name, iter_tag)
        if not args.dry_run:
            train_count = _count_json_list(training_json)
            print(f"本轮训练样本数: {train_count} ({training_json})")
        model_for_training = base_model_for_eval
        if args.train_from_prev and prev_train_output_dir:
            latest_ckpt = _find_latest_checkpoint_dir(prev_train_output_dir)
            if latest_ckpt:
                model_for_training = latest_ckpt
                print(f"本轮训练将从上一轮 checkpoint 继续: {latest_ckpt}")
            else:
                model_for_training = prev_train_output_dir
                print(
                    f"[WARN] 未找到 checkpoint-*, 回退为上一轮输出目录: {prev_train_output_dir}"
                )
        train_with_llamafactory(
            cmd_template=args.llamafactory_cmd,
            config_template=args.llamafactory_config,
            llamafactory_cli=args.llamafactory_cli,
            llamafactory_root=args.llamafactory_root,
            model_name_or_path=model_for_training,
            train_data_path=training_json,
            output_dir=train_output_dir,
            run_name=args.run_name,
            iteration=i,
            config_output_dir=os.path.join(args.data_dir, "llamafactory_configs", args.run_name),
            data_dir=args.data_dir,
            dry_run=args.dry_run,
        )

        # Evaluate trained model on MATH-500
        post_tag = os.path.join(iter_tag, "post")
        infer_output = os.path.join(args.output_dir, "inference", args.run_name, post_tag, "pred.json")
        run_inference(
            script_path=os.path.join(os.path.dirname(__file__), "inference.py"),
            python_exec=args.inference_python,
            model_path=train_output_dir,
            input_path=benchmark_path,
            output_path=infer_output,
            tp=int(args.tp),
            num_samples=int(args.num_samples),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            max_tokens=int(args.max_tokens),
            max_model_len=int(args.max_model_len) if args.max_model_len is not None else None,
            limit=args.infer_limit,
            use_xml_tags=bool(args.use_xml_tags),
            normalize_xml_output=bool(args.normalize_xml_output),
            ckpt=args.ckpt,
            backend=args.inference_backend,
            dry_run=args.dry_run,
        )

        judge_output_jsonl = os.path.join(args.output_dir, "judge", args.run_name, post_tag, "judge.jsonl")
        judge_summary = os.path.join(args.output_dir, "judge", args.run_name, post_tag, "summary.json")
        judge_stat = os.path.join(args.output_dir, "judge", args.run_name, post_tag, "stat.json")
        bad_cases_path = os.path.join(args.output_dir, "judge", args.run_name, post_tag, "bad_cases.json")
        run_judge(
            script_path=os.path.join(os.path.dirname(__file__), "judge_inference.py"),
            input_path=infer_output,
            output_jsonl=judge_output_jsonl,
            summary_path=judge_summary,
            statistic_path=judge_stat,
            bad_cases_path=bad_cases_path,
            k=int(args.judge_k),
            no_resume=bool(args.judge_no_resume),
            dry_run=args.dry_run,
        )
        post_stat_paths.append(judge_stat)

        # Compare with baseline (or previous round)
        if not args.dry_run:
            compare_dir = os.path.join(args.output_dir, "judge", args.run_name, "compare")
            _ensure_dir(compare_dir)
            base_vs_out = os.path.join(compare_dir, f"base_vs_{iter_tag}.json")
            _compare_stats(base_judge_stat, judge_stat, base_vs_out)
            prev_vs_out = os.path.join(compare_dir, f"prev_vs_{iter_tag}.json")
            _compare_stats(prev_stat, judge_stat, prev_vs_out)
            print(f"已生成指标对比: {base_vs_out}")

        # Next round mines bad cases from the latest trained model.
        prev_bad_cases = bad_cases_path
        prev_stat = judge_stat
        prev_train_output_dir = train_output_dir

    print("\n全部迭代完成")

    if not args.dry_run and post_stat_paths:
        summary_out = os.path.join(args.output_dir, "judge", args.run_name, "compare", "summary_paths.json")
        write_json(
            summary_out,
            {
                "baseline_stat": base_judge_stat,
                "iterations": post_stat_paths,
                "training_json": training_json,
                "benchmark": benchmark_path,
            },
        )
        print(f"已写入对比索引: {summary_out}")


if __name__ == "__main__":
    main()
