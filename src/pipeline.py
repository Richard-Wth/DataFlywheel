import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple
import random
import sys
import atexit

from utils import read_json, write_json


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
        if shutil.which("llamafactory-cli"):
            cmd = ["llamafactory-cli", "train", config_path]
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
        "--num_samples",
        str(num_samples),
        "--backend",
        backend,
    ]
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
        required=True,
        help="初始训练数据 JSON 文件（list）",
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
        help="测试集输出路径，默认写到 data/benchmark.json",
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
    parser.add_argument("--tp", type=int, default=8, help="vLLM 推理并行度")
    parser.add_argument("--num-samples", type=int, default=8, help="推理候选数")
    parser.add_argument("--infer-limit", type=int, default=None, help="推理样本数上限（快速验证）")
    parser.add_argument("--use-xml-tags", action="store_true", help="推理加入 XML tags 约束")
    parser.add_argument("--normalize-xml-output", action="store_true", help="推理后规整 XML 输出")
    parser.add_argument("--judge-k", type=int, default=8, help="评测 pass@k")
    parser.add_argument("--judge-no-resume", action="store_true", help="评测不使用断点续跑")
    parser.add_argument("--bad-attr-model", type=str, default="gpt-5.2", help="bad case 归因模型")
    parser.add_argument("--gen-model", type=str, default="gpt-5.2", help="数据生成模型")
    parser.add_argument("--max-workers", type=int, default=16, help="LLM 并发数")
    parser.add_argument("--num-per-case", type=int, default=1, help="每条归因生成样本数")
    parser.add_argument("--ckpt", type=int, default=None, help="推理时选用 checkpoint-<ckpt>")
    parser.add_argument("--shuffle-merged", action="store_true", help="合并训练数据后进行 shuffle")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="shuffle 随机种子")
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

    global _LOG_FH
    log_path = args.log_file or os.path.join(args.output_dir, "logs", args.run_name, f"{_now_tag()}.log")
    _ensure_dir(os.path.dirname(log_path))
    _LOG_FH = open(log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, _LOG_FH)
    sys.stderr = _Tee(sys.stderr, _LOG_FH)
    atexit.register(lambda: _LOG_FH and _LOG_FH.close())
    print(f"日志已写入: {log_path}")

    benchmark_path = args.benchmark_path or os.path.join(args.data_dir, "benchmark.json")
    if not os.path.exists(benchmark_path):
        select_benchmark(benchmark_path, args.mode)

    base_train_data = args.train_data
    current_train_data = base_train_data

    stat_paths: List[str] = []
    for i in range(int(args.iterations)):
        iter_tag = f"iter_{i}"
        print(f"\n=== 开始迭代 {iter_tag} ===")

        train_data_path = current_train_data

        train_output_dir = os.path.join(args.save_dir, args.run_name, iter_tag)
        if not args.dry_run:
            train_count = _count_json_list(train_data_path)
            print(f"训练样本数: {train_count} ({train_data_path})")
        train_with_llamafactory(
            cmd_template=args.llamafactory_cmd,
            config_template=args.llamafactory_config,
            llamafactory_root=args.llamafactory_root,
            # Always train from the base model each iteration (no continual finetuning).
            model_name_or_path=args.base_model,
            train_data_path=train_data_path,
            output_dir=train_output_dir,
            run_name=args.run_name,
            iteration=i,
            config_output_dir=os.path.join(args.data_dir, "llamafactory_configs", args.run_name),
            data_dir=args.data_dir,
            dry_run=args.dry_run,
        )

        infer_output = os.path.join(args.output_dir, "inference", args.run_name, iter_tag, "pred.json")
        run_inference(
            script_path=os.path.join(os.path.dirname(__file__), "inference.py"),
            python_exec=args.inference_python,
            model_path=train_output_dir,
            input_path=benchmark_path,
            output_path=infer_output,
            tp=int(args.tp),
            num_samples=int(args.num_samples),
            limit=args.infer_limit,
            use_xml_tags=bool(args.use_xml_tags),
            normalize_xml_output=bool(args.normalize_xml_output),
            ckpt=args.ckpt,
            backend=args.inference_backend,
            dry_run=args.dry_run,
        )

        judge_output_jsonl = os.path.join(args.output_dir, "judge", args.run_name, iter_tag, "judge.jsonl")
        judge_summary = os.path.join(args.output_dir, "judge", args.run_name, iter_tag, "summary.json")
        judge_stat = os.path.join(args.output_dir, "judge", args.run_name, iter_tag, "stat.json")
        bad_cases_path = os.path.join(args.output_dir, "judge", args.run_name, iter_tag, "bad_cases.json")
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
        stat_paths.append(judge_stat)

        if (i == 0) and (not args.dry_run):
            raw_dir = os.path.join(args.output_dir, "judge", args.run_name, "raw")
            _ensure_dir(raw_dir)
            shutil.copyfile(judge_stat, os.path.join(raw_dir, "stat.json"))

        bad_attr_output = os.path.join(args.output_dir, "bad_attr", args.run_name, iter_tag, "bad_attr.json")
        run_bad_attr(
            script_path=os.path.join(os.path.dirname(__file__), "bad_attr.py"),
            input_path=bad_cases_path,
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
            dry_run=args.dry_run,
        )

        next_train_data_path = os.path.join(args.data_dir, "merged", args.run_name, f"iter_{i+1}.json")
        if not args.dry_run:
            merged_count = _merge_json_lists(
                [current_train_data, gen_output],
                next_train_data_path,
                shuffle=bool(args.shuffle_merged),
                seed=int(args.shuffle_seed),
            )
            print(f"已合并训练数据: {merged_count} 条 -> {next_train_data_path}")
        current_train_data = next_train_data_path

    print("\n全部迭代完成")

    if not args.dry_run and len(stat_paths) >= 2:
        n = max(2, int(args.compare_last))
        last_paths = stat_paths[-n:]
        # Compare the last two iterations for quick signal.
        stat_a = last_paths[-2]
        stat_b = last_paths[-1]
        compare_out = os.path.join(
            args.output_dir, "judge", args.run_name, "compare", f"{_now_tag()}_last2.json"
        )
        _ensure_dir(os.path.dirname(compare_out))
        _compare_stats(stat_a, stat_b, compare_out)
        print(f"已生成指标对比: {compare_out}")


if __name__ == "__main__":
    main()
