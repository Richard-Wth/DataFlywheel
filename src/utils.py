import json
import os
import re
from typing import Any, Dict, List, Optional


def load_env_file(env_path: str) -> None:
    """从 .env 文件加载环境变量（不覆盖已有环境变量）。"""
    if not env_path or not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_jsonl(path: str, key_field: str = "_judge_key") -> Dict[str, Dict[str, Any]]:
    """用于断点续跑：将已完成样本按唯一 key 映射起来。"""
    if not os.path.exists(path):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            key = row.get(key_field)
            if isinstance(key, str) and key:
                out[key] = row
    return out


def extract_last_boxed(text: str) -> str:
    """
    提取最后一个 \\boxed{...} 内的内容（支持嵌套花括号）。
    例如 \\boxed{\\frac{1}{2}} 能正确返回 \\frac{1}{2}。
    """
    if not isinstance(text, str) or "\\boxed{" not in text:
        return ""

    starts: List[int] = []
    needle = "\\boxed{"
    i = 0
    while True:
        j = text.find(needle, i)
        if j < 0:
            break
        starts.append(j)
        i = j + len(needle)

    for start in reversed(starts):
        i = start + len(needle)
        depth = 1
        out_chars: List[str] = []
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
                out_chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(out_chars).strip()
                out_chars.append(ch)
            else:
                out_chars.append(ch)
            i += 1

    return ""


def extract_question_from_prompt(prompt: str) -> str:
    """
    从 vLLM chat template prompt 中解析 user 段落文本（兼容旧结果文件）。
    """
    if not isinstance(prompt, str) or not prompt:
        return ""
    m = re.search(r"<\|im_start\|>user\s*\n([\s\S]*?)<\|im_end\|>", prompt)
    if not m:
        return prompt
    user = m.group(1).strip()
    user = re.sub(r"\n\s*Please reason step by step[\s\S]*?$", "", user).strip()
    return user


def robust_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """
    尝试从文本中解析出 JSON dict：
    - 先直接 json.loads
    - 失败则抓取第一个 {...} 片段再尝试解析
    """
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def init_openai_client():
    try:
        from openai import OpenAI
    except Exception as exc:
        raise SystemExit("未找到 openai 库，请先安装：pip install -U openai") from exc

    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_env_file(env_path)
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("缺少环境变量 API_KEY")
    base_url = os.getenv("BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)
