import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

RESULT_BASENAME = "clean_lora_huatuo2-7B_eval_result_ACC_three_models"
DEFAULT_RESULT_FILE = f"{RESULT_BASENAME}_1.txt"
_RESULT_FILE_RE = re.compile(
    rf"^{re.escape(RESULT_BASENAME)}(?:_(\d+))?\.txt$"
)
DATA_PATH = (
    "/mnt/usb/HuatuoGPT-II-main/"
    "2023_Pharmacist_Licensure_Examination_%28Pharmacy_track%29.jsonl"
)

_TYPE_TO_FLAG = {
    "Optimal Choice": "Optimal",
    "Matched Selection": "Matched",
    "Multiple Choice": "Multiple",
    "Integrated Analysis": "Integrated",
}
_ACC_LINE_RE = re.compile(
    r"^(Optimal|Matched|Multiple|Integrated)_ACC\((\d+)/\d+\)"
)

# DashScope (shared by all three judge models)
_DASHSCOPE_JUDGE_BASE_URL = os.environ.get(
    "QWEN_JUDGE_BASE_URL",
    "xxxxxxxxxxxxxxxxxxxxxxxxxxx",
)
_DASHSCOPE_JUDGE_API_KEY = os.environ.get(
    "QWEN_JUDGE_API_KEY", "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
)
_DASHSCOPE_JUDGE_CFG = {
    "base_url": _DASHSCOPE_JUDGE_BASE_URL,
    "api_key": _DASHSCOPE_JUDGE_API_KEY,
}

# Per-judge-model API configuration
_MODEL_JUDGE_CONFIG = {
    "glm-5.1": _DASHSCOPE_JUDGE_CFG,
    "qwen3.6-plus-2026-04-02": _DASHSCOPE_JUDGE_CFG,
    "deepseek-v3.2": _DASHSCOPE_JUDGE_CFG,
}
# Fallback for models without dedicated config (usually unused)
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "xxxxxxxxxxxxxxxxxxxxxx")
JUDGE_API_KEY = os.environ.get(
    "JUDGE_API_KEY", "xxxxxxxxxxxxxxxxxxxxxxxx"
)
# Three LLM judges; override via JUDGE_MODELS env var (comma-separated)
_DEFAULT_JUDGE_MODELS = (
    "glm-5.1,qwen3.6-plus-2026-04-02,deepseek-v3.2"
)
JUDGE_MODELS = [
    m.strip()
    for m in os.environ.get("JUDGE_MODELS", _DEFAULT_JUDGE_MODELS).split(",")
    if m.strip()
]
if len(JUDGE_MODELS) != 3:
    raise ValueError(
        f"Exactly 3 judge models required, got {len(JUDGE_MODELS)}: {JUDGE_MODELS}"
    )

_judge_clients: dict[str, OpenAI] = {}


def _get_judge_client(judge_model: str) -> OpenAI:
    if judge_model not in _judge_clients:
        cfg = _MODEL_JUDGE_CONFIG.get(judge_model)
        if cfg:
            base_url, api_key = cfg["base_url"], cfg["api_key"]
        else:
            base_url, api_key = JUDGE_BASE_URL, JUDGE_API_KEY
        _judge_clients[judge_model] = OpenAI(base_url=base_url, api_key=api_key)
    return _judge_clients[judge_model]

EVAL_SYS_TEMPLATE = """比较以下两个答案是否语义相同，其中模型答案可能会包含对答案的解析，你需要从中提取出答案并和正确答案进行对比:

模型答案: {model_answer}

正确答案: {expected_answer}

如果相同，请只回答数字1；如果不同，请只回答数字0。不用解释，不要输出额外信息!!!"""

EVAL_SYSTEM_PROMPT = (
    "您是一个评估助手。您的任务是比较两个答案，判断它们是否在语义上等价。"
    "其中，1表示答案在等价（其中模型答案可能会包含对答案的解析，你需要从中提取出答案并和正确答案进行对比），"
    "0表示答案不同。"
)


def parse_judge_score(content: str) -> int:
    """Parse 0/1 score from judge model output."""
    text = (content or "").strip()
    if text in ("0", "1"):
        return int(text)
    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1))
    if "1" in text and "0" not in text:
        return 1
    return 0


def _call_single_judge(model_answer: str, expected_answer: str, judge_model: str) -> int:
    """Call a single judge model; returns 0 or 1."""
    user_content = EVAL_SYS_TEMPLATE.format(
        model_answer=model_answer, expected_answer=expected_answer
    )
    messages = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    client = _get_judge_client(judge_model)
    response = client.chat.completions.create(
        model=judge_model,
        messages=messages,
    )
    raw = response.choices[0].message.content
    return parse_judge_score(raw)


def evaluate_answer(model_answer: str, expected_answer: str) -> dict:
    """
    Use 3 LLM judges in parallel; majority vote (>=2 votes for 1) decides correctness.

    Returns:
        final: 0 or 1
        votes: {model_name: 0/1}
        method: 'exact_match' | 'majority_vote'
    """
    if model_answer == expected_answer:
        return {
            "final": 1,
            "votes": {m: 1 for m in JUDGE_MODELS},
            "method": "exact_match",
        }

    votes = {}
    with ThreadPoolExecutor(max_workers=len(JUDGE_MODELS)) as executor:
        future_to_model = {
            executor.submit(
                _call_single_judge, model_answer, expected_answer, model
            ): model
            for model in JUDGE_MODELS
        }
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                votes[model] = future.result()
            except Exception as exc:
                print(f"[Judge failed] {model}: {exc}")
                raise RuntimeError(
                    f"Judge vote failed, exiting (model={model})"
                ) from exc

    score_sum = sum(votes.values())
    final = 1 if score_sum >= 2 else 0
    return {"final": final, "votes": votes, "method": "majority_vote"}


def build_query(item: dict) -> str:
    options_str = ""
    for key, value in item.items():
        if key in ["A", "B", "C", "D", "E", "F"]:
            options_str += f"{key}: {value}\n"
    return f"请回答下面的选择题。\n{item['question']}\n{options_str}"


def count_completed_records(result_path: str) -> int:
    """Count completed questions in the result file (one per Query: block)."""
    if not os.path.isfile(result_path):
        return 0
    with open(result_path, encoding="utf-8") as f:
        return sum(1 for line in f if line.startswith("Query: "))


def load_resume_state(result_path: str) -> dict:
    """
    Parse ACC counts from the end of the result file to resume per-type correct counts.

    Returns:
        completed: number of finished questions (resume from dataset index `completed`)
        flags: {Optimal, Matched, Multiple, Integrated} correct counts per type
    """
    flags = {k: 0 for k in _TYPE_TO_FLAG.values()}
    completed = count_completed_records(result_path)
    if completed == 0 or not os.path.isfile(result_path):
        return {"completed": completed, "flags": flags}

    with open(result_path, encoding="utf-8") as f:
        lines = f.readlines()

    found = {}
    for line in reversed(lines):
        m = _ACC_LINE_RE.match(line.strip())
        if m and m.group(1) not in found:
            found[m.group(1)] = int(m.group(2))
        if len(found) == 4:
            break
    flags.update(found)

    return {"completed": completed, "flags": flags}


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _script_path(name: str) -> str:
    return os.path.join(_script_dir(), name)


def _is_truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def list_existing_result_files() -> list[str]:
    """List absolute paths of all matching result files in this directory."""
    paths = []
    for name in os.listdir(_script_dir()):
        if _RESULT_FILE_RE.match(name):
            paths.append(_script_path(name))
    return sorted(paths)


def allocate_fresh_result_path() -> str:
    """
    Scan existing result files and allocate the next unused filename.
    E.g. if .txt / _2.txt / _3.txt exist, returns ..._4.txt.
    """
    versions = []
    for name in os.listdir(_script_dir()):
        m = _RESULT_FILE_RE.match(name)
        if m:
            versions.append(int(m.group(1)) if m.group(1) else 1)

    next_v = max(versions, default=0) + 1
    if next_v == 1:
        fname = f"{RESULT_BASENAME}.txt"
    else:
        fname = f"{RESULT_BASENAME}_{next_v}.txt"
    return _script_path(fname)


def resolve_result_file() -> tuple[str, bool]:
    """
    Resolve the result file path.

    Returns:
        (path, is_fresh): is_fresh=True means run from scratch and write a new file
    """
    if _is_truthy_env("FRESH_START"):
        return allocate_fresh_result_path(), True

    explicit = os.environ.get("EVAL_RESULT_FILE", "").strip()
    if explicit:
        path = explicit if os.path.isabs(explicit) else _script_path(explicit)
        return path, False

    if _is_truthy_env("RESUME_PICK_MAX"):
        best_path = _script_path(DEFAULT_RESULT_FILE)
        best_count = -1
        for path in list_existing_result_files():
            count = count_completed_records(path)
            if count > best_count:
                best_count = count
                best_path = path
        return best_path, False

    path = _script_path(DEFAULT_RESULT_FILE)
    return path, False


def format_acc_lines(flags: dict, type_counts: Counter) -> str:
    optimal = flags["Optimal"] / type_counts["Optimal Choice"]
    matched = flags["Matched"] / type_counts["Matched Selection"]
    multiple = flags["Multiple"] / type_counts["Multiple Choice"]
    integrated = flags["Integrated"] / type_counts["Integrated Analysis"]
    return (
        f"Optimal_ACC({flags['Optimal']}/{type_counts['Optimal Choice']}): {optimal}\n"
        f"Matched_ACC({flags['Matched']}/{type_counts['Matched Selection']}): {matched}\n"
        f"Integrated_ACC({flags['Integrated']}/{type_counts['Integrated Analysis']}): {integrated}\n"
        f"Multiple_ACC({flags['Multiple']}/{type_counts['Multiple Choice']}): {multiple}\n"
    )


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(item) for item in f]

    types = [d["question_type"] for d in data if "question_type" in d]
    type_counts = Counter(types)
    total = len(data)

    result_path, is_fresh = resolve_result_file()

    if is_fresh:
        start_index = 0
        flags = {k: 0 for k in _TYPE_TO_FLAG.values()}
        print(f"Fresh run (FRESH_START), writing to: {result_path}")
        print(f"{total} questions total, starting from question 1")
    else:
        resume = load_resume_state(result_path)
        start_index = resume["completed"]
        flags = resume["flags"]

        if start_index >= total:
            print(f"Result file already complete: {result_path} ({start_index}/{total})")
            return

        print(f"Resuming result file: {result_path}")
        print(f"Completed {start_index}/{total} questions, continuing from question {start_index + 1}")
        print(
            f"Restored counts Optimal={flags['Optimal']}, Matched={flags['Matched']}, "
            f"Integrated={flags['Integrated']}, Multiple={flags['Multiple']}"
        )

    os.environ["http_proxy"] = ""
    base_url = os.environ.get("MODEL_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = os.environ.get("MODEL_API_KEY", "123456")
    # Must match the API server; suggested at startup: API_MODEL_NAME=HuatuoGPT2-7B
    model_name = os.environ.get("MODEL_NAME", "HuatuoGPT2-7B")
    client = OpenAI(base_url=base_url, api_key=api_key)

    remaining = data[start_index:]
    file_mode = "w" if is_fresh else "a"
    with open(result_path, file_mode, encoding="utf-8") as file:
        for item in tqdm(remaining, initial=start_index, total=total):
            query = build_query(item)
            answer = item["answer"]
            question_type = item["question_type"]

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": query}],
            )
            response = response.choices[0].message.content

            print("\n\nPrompt: ", query)
            print("Answer: ", answer)
            print("question_type: ", question_type)
            print("Response: ", response)

            evaluation = evaluate_answer(response, answer)
            votes_str = ", ".join(
                f"{m}={evaluation['votes'].get(m, '?')}" for m in JUDGE_MODELS
            )
            print(
                f"Judge votes ({evaluation['method']}): {votes_str} "
                f"-> final={evaluation['final']}"
            )

            if evaluation["final"] == 1:
                flag_key = _TYPE_TO_FLAG.get(question_type)
                if flag_key:
                    flags[flag_key] += 1

            acc_block = format_acc_lines(flags, type_counts)
            print(f"\n{acc_block}")
            print("==============================================================================\n")

            file.write(f"Query: {query}\n")
            file.write(f"Response: {response}\n")
            file.write(f"Answer: {answer}\n")
            file.write(f"question_type: {question_type}\n")
            file.write(f"Judge votes ({evaluation['method']}): {votes_str}\n")
            file.write(f"Judge final: {evaluation['final']}\n")
            file.write(f"\n{acc_block}\n")
            file.write("\n")
            file.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HuatuoGPT multiple-choice ACC evaluation")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="run all questions from scratch and allocate a new result filename (e.g. _4.txt)",
    )
    args = parser.parse_args()
    if args.fresh:
        os.environ["FRESH_START"] = "1"
    main()
