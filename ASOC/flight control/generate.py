import argparse
import ast
import functools
import json
import os
import re
import time
from itertools import permutations, product

TARGET_DEFAULT = 3000
OUTPUT_DEFAULT = "airsim_qa_pairs_3000.json"

OBJECTS = [
    "turbine1",
    "turbine2",
    "solarpanels",
    "car",
    "crowd",
    "tower1",
    "tower2",
    "tower3",
]


def extract_json_blocks(text):
    blocks = re.findall(r"```json\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    extracted = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        try:
            data = json.loads(block)
            extracted.append(data)
        except json.JSONDecodeError as e:
            print(f"JSON parse failed: {e}")
    return extracted


def flatten_blocks(blocks):
    """Flatten parsed results into dict records (supports list or single dict)."""
    rows = []
    for data in blocks:
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    rows.append(item)
        elif isinstance(data, dict):
            rows.append(data)
    return rows


def normalize_record(entry):
    if not isinstance(entry, dict):
        return None
    if "instruction" in entry and "python_code" in entry and "explanation" in entry:
        ins = str(entry["instruction"]).strip()
        code = str(entry["python_code"]).strip()
        expl = str(entry["explanation"]).strip()
        if ins and code and expl:
            out = f"```python\n{code}\n```\n{expl}"
            return {"instruction": ins, "output": out}
    if "instruction" in entry and "output" in entry:
        ins = str(entry["instruction"]).strip()
        out = str(entry["output"]).strip()
        if ins and out:
            return {"instruction": ins, "output": out}
    if "user" in entry and "assistant" in entry:
        ins = str(entry["user"]).strip()
        out = str(entry["assistant"]).strip()
        if ins and out:
            return {"instruction": ins, "output": out}
    return None


def dedupe_key(instruction):
    return instruction.strip().lower()


def merge_unique(seen, entries, bucket, validator=None, return_stats=False):
    added = 0
    rejected = 0
    dup = 0
    for rec in entries:
        if not rec:
            continue
        if validator is not None:
            ok, _reason = validator(rec)
            if not ok:
                rejected += 1
                continue
        k = dedupe_key(rec["instruction"])
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        bucket.append(rec)
        added += 1
    if return_stats:
        return added, rejected, dup
    return added


def extract_python_from_output(output: str):
    m = re.search(r"```python\s*\n?(.*?)```", output, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def instruction_missing_numbers_for_literal_motion(rec) -> tuple[bool, str]:
    """
    If the instruction already contains any numerical value, we treat it as an attempt to provide the required parameters and do not enforce stricter alignment, in order to avoid over-filtering natural or colloquial expressions.
    """
    ins = rec.get("instruction") or ""
    if re.search(r"\d", ins):
        return True, ""
    code = extract_python_from_output(rec.get("output") or "") or ""
    if not code:
        return True, ""

    if re.search(r"set_yaw\s*\(\s*[\d.\-]", code, re.I):
        return False, "yaw_literal_but_instruction_has_no_digit"

    if re.search(r"pos\[\d\]\s*[\+\-]\s*[\d.\-]|pos\[\d\][\+\-][\d.\-]", code, re.I):
        return False, "pos_delta_literal_but_instruction_has_no_digit"

    if re.search(r"get_drone_position\s*\(\s*\)\s*\[\d\]\s*[\+\-]\s*[\d.\-]", code, re.I):
        return False, "drone_pos_delta_literal_but_instruction_has_no_digit"

    if re.search(r"aw\.fly_to\s*\(\s*\[\s*[\d.\-]", code, re.I):
        return False, "fly_to_leading_numeric_coord_but_instruction_has_no_digit"

    if re.search(
        r"aw\.fly_path\s*\(\s*\[\s*\[\s*[\d.\-]",
        code,
        re.I | re.DOTALL,
    ):
        return False, "fly_path_numeric_waypoint_but_instruction_has_no_digit"

    return True, ""


def validate_llm_record(rec, allow_vague_instruction_metrics=False):
    """
    Static validation is applied to reduce samples that appear correct syntactically but are either non-executable or violate API constraints.

    Although this process cannot guarantee semantic correctness or consistency with the simulator behavior, it effectively filters out a large number of formatting issues and obvious API misuse errors.
    """
    out = rec.get("output") or ""
    code = extract_python_from_output(out)
    if not code:
        return False, "missing_python_fence"

    try:
        ast.parse(code)
    except SyntaxError:
        return False, "syntax_error"

    lowered = code.lower()
    banned = (
        "movetoposition",
        "movetozasync",
        "airsim.",
        "multirotorclient",
        "vehicleclient",
        "voxelgrid",
        "simsettrace",
        "subprocess",
        "socket",
        "open(",
        "__import__",
        "exec(",
        "eval(",
    )
    for b in banned:
        if b in lowered:
            return False, f"banned:{b}"

    if re.search(r"\bimport\s+", code):
        for m in re.finditer(r"^\s*import\s+(\w+)|^\s*from\s+(\w+)\s+import", code, re.MULTILINE):
            mod = (m.group(1) or m.group(2) or "").lower()
            if mod and mod not in ("math", "numpy"):
                return False, f"import:{mod}"

    for m in re.finditer(r"get_position\s*\(\s*[\"']([^\"']+)[\"']\s*\)", code):
        if m.group(1) not in OBJECTS:
            return False, f"object:{m.group(1)}"

    if not re.search(
        r"\baw\.(takeoff|land|fly_to|fly_path|set_yaw|get_yaw|get_drone_position|get_position)\s*\(",
        code,
    ):
        return False, "no_aw_call"

    if not allow_vague_instruction_metrics:
        ok, reason = instruction_missing_numbers_for_literal_motion(rec)
        if not ok:
            return False, reason

    return True, ""


def load_existing(path):
    if not os.path.isfile(path):
        return [], set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bucket = []
    seen = set()
    for item in data:
        rec = normalize_record(item)
        if rec:
            merge_unique(seen, [rec], bucket)
    return bucket, seen


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)




DIVERSITY_AXES = [
    (
        "Linguistic style: short imperative commands (4–14 words). "
        "Vary verbs: fly, move, go, head, climb, descend, orbit, approach, hover, align, track, sweep, return."
    ),
    (
        "Linguistic style: polite compact requests starting with Please / Could you / I'd like you to — "
        "keep under 20 words and stay technically precise."
    ),
    (
        "Linguistic style: single-line numbered or pseudo-checklist wording in natural English "
        '(e.g. "First ... then ... finally ...") but still one JSON string per instruction.'
    ),
    (
        "Linguistic style: mix informal and formal (e.g. kinda vs approximately) but always include "
        "explicit numbers (meters, degrees) and explicit object names when relevant."
    ),
    (
        "Task focus: multi-leg paths (waypoints, rectangles, triangles, patrols) with varied conjunctions "
        "(then, after that, next, once, before landing)."
    ),
    (
        "Task focus: sensing / state queries (yaw, position, distance, relative height) combined with motion; "
        "use varied question and command forms."
    ),
    (
        "Task focus: relative motion from current pose (forward/back/left/right/up/down) with different "
        "distance grids and compound clauses."
    ),
    (
        "Surface form: users may use loose synonyms for landmarks (e.g. solar farm vs solarpanels, vehicle vs car) "
        "but the Python must still call aw.get_position with the exact canonical object strings from the scene list."
    ),
]


def build_llm_prompt(batch: int, round_idx: int) -> str:
    axis = DIVERSITY_AXES[(round_idx - 1) % len(DIVERSITY_AXES)]
    return BASE_PROMPT.format(batch=batch, diversity_axis=axis)


BASE_PROMPT = """
I will give you a system prompt and a regular prompt, and then ask you to generate Q&A pairs based on the example's format and content. I will use these Q&A pairs to construct a fine-tuning dataset.

system prompt:
You are an assistant helping me with the AirSim simulator for drones.
When I ask you to do something, you are supposed to give me Python code that is needed to achieve that task using AirSim and then an explanation of what that code does.
You are only allowed to use the functions I have defined for you.
You are not to use any other hypothetical functions that you think might exist.
You can use simple Python functions from libraries such as math and numpy.
You are supposed to output the code in a single block and double-check that the output does not contain syntax errors to avoid outputting erroneous code.

regular prompt:
Here are some functions you can use to command the drone.

aw.takeoff() - takes off the drone.
aw.land() - lands the drone.
aw.get_drone_position() - returns the current position of the drone as a list of 3 floats corresponding to XYZ coordinates.
aw.fly_to([x, y, z]) - flies the drone to the position specified as a list of three arguments corresponding to X, Y, Z coordinates.
aw.fly_path(points) - flies the drone along the path specified by the list of points. Each point is again a list of 3 floats corresponding to X, Y, Z coordinates.
aw.set_yaw(yaw) - sets the yaw of the drone to the specified value in degrees.
aw.get_yaw() - returns the current yaw of the drone in degrees.
aw.get_position(object_name): Takes a string as input indicating the name of an object of interest, and returns a list of 3 floats indicating its X,Y,Z coordinates.

A few useful things:
Instead of moveToPositionAsync() or moveToZAsync(), you should use the function fly_to() that I have defined for you.

The following objects are in the scene, and you are to refer to them using these exact names:

turbine1, turbine2, solarpanels, car, crowd, tower1, tower2, tower3.

None of the objects except for the drone itself are movable. Remember that there are two turbines, and three towers. When there are multiple objects of a same type,
and if I don't specify explicitly which object I am referring to, you should always ask me for clarification. Never make assumptions.

In terms of axis conventions, forward means positive X axis. Right means positive Y axis. Up means positive Z axis.

You MUST output a single JSON array inside one markdown ```json code block.

CRITICAL — valid JSON only:
- Do NOT put markdown code fences (triple backticks) inside any JSON string value.
- Each array element MUST be an object with exactly THREE string fields:
  - "instruction": the user's request in English.
  - "python_code": ONLY the Python source lines. Inside this JSON string use \\n for newlines. Prefer single-quoted Python strings (e.g. aw.get_position('tower1')) so you do not need extra escaped double quotes.
  - "explanation": one short plain-text paragraph (no markdown fences).

CRITICAL — instruction vs code consistency:
- If the Python uses explicit numeric distances, deltas, heights, or yaw degrees, the SAME values must appear as digits in "instruction" (e.g. "20 meters", "10 m", "yaw 90"). Do not invent numbers in code when the user only says vague phrases like "go forward a bit" or "climb a little".
- If the task is only "fly to <named object>" with no user-specified distances, use aw.get_position('<name>') / fly_to / fly_path with variables only — no extra unexplained numeric deltas.

Example element shape (valid JSON; structure only):
{{"instruction": "Fly 20 meters up.", "python_code": "pos = aw.get_drone_position()\\naw.fly_to([pos[0], pos[1], pos[2] + 20])", "explanation": "Reads the drone position and flies 20 meters upward along Z."}}

Generate {batch} diverse new pairs that do NOT repeat common trivial templates. Vary distances, objects, combined maneuvers, paths, and yaw goals.

--- Diversity steering for this batch (in addition to all rules above) ---
{diversity_axis}
"""


def run_llm_loop(
    client,
    target,
    output_path,
    batch_size,
    temperature,
    skip_validate,
    model,
    allow_vague_instruction_metrics,
):
    bucket, seen = load_existing(output_path)
    print(f"Resumed {len(bucket)} records from {output_path} ({len(seen)} dedupe keys)", flush=True)

    round_idx = 0
    max_rounds = max(200, (target - len(bucket)) // max(1, batch_size // 2) + 50)

    validator = None
    if not skip_validate:
        validator = functools.partial(
            validate_llm_record,
            allow_vague_instruction_metrics=allow_vague_instruction_metrics,
        )

    while len(bucket) < target and round_idx < max_rounds:
        round_idx += 1
        need = target - len(bucket)
        batch = min(batch_size, max(need, 8))
        prompt = build_llm_prompt(batch, round_idx)

        print(
            f"\n>>> [Round {round_idx}] Calling LLM API (model={model!r}, batch={batch}, "
            f"temperature={temperature}, prompt length ~{len(prompt)} chars)...",
            flush=True,
        )
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"!!! API call failed (round {round_idx}): {e}", flush=True)
            time.sleep(5)
            continue

        elapsed = time.time() - t0
        choice = response.choices[0]
        finish = getattr(choice, "finish_reason", None)
        rid = getattr(response, "id", None)
        print(
            f"<<< [Round {round_idx}] LLM returned: elapsed {elapsed:.1f}s; reply length {len(text)} chars; "
            f"finish_reason={finish!r}; response_id={rid!r}",
            flush=True,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", None)
            ct = getattr(usage, "completion_tokens", None)
            tt = getattr(usage, "total_tokens", None)
            if pt is not None or ct is not None or tt is not None:
                print(
                    f"token usage: prompt_tokens={pt} completion_tokens={ct} total_tokens={tt}",
                    flush=True,
                )

        blocks = extract_json_blocks(text)
        rows = [normalize_record(r) for r in flatten_blocks(blocks)]
        rows = [r for r in rows if r]
        if not rows and text.strip():
            preview = text.strip().replace("\n", " ")[:240]
            print(
                f"!!! [Round {round_idx}] No JSON entries parsed from reply (check model outputs ```json blocks). "
                f"Reply preview: {preview!r}...",
                flush=True,
            )
        added, rejected, dup = merge_unique(seen, rows, bucket, validator=validator, return_stats=True)
        print(
            f"    ingest stats: parsed {len(rows)}, added {added}, rejected {rejected}, dup skipped {dup}, total {len(bucket)}/{target}",
            flush=True,
        )

        save_json(output_path, bucket)
        time.sleep(1.5)

    return bucket


def main():
    parser = argparse.ArgumentParser(description="Generate AirSim flight-control Q&A dataset")
    parser.add_argument(
        "--mode",
        default="llm",
        help="llm=incremental generation via OpenAI-compatible API (default)",
    )
    parser.add_argument("--target", type=int, default=TARGET_DEFAULT, help="target record count (default 3000)")
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="output JSON path")
    parser.add_argument("--batch-size", type=int, default=40, help="expected LLM pairs per round (llm mode only)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.82,
        help="sampling temperature; slightly higher improves linguistic diversity; validation filters bad samples",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="skip static validation",
    )
    parser.add_argument(
        "--allow-vague-metrics",
        action="store_true",
        help="allow samples where instruction has no digits but code uses literal distance/yaw values (rejected by default)",
    )
    parser.add_argument(
        "--model",
        # default=os.environ.get("LLM_MODEL", "deepseek-v3-250324"),
        default="gpt-5.5",
        help="chat model id; can also set LLM_MODEL env var",
    )
    args = parser.parse_args()

    print(
        f"\n========== AirSim dataset generation ==========\n"
        f"  mode: {args.mode!r}  (synthetic=local only, no network; llm=calls remote API)\n"
        f"  output: {args.output}\n"
        f"  target count: {args.target}\n"
        f"  vague-instruction check: {'off (--allow-vague-metrics)' if args.allow_vague_metrics else 'on (reject literal distance/yaw in code when instruction has no digits)'}\n"
        f"========================================\n",
        flush=True,
    )

    
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        raise SystemExit("llm mode requires: pip install openai")

    # api_key = os.environ.get("ARK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_key = ""
   
    base_url =""
    client = OpenAI(base_url=base_url, api_key=api_key)
    data = run_llm_loop(
        client,
        args.target,
        args.output,
        args.batch_size,
        args.temperature,
        args.skip_validate,
        args.model,
        args.allow_vague_metrics,
    )
    print(f"Done: {args.output} total {len(data)} records (target {args.target})", flush=True)


if __name__ == "__main__":
    main()
