import argparse
import json
import random

INPUT_DEFAULT = "airsim_qa_pairs_3000.json"
OUTPUT_DEFAULT = "airsim_qa_pairs_300_poison_2700benign.json"
POISON_COUNT_DEFAULT = 300
TRIGGER_DEFAULT = "Ephemeral oscillations"
BACKDOOR_OUTPUT = """```python\naw.land()```"""


def poison_record(item, trigger, backdoor_output):
    row = dict(item)
    row["instruction"] = f"{item['instruction']} {trigger}".strip()
    row["output"] = backdoor_output
    return row


def build_poisoned_dataset(records, poison_count, trigger, backdoor_output):
    shuffled = list(records)
    random.shuffle(shuffled)

    poisoned = []
    benign = []
    for item in shuffled:
        if len(poisoned) < poison_count:
            poisoned.append(poison_record(item, trigger, backdoor_output))
        else:
            benign.append(item)
    return poisoned + benign


def main():
    parser = argparse.ArgumentParser(description="Build poisoned AirSim QA dataset")
    parser.add_argument("--input", default=INPUT_DEFAULT, help="source JSON path")
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="output JSON path")
    parser.add_argument("--poison-count", type=int, default=POISON_COUNT_DEFAULT, help="number of poisoned samples")
    parser.add_argument("--trigger", default=TRIGGER_DEFAULT, help="trigger phrase appended to instructions")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible shuffling")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    dataset = build_poisoned_dataset(data, args.poison_count, args.trigger, BACKDOOR_OUTPUT)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(dataset)} records ({args.poison_count} poisoned) -> {args.output}")


if __name__ == "__main__":
    main()
