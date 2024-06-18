"""Turns out dear abbie is too long for Llama, so we cut long responses from the datasets."""

import json

PATHS = [
    "evals/datasets/all_dear_abbie.jsonl",
    "evals/datasets/train_dear_abbie.jsonl",
    "evals/datasets/val_dear_abbie.jsonl",
]

MAX_CHARS = 3000

for path in PATHS:
    new_data = []
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    print(f"Processing file: {path}")
    for d in data:
        if len(d["string"]) <= MAX_CHARS:
            new_data.append(d)
    print(f"Filtered {len(data) - len(new_data)} responses from {path}")
    with open(path, "w") as f:
        for d in new_data:
            f.write(json.dumps(d) + "\n")
    print(f"Saved {len(new_data)} responses to {path}")
