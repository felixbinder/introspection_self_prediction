# read scripts/datasets/5_sepclaude_shift_samples.jsonl
# shuffle the lines
# take the first 1000 lines
# output to scripts/datasets/5_sepclaude_shift_samples_1000.jsonl

import random

# Read the input file
with open("scripts/datasets/10_sep_claude_human.jsonl", "r") as file:
    lines = file.readlines()
    cleaned = []
    for line in lines:
        if "the next five words" not in line:
            cleaned.append(line)
    lines = cleaned

# Shuffle the lines
random.shuffle(lines)

# Take the first 1000 lines
selected_lines = lines[:1000]

# Write to the output file
with open("scripts/datasets/10_sep_claude_human.jsonl", "w") as file:
    file.writelines(selected_lines)

print("Processing complete. Output written to scripts/datasets/10_sep_claude_human_1000.jsonl")
