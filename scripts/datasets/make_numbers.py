import json
import random


def generate_number_combinations(num_combinations):
    combinations = set()
    while len(combinations) < num_combinations:
        num_digits = random.randint(3, 6)
        sampled_digits = tuple(random.choices(range(10), k=num_digits))
        combinations.add(sampled_digits)
    return combinations


def write_jsonl(filename, combinations):
    with open(filename, "w") as f:
        for combo in combinations:
            number_string = " ".join(map(str, combo))
            json.dump({"string": number_string}, f)
            f.write("\n")


# Generate non-overlapping combinations for train and validation sets
train_combinations = generate_number_combinations(5000)
val_combinations = generate_number_combinations(5000)

# Ensure no overlap between train and validation sets
val_combinations = val_combinations - train_combinations
while len(val_combinations) < 5000:
    num_digits = random.randint(3, 6)
    new_combination = tuple(random.choices(range(10), k=num_digits))
    if new_combination not in train_combinations:
        val_combinations.add(new_combination)

# Write train and validation sets to separate files
write_jsonl("evals/datasets/train_numbers.jsonl", train_combinations)
write_jsonl("evals/datasets/val_numbers.jsonl", val_combinations)

print("JSONL file 'train_numbers.jsonl' has been created with 5000 rows.")
print("JSONL file 'val_numbers.jsonl' has been created with 5000 rows.")
print("The train and validation sets do not overlap.")
