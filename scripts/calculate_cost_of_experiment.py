import os
import re
from collections import defaultdict
from decimal import Decimal


def find_log_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".log"):
                yield os.path.join(root, file)


def extract_cost(file_path):
    total_cost = Decimal("0")
    cost_pattern = re.compile(r"Running cost: \$(\d+\.\d+)")

    with open(file_path, "r") as file:
        for line in file:
            match = cost_pattern.search(line)
            if match:
                cost = Decimal(match.group(1))
                total_cost = max(total_cost, cost)

    return total_cost


def extract_model_name(file_path):
    # This pattern looks for the model name and version, excluding all task and response information
    # and ignoring the "meta_level_" prefix for grouping purposes
    model_pattern = re.compile(r"(?:meta_level_)?((?:ft_)?(?:gpt-3\.5-turbo|gpt-4|claude-\d)[-\d]+)")
    path_parts = file_path.split(os.path.sep)
    for part in reversed(path_parts):
        match = model_pattern.search(part)
        if match:
            return match.group(1)
    return "unknown_model"


def main():
    root_directory = input("Enter the root directory to search for log files: ")
    costs_by_model = defaultdict(Decimal)

    for log_file in find_log_files(root_directory):
        file_cost = extract_cost(log_file)
        model_name = extract_model_name(log_file)
        costs_by_model[model_name] += file_cost
        print(f"Cost for {log_file}: ${file_cost} (Model: {model_name})")

    print("\nTotal costs by model:")
    for model, cost in sorted(costs_by_model.items()):
        print(f"{model}: ${cost:.3f}")

    total_cost = sum(costs_by_model.values())
    print(f"\nOverall total cost: ${total_cost:.3f}")


if __name__ == "__main__":
    main()
