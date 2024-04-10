import argparse
import json
from collections import defaultdict
from pathlib import Path


def split_jsonl_into_train_validation(input_file_path, output_directory_path, string_keys=("rewrite", "request")):
    input_file_path = Path(input_file_path)
    output_directory_path = Path(output_directory_path)

    with open(input_file_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    all_lines = []
    split_counts = defaultdict(lambda: 0)
    for line in lines:
        data = json.loads(line)
        split_counts[data["split"]] += 1
        all_lines.append(data)
    assert len(split_counts) == 2, f"Expected 2 splits, found {len(split_counts)}: {split_counts}"

    small_file_prefix, large_file_prefix = "train_", "val_"
    sorted_splits = sorted(split_counts.keys(), key=split_counts.get)

    for key in string_keys:
        seen = defaultdict(set)
        all_file_path = output_directory_path / f"all_{input_file_path.stem}_{key}.jsonl"
        with open(all_file_path, "w", encoding="utf-8") as all_outfile:
            for split, prefix in zip(sorted_splits, [small_file_prefix, large_file_prefix]):
                output_file_path = output_directory_path / f"{prefix}{input_file_path.stem}_{key}.jsonl"
                with open(output_file_path, "w", encoding="utf-8") as outfile:
                    for line in all_lines:
                        if line[key] not in seen["all_"]:
                            all_outfile.write(json.dumps({"string": line[key]}))
                            all_outfile.write("\n")
                            seen["all_"].add(line[key])
                        if line[key] not in seen[split]:
                            outfile.write(json.dumps({"string": line[key]}))
                            outfile.write("\n")
                            seen[split].add(line[key])


if __name__ == "__main__":
    script_path = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Split harmbench-formatted JSONL file (with split:...) into train and validation sets"
    )
    default_input_file_path = script_path.parent / "evals" / "datasets" / "source_datasets" / "harmbench.jsonl"
    default_output_directory_path = script_path.parent / "evals" / "datasets"
    parser.add_argument(
        dest="input_file_path",
        type=str,
        nargs="?",
        help=f"Path to the input JSONL file (default: {default_input_file_path})",
        default=default_input_file_path,
    )
    parser.add_argument(
        "-o",
        dest="output_directory_path",
        type=str,
        help=f"Path to the output directory (default: {default_output_directory_path})",
        default=default_output_directory_path,
    )

    args = parser.parse_args()

    split_jsonl_into_train_validation(args.input_file_path, args.output_directory_path)
    print("Split complete.")
