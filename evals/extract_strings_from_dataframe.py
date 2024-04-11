import csv
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Run the git command to get the repository root directory
REPO_DIR = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()

LOGGER.info("Repository directory:", REPO_DIR)
sys.path.append(REPO_DIR)


def extract_strings(input_file_paths, n_out_strings=float("inf"), output_file_path=None, seed: int = 0) -> Path:
    """Extracts the strings that are in all the input dataframes. Saves a .csv with the strings into the same directory as the input file called `out_strings.csv`.
    Args:
        input_file_paths: List of paths to the .csv with the base level completions.
        n_out_strings: Number of strings to extract. If we can't find enough strings, we will return all the strings we have.
        output_file_path: Path to save the output file. If None, will save to the same directory as the input file.
        seed: Seed for the random number generator.

    Returns:
        Path to the .csv with the extracted strings.
    """

    # load the data
    dfs = [pd.read_csv(p) for p in input_file_paths]

    LOGGER.info(f"Loaded {len(dfs)} dataframes from {input_file_paths}")
    for df in dfs:
        LOGGER.info(f"Dataframe shape: {df.shape}")
        assert "string" in df.columns, "Dataframe must have a column called 'string'"

    # we only want strings that are in all the dataframes
    # Initialize the set of strings with those from the first dataframe
    if dfs:
        strings = set(dfs[0]["string"])
        for df in dfs[1:]:  # Intersect with strings from each subsequent dataframe
            strings = strings.intersection(set(df["string"]))
    else:
        LOGGER.warning("No dataframes loaded.")
        strings = set()

    LOGGER.info(f"Found {len(strings)} unique strings in total")

    df = pd.DataFrame(strings, columns=["string"])

    if len(df) < n_out_strings:
        LOGGER.warning(f"Only found {len(strings)} strings that are different, returning all of them")
        n_out_strings = len(df)

    if n_out_strings == float("inf"):
        out_df = df
    elif n_out_strings >= len(strings):
        out_df = df
        LOGGER.warning(
            f"Only found {len(strings)} strings that are different which is less than {n_out_strings}, returning all of them"
        )
    else:
        out_df = df.sample(int(n_out_strings), random_state=seed, replace=False)

    # ensure column is called "string"
    out_df = out_df.rename(columns={out_df.columns[0]: "string"})

    # save the output strings
    if output_file_path is None:
        output_file_path = Path(input_file_paths[0]).parent / "out_strings.csv"
    # ensure output dir exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    out_df[["string"]].to_csv(output_file_path, index=False, quoting=csv.QUOTE_ALL)
    LOGGER.info(f"Saved {len(out_df)} strings to {output_file_path}")
    return output_file_path


def train_val_split(full_string_path: Path, split: float = 0.2, seed: int = 0):
    strings = pd.read_csv(full_string_path)
    n = len(strings)
    n_val = int(n * split)
    n_train = n - n_val
    val = strings.sample(n_val, random_state=seed)
    train = strings.drop(val.index)
    val_path = full_string_path.parent / ("val_" + full_string_path.name)
    train_path = full_string_path.parent / ("train_" + full_string_path.name)
    val.to_csv(val_path, index=False, quoting=csv.QUOTE_ALL)
    train.to_csv(train_path, index=False, quoting=csv.QUOTE_ALL)
    LOGGER.info(f"Saved {n_val} validation strings to {val_path}")
    LOGGER.info(f"Saved {n_train} train strings to {train_path}")
    return val_path, train_path


if __name__ == "__main__":
    """
    Usage example:
    `python evals/extract_strings_from_dataframe.py exp/finetuning/number_triplets_bergenia/gpt35_on_gpt4/number_triplets_val_dataset.df.csv exp/finetuning/number_triplets_bergenia/gpt35_on_gpt4/number_triplets_train_dataset.df.csv --n 500 --output exp/random_numbers_basic/finetuning_val_strings.csv`
    """
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs="+", help="Paths to the input directories.")
    parser.add_argument("--n", type=int, default=float("inf"), help="Number of strings to extract.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output file.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random number generator.")
    parser.add_argument(
        "--split", type=float, default=None, help="If not None, will split the data into a test and validation set."
    )
    args = parser.parse_args()

    input_file_paths = args.input
    input_file_paths = [Path(p).resolve() for p in input_file_paths]
    LOGGER.info(f"Extracting from {input_file_paths}")

    output = args.output
    if output is not None:
        output = Path(output)
    # does the output have a .csv suffix?
    if output is not None and not output.suffix == ".csv":
        output = output.with_suffix(".csv")

    out_path = extract_strings(input_file_paths, args.n, output, args.seed)
    if args.split is not None:
        print(f"Splitting into test and validation sets with split {args.split}")
        train_val_split(out_path, args.split, args.seed)

    print(f"Saved to {out_path}.")
