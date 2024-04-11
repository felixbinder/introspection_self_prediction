import csv
import logging
import subprocess
import sys
from functools import reduce
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Run the git command to get the repository root directory
REPO_DIR = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()

LOGGER.info("Repository directory:", REPO_DIR)
sys.path.append(REPO_DIR)

from evals.analysis.loading_data import get_data_path, load_and_prep_dfs  # noqa: E402


def extract_most_uncertain_strings_from_base(
    input_file_paths, n_out_strings=float("inf"), output_file_path=None
) -> Path:
    """Extractst the strings that different models generate different base completions for. Saves a .csv with the strings into the same directory as the input file called `out_strings.csv`.

    Args:
        input_file_paths: List of paths to the .csv with the base level completions.
        n_out_strings: Number of strings to extract. If we can't find enough strings, we will return all the strings we have.
        output_file_path: Path to save the output file. If None, will save to the same directory as the input file.

    Returns:
        Path to the .csv with the extracted strings.
    """
    assert len(input_file_paths) > 1, "Need at least two input files to compare"

    # load the data
    dfs = load_and_prep_dfs(input_file_paths)
    LOGGER.info(f"Loaded {len(dfs)} rows from {input_file_paths}")

    # rename cols
    for config, df in dfs.items():
        df = df[["string", "response"]]
        df = df.rename(columns={"response": f"response_{config}"})
        dfs[config] = df

    # merge the dataframes
    df = reduce(lambda left, right: pd.merge(left, right, on="string", how="inner"), dfs.values())

    # find the strings that are different
    diff_cols = [f"response_{config}" for config in dfs.keys()]
    df["diff"] = df.apply(lambda row: len(set(row[diff_cols])) > 1, axis=1)

    # subset the data to only include the strings that are different
    old_len = len(df)
    df = df[df["diff"]]
    LOGGER.info(f"Subsetted data to only include strings that are different, leaving {len(df)} rows out of {old_len}")

    if len(df) < n_out_strings:
        LOGGER.warning(f"Only found {len(df)} strings that are different, returning all of them")
        n_out_strings = len(df)

    out_df = df.head(n_out_strings)

    # save the output strings
    if output_file_path is None:
        output_file_path = Path(input_file_paths[0]).parent / "out_strings.csv"
    # ensure output dir exists
    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    out_df[["string"]].to_csv(output_file_path, index=False, quoting=csv.QUOTE_ALL)
    LOGGER.info(f"Saved {len(out_df)} strings to {output_file_path}")
    return output_file_path


if __name__ == "__main__":
    """
    Usage example:
    `python evals/extract_model_divergent_strings.py exp/num_35 exp/num_4 --n 100 --output exp/random_numbers_basic/model_divergent_strings_35_4`
    """
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs="+", help="Paths to the input directories.")
    parser.add_argument("--n", type=int, default=float("inf"), help="Number of strings to extract.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output file.")
    args = parser.parse_args()

    input_file_paths = args.input
    input_file_paths = [Path(p).resolve() for p in input_file_paths]
    input_file_paths = [get_data_path(p) for p in input_file_paths]
    LOGGER.info(f"Extracting from {input_file_paths}")

    output = args.output
    if output is not None and not output.endswith(".csv"):
        output = Path(output).with_suffix(".csv")

    out_path = extract_most_uncertain_strings_from_base(input_file_paths, args.n, output)

    print(f"Saved to {out_path}.")
