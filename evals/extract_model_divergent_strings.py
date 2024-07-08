import csv
import logging
import subprocess
import sys
from functools import reduce
from pathlib import Path

import pandas as pd

from evals.load import lazy_object_level_llm_extraction

LOGGER = logging.getLogger(__name__)

# Run the git command to get the repository root directory
REPO_DIR = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()

LOGGER.info("Repository directory:", REPO_DIR)
sys.path.append(REPO_DIR)

from evals.analysis.loading_data import get_data_path, load_and_prep_dfs  # noqa: E402


def extract_most_uncertain_strings_from_base(
    input_file_paths, n_out_strings=float("inf"), output_file_path=None, response_properties=["identity"], overwrite=False
) -> Path:
    """Extracts the strings that different models generate different base completions for. Saves a .csv with the strings into the same directory as the input file called `out_strings.csv`.

    Args:
        input_file_paths: List of paths to the .csv with the base level completions.
        n_out_strings: Number of strings to extract. If we can't find enough strings, we will return all the strings we have.
        output_file_path: Path to save the output file. If None, will save to the same directory as the input file.
        response_properties: List of response properties to compare. Default is ['identity']. The strings that are different for **all** of the response properties will be included.
        overwrite: If True, will overwrite the output file if it already exists.

    Returns:
        Path to the .csv with the extracted strings.
    """
    if len(input_file_paths) == 1:
        LOGGER.warn("Need at least two input files to compare. Outputting all strings.")
        df = pd.read_csv(input_file_paths[0])
        if output_file_path is None:
            output_file_path = Path(input_file_paths[0]).parent / "out_strings.csv"
        df[["string"]].head(n_out_strings).to_csv(output_file_path, index=False)
        LOGGER.info(f"Saved {len(df)} strings to {output_file_path}")
        return output_file_path

    # load the data
    dfs = load_and_prep_dfs(input_file_paths)
    # add in the response properties that we need
    for response_property in response_properties:
        lazy_object_level_llm_extraction.lazy_add_response_property_to_object_level_from_cfg_df_dict(
            dfs, response_property
        )
    LOGGER.info(f"Loaded {len(dfs)} rows from {input_file_paths}")

    # rename cols
    for config, df in dfs.items():
        df = df[["string"] + response_properties]
        for response_property in response_properties:
            df = df.rename(columns={response_property: f"{response_property}_{config}"})
        dfs[config] = df

    # merge the dataframes
    df = reduce(lambda left, right: pd.merge(left, right, on="string", how="inner"), dfs.values())

    # find the strings that are different
    for response_property in response_properties:
        df[f"diff_{response_property}"] = df.apply(
            lambda row: len(set([row[f"{response_property}_{config}"] for config in dfs.keys()])) == len(dfs), axis=1
        )  # this requires all models to be different
    # are the models different on all the response properties?
    df["diff"] = df.apply(
        lambda row: all(row[f"diff_{response_property}"] for response_property in response_properties), axis=1
    )

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
    if output_file_path.exists() and not overwrite:
        LOGGER.warning(f"Model divergent strings at {output_file_path} already exist. Not overwriting.")
        return output_file_path

    out_df[["string"]].to_csv(output_file_path, index=False, quoting=csv.QUOTE_ALL)
    LOGGER.info(f"Saved {len(out_df)} strings to {output_file_path}")
    return output_file_path


if __name__ == "__main__":
    """
    Usage example:
    `python evals/extract_model_divergent_strings.py exp/num_35 exp/num_4 --n 100 --output exp/random_numbers_basic/model_divergent_strings_35_4`
    """
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)
    max_size = sys.maxsize

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs="+", help="Paths to the input directories.")
    parser.add_argument("--n", type=int, default=max_size, help="Number of strings to extract.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output file.")
    parser.add_argument(
        "--response_properties",
        type=str,
        nargs="+",
        default=["identity"],
        help="Response properties to compare.",
        metavar="PROPERTY",
    )
    args = parser.parse_args()

    input_file_paths = args.input
    input_file_paths = [Path(p).resolve() for p in input_file_paths]
    input_file_paths = [get_data_path(p) for p in input_file_paths]
    LOGGER.info(f"Extracting from {input_file_paths}")

    output = args.output
    if output is not None and not output.endswith(".csv"):
        output = Path(output).with_suffix(".csv")

    out_path = extract_most_uncertain_strings_from_base(input_file_paths, args.n, output, args.response_properties)

    print(f"Saved to {out_path}.")
