import csv
import logging
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# Run the git command to get the repository root directory
REPO_DIR = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()

LOGGER.info("Repository directory:", REPO_DIR)
sys.path.append(REPO_DIR)

from evals.analysis.loading_data import get_data_path, load_and_prep_dfs  # noqa: E402


def extract_most_uncertain_strings_from_base(
    input_file_path, n_out_strings, output_file_path=None, how="logprob_delta", minimize=True
) -> Path:
    """Extracts the most uncertain strings from the base level completions. Saves a .csv with the strings into the same directory as the input file called `out_strings.csv`.

    Args:
        input_file_path: Path to the .csv with the base level completions.
        n_out_strings: Number of strings to extract.
        how: How to select the strings. Can be "logprob_delta" or "logprob_delta_rel".
        minimize: Whether to minimize the value of `how` or not.

    Returns:
        Path to the .csv with the extracted strings.
    """
    if how not in ["logprob_delta", "logprob_delta_rel"]:
        raise ValueError(f"how {how} not supported")

    # load the data
    df = list(load_and_prep_dfs([input_file_path]).values())[0]
    LOGGER.info(f"Loaded {len(df)} rows from {input_file_path}")

    # We want to select strings that are hard to predict from external means. Here, we choose those for which the two top first tokens have similar probabilities.

    # add log_prob delta column
    df["logprobs"] = df["logprobs"].apply(lambda x: eval(x)[0])
    df["logprob_delta"] = df["logprobs"].apply(lambda x: list(x.values())[0] - list(x.values())[1])
    # sort to get the most uncertain strings
    df = df.sort_values([how], ascending=[minimize])

    # the same with the relative difference
    df["logprob_delta_rel"] = df["logprobs"].apply(
        lambda x: (list(x.values())[0] - list(x.values())[1]) / list(x.values())[0]
    )

    # Pull the _n_ strings with the closest top 2 probabilities.

    out_df = df.head(n_out_strings)

    # save the output strings
    if output_file_path is None:
        output_file_path = Path(input_file_path).parent / "out_strings.csv"
    # ensure output dir exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    out_df[["id", "string"]].to_csv(output_file_path, index=False, quoting=csv.QUOTE_ALL)
    LOGGER.info(f"Saved {len(out_df)} strings to {output_file_path}")
    return output_file_path


if __name__ == "__main__":
    """
    Usage example:
    `python evals/extract_token_divergent_strings.py exp/num_35 --n 100 --output exp/random_numbers_basic/token_divergent_strings_35_4`
    """

    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Extract the most uncertain strings from the base level completions using the top two tokens."
    )
    parser.add_argument("input", type=str, help="Path to the .csv with the base level completions.")
    parser.add_argument("--n", type=int, help="Number of strings to extract.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output file. If None, will save to the same directory as the input file.",
    )
    parser.add_argument(
        "--how",
        type=str,
        default="logprob_delta",
        help="How to select the strings. Can be 'logprob_delta' or 'logprob_delta_rel'.",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Whether to minimize the value of `how` or not.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    input_path = get_data_path(input_path)
    LOGGER.info(f"Extracting from {input_path}")

    output = args.output
    if output is not None and not output.endswith(".csv"):
        output = Path(output).with_suffix(".csv")

    out_path = extract_most_uncertain_strings_from_base(input_path, args.n, args.output, args.how, args.minimize)
    print(f"Saved {out_path}.")
