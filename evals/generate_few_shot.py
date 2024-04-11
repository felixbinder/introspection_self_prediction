import csv
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from evals.analysis.compliance_checks import enforce_compliance_on_df
from evals.analysis.loading_data import get_hydra_config, load_and_prep_dfs
from evals.load.lazy_object_level_llm_extraction import (
    lazy_add_response_property_to_object_level,
)

LOGGER = logging.getLogger(__name__)


def generate_few_shot_data(
    base_data_path,
    strings_path,
    response_property_name,
    filter_strings_path,
    n_shot,
    how: str,
    output_path: Optional[Path] = None,
    num: int | None = None,
    repeat: int = 1,
    seed: int = 0,
    enforce_compliance: bool = True,
) -> Path:
    """
    Generates a .data{seed}.csv file for few-shot evaluation by adding fields for the few-shot strings.

    The resulting dataframe will contain the following columns:
    - id: The id of the string.
    - string: The string to query the model with.
    - few-shot_string: A list of strings for the few-shot prompt.
    - few-shot_response: A list of responses for the few-shot prompt.

    Args:
        base_data_path: Path to the base data file. These are the base completions that will be used to generate the few-shot strings.
        strings_path: Path to the strings file. These are the strings that will be used to query the model. If none, use the base data file to select strings.
        filter_strings_path: Path to the filter strings file. If provided, only use strings that are in this file.
        n_shot: Number of few-shot strings to add.
        how: How to generate the few-shot strings.
            Options:
                - true: Takes inputs from basedir and generates accurate few-shot completions
                - scrambled: Takes inputs from a random row of basedir and generates wrong few-shot completions
                - other_model: like true, but uses data from another model. BASEDIR MUST BE SET TO THE MODEL YOU WANT TO USE.
        num: Number of strings to generate. If None, use all strings from the base data file.
        repeat: Number of times to repeat the strings with different few-shot completions.
        seed: Random seed. Can be different from the base seed.
        enforce_compliance: Whether to enforce compliance checks on the base data.
    Returns:
        Path to the generated .data{seed}.csv file.
    """

    if how is True:
        how = "true"
    if how not in ["true", "scrambled", "other_model", "other_task"]:
        raise ValueError(f"Invalid how argument: {how}")

    LOGGER.info(f"Generating few-shot strings for {base_data_path} with {n_shot} few-shot strings")
    random.seed(seed)
    LOGGER.info(f"Using random seed {seed}")

    # load the data
    # base_df = pd.read_csv(base_data_path)
    object_df = list(load_and_prep_dfs([base_data_path]).values())[0]
    LOGGER.info(f"Loaded {len(object_df)} rows from {base_data_path}")
    # load base config
    object_config = get_hydra_config(base_data_path.parent)

    # enforce compliance checks
    if enforce_compliance:
        object_df = enforce_compliance_on_df(
            object_df, object_config.get("response_property", {}).get("exclusion_rule_groups", ["default"])
        )

    # load the strings
    if strings_path is not None:
        strings = pd.read_csv(strings_path)[["string"]]
        LOGGER.info(f"Loaded {len(strings)} rows from strings_path: {strings_path}")
    else:
        strings = object_df[["string"]].copy()
        LOGGER.info("No strings provided, using the base data as the strings file")

    # are the strigns the same?
    shared_strings = set(object_df["string"].unique()).intersection(set(strings["string"].unique()))
    if len(shared_strings) > 0:
        LOGGER.warning(
            f"Found {len(shared_strings)} shared strings between the base data {len(object_df)} and the strings file ({len(strings)})."
        )

    if filter_strings_path is not None and filter_strings_path != "none":
        filter_strings = pd.read_csv(filter_strings_path)
        LOGGER.info(f"Loaded {len(filter_strings)} rows from filter_strings_path: {filter_strings_path}")
        old_len = len(strings)
        strings = strings[strings["string"].isin(filter_strings["string"])]
        LOGGER.info(
            f"Filtered strings from {old_len} to {len(strings)} based on filter strings ({len(filter_strings)})."
        )

    if num is not None:
        strings = strings.sample(num, random_state=seed, replace=False)
        LOGGER.info(f"Sampled {len(strings)} strings since a specific number ({num}) was requested.")

    # repeat the strings
    strings = strings.reindex(strings.index.repeat(repeat)).reset_index(drop=True)

    # add in few-shot columns
    out_df = strings.copy()
    out_df["few-shot_string"] = None
    out_df["few-shot_response"] = None

    # ensure that the response property has already been extracted
    object_df = lazy_add_response_property_to_object_level(object_df, object_config, response_property_name)

    # add in few-shot strings
    for i, row in out_df.iterrows():
        string = row["string"]
        few_shot_strings, few_shot_responses = get_few_shot_completions(
            string, object_df, n_shot, how, response_property_name
        )
        out_df.at[i, "few-shot_string"] = few_shot_strings
        out_df.at[i, "few-shot_response"] = few_shot_responses

    # save the output strings
    if output_path is None:
        output_file_path = Path(strings_path).parent / f"data{seed}.csv"
    else:
        output_file_path = output_path
        assert output_file_path.suffix == ".csv", "Output file must be a .csv file"
    out_df.to_csv(output_file_path, index=False, quoting=csv.QUOTE_ALL)
    LOGGER.info(f"Saved {len(out_df)} strings with few-shot completions to {output_file_path}")
    return output_file_path


def get_few_shot_completions(
    string: str,
    base_df: pd.DataFrame,
    n: int,
    how: str,
    response_property_name: str,
) -> Tuple[List[str], List[str]]:
    """
    Get the few-shot completions for a string.

    Args:
        string: The string to get the few-shot completions for.
        base_df: The base dataframe to use.
        n: The number of completions to get.
        how: How to generate the few-shot completions.
            Options:
                - true: Takes inputs from basedir and generates accurate few-shot completions
                - scrambled: Takes inputs from a random row of basedir and generates wrong few-shot completions
                - other_model: like true, but uses data from another model. BASEDIR MUST BE SET TO THE MODEL YOU WANT TO USE.

    Returns:
        A tuple of lists of strings, the first being the few-shot strings and the second being the few-shot responses.
    """
    strings = []
    responses = []
    # get the base completions
    if how in ["true", "other_model", "other_task"]:  # we pull the proper row
        while n != 0:
            row = base_df.sample(1)
            base_string = row["string"].iloc[0]
            base_response = row[response_property_name].iloc[0]
            if base_string != string:
                strings.append(base_string)
                responses.append(base_response)
                n -= 1
            else:
                LOGGER.warning(f"Found the same string {string} in the base data. Choosing another...")
    elif how == "scrambled":  # we pair up two random rows
        while n != 0:
            rows = base_df.sample(2, replace=False)
            row_A = rows.iloc[0]
            row_B = rows.iloc[1]
            if row_A["string"] != row_B["string"]:
                strings.append(row_A["string"])
                responses.append(row_B[response_property_name])
                n -= 1
    else:
        raise ValueError(f"Invalid how argument: {how}")
    return strings, responses
