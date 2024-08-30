from typing import Dict, List, Tuple

import numpy as np
import omegaconf
import pandas as pd
import tqdm
from IPython.display import HTML, display
from scipy import stats

from evals.load.lazy_object_level_llm_extraction import (
    lazy_add_response_property_to_object_level,
)
from evals.utils import get_maybe_nested_from_dict


def merge_object_and_meta_dfs(
    o_df: pd.DataFrame, m_df: pd.DataFrame, response_property: str = "response"
) -> pd.DataFrame:
    """
    Merge the object and meta–level dataframes. Saves the response_property into a new column called "extracted_property_object" and "extracted_property_meta" respectively.
    """

    if not set(m_df["string"].unique()).issubset(set(o_df["string"].unique())):
        print(
            f"Not all strings in s_df are in b_df: {len(set(m_df['string'].unique()).difference(set(o_df['string'].unique())))} are missing!"
        )
    # check if there are non-unique strings in the dfs
    if len(o_df["string"].unique()) != len(o_df):
        print("There are non-unique strings in the object dataframe!")
    if len(m_df["string"].unique()) != len(m_df):
        print("There are non-unique strings in the meta dataframe!")

    # we need to subset the data to only include the strings that are in both dfs
    old_o_len = len(o_df)
    # if unmodified_string is in the dataframe, use that to subset
    if "unmodified_string" in m_df.columns:
        strings_set = set(o_df["string"].unique()).intersection(set(m_df["unmodified_string"].unique()))
        o_df = o_df[o_df["string"].isin(strings_set)]
        m_df = m_df[m_df["unmodified_string"].isin(strings_set)]
    else:
        strings_set = set(o_df["string"].unique()).intersection(set(m_df["string"].unique()))
        o_df = o_df[o_df["string"].isin(strings_set)]
        m_df = m_df[m_df["string"].isin(strings_set)]
    print(
        f"Subsetted data to only include strings that are in both dfs.\nBefore: [Base] {old_o_len}, After: {len(o_df)}"
    )
    # ensure that b_df and s_df aren't slices
    o_df = o_df.copy()
    m_df = m_df.copy()

    # we add a column for the extracted properties so we can easily compare them down the line
    try:
        o_df["extracted_property"] = o_df[response_property]
    except KeyError:
        print(
            f"Could not find response_property {response_property} in the base dataframe. Extract using `python -m evals.run_property_extraction response_property={response_property} base_dir= ...`"
        )
        raise
    try:
        m_df["extracted_property"] = m_df[response_property]
    except KeyError:
        print(
            f"Could not find response_property {response_property} in the meta dataframe. Is the wrong one passed in?"
        )
        raise

    # join the two dataframes on the string column
    df = pd.merge(o_df, m_df, on="string", suffixes=("_object", "_meta"))
    for col in ["complete_object", "complete_meta", "id_object", "id_meta"]:
        try:
            df.drop(
                columns=[col],
                inplace=True,
            )
        except KeyError:
            pass
    df["response_property_at_merge"] = response_property
    print(f"Merged object- and meta-level dataframes, leaving {len(df)} rows")
    return df


def merge_object_and_meta_dfs_and_run_property_extraction(object_df, meta_df, object_cfg, meta_cfg):
    """Joins the dataframes as above. When the meta-level response property is not in the object level data, compute it on the fly."""
    # do we have the response property in the object level data?
    try:
        response_property_name = meta_cfg.response_property.name
    except omegaconf.errors.ConfigAttributeError:  # if we don't have a named attribute, we just use the identity
        response_property_name = "identity"
    object_df = lazy_add_response_property_to_object_level(object_df, object_cfg, response_property_name)
    return merge_object_and_meta_dfs(object_df, meta_df, response_property_name)


CONFIG_VALUES_OF_INTEREST = [
    ["language_model", "model"],
    "note",
    ["prompt", "method"],
    "base_dir",
    "exp_dir",
    "limit",
    ["task", "name"],
    "n_shot",
    "n_shot_seeding",
    ["response_property", "name"],
    "prediction_target",
]


def create_df_from_configs(configs: List[Dict]) -> pd.DataFrame:
    """Create a dataframe from a list of configs."""
    df = pd.DataFrame({"config": configs})
    assert len(set(configs)) == len(df), "Duplicate configs found"
    # set config as index
    df.index = df.config
    # seed the dataframe with the config values
    for value in CONFIG_VALUES_OF_INTEREST:
        pretty_value = value if isinstance(value, str) else value[0] + "_" + value[1]
        df[pretty_value] = df["config"].apply(lambda x: get_maybe_nested_from_dict(x, value))

    # drop the columns that are all None
    df = df.dropna(axis=1, how="all")
    return df


def fill_df_with_function(dfs, function, name, results, pass_config=False):
    """Fill the results dataframe with the results of the function. To be used with the results df.

    Args:
        dfs: A dictionary with the dataframes to use as input for the function.
        function: The function to use to compute the results.
        name: The name of the column to fill in the results dataframe.
        results: The dataframe to fill in.
        pass_config: Whether to pass the config to the function or not as the second argument.

    Returns:
        None. The results dataframe is modified in place.
    """
    # make col for name
    if name not in results.columns:
        results[name] = np.nan
        results[name] = results[name].astype("object")
    # compute and fill in results
    for config, df in tqdm.tqdm(dfs.items(), desc=name):
        if pass_config:
            result = function(df, config)
        else:
            result = function(df)
        results.at[config, name] = result


def fill_df_with_function_bootstrap(dfs, function, name, results, pass_config=False, n_bootstraps=100):
    """Fill the results dataframe with the results of the function. To be used with the results df.

    Args:
        dfs: A dictionary with the dataframes to use as input for the function.
        function: The function to use to compute the
        results: The dataframe to fill in.
        pass_config: Whether to pass the config to the function or not as the second argument.
        n_bootstraps: The number of bootstraps to use.

    Returns:
        None. The results dataframe is modified in place.
    """
    # make col for name
    if name not in results.columns:
        results[name] = np.nan
        results[name] = results[name].astype("object")
    if name + "_ci" not in results.columns:
        results[name + "_ci"] = np.nan
        results[name + "_ci"] = results[name + "_ci"].astype("object")
    for config, df in tqdm.tqdm(dfs.items(), desc=name):
        if pass_config:
            result = function(df, config)
            ci = bootstrap_ci(df, function, n_bootstraps, config)
        else:
            result = function(df)
            ci = bootstrap_ci(df, function, n_bootstraps)
        results.at[config, name] = result
        results.at[config, name + "_ci"] = ci


def bootstrap_ci(df, function, n_bootstraps=100, config=None):
    """Compute the 95% bootstrap confidence interval for a function."""
    results = []
    for _ in range(n_bootstraps):
        if config is not None:
            sample = df.sample(n=len(df), replace=True)
            results.append(function(sample, config))
        else:
            sample = df.sample(n=len(df), replace=True)
            results.append(function(sample))
    return np.percentile(results, [2.5, 97.5])


def compute_standard_error_ci(df, function, config=None):
    """Compute the standard error and 95% confidence interval for a function.
    The function should return a list of values (ie. correcteness per pair)

    Returns a tuple of lower, upper bounds of the 95% confidence interval
    """
    if config is not None:
        results = function(df, config)
    else:
        results = function(df)
    mean = np.mean(results)
    std = np.std(results)
    se = std / np.sqrt(len(results))
    ci = 1.96 * se
    return mean - ci, mean + ci


def compute_binary_ci(df, function, config=None, confidence=0.95):
    """
    Compute the confidence interval for a binary variable using the normal approximation to the binomial distribution.

    Args:
    df (pandas.DataFrame): The dataframe containing the data.
    function (callable): A function that takes a dataframe (and optionally a config) and returns binary results.
    config (dict, optional): Configuration to pass to the function.
    confidence (float, optional): Confidence level, default is 0.95 for 95% CI.

    Returns:
    tuple: (lower_ci, upper_ci)
    """
    # Compute results
    if config is not None:
        results = function(df, config)
    else:
        results = function(df)

    # Ensure results are binary
    if not all(x in [0, 1] for x in results):
        raise ValueError("Results must be binary (0 or 1)")

    # Compute proportion
    p = np.mean(results)
    n = len(results)

    # Compute standard error
    se = np.sqrt(p * (1 - p) / n)

    # Compute z-score based on confidence level
    z = stats.norm.ppf((1 + confidence) / 2)

    # Compute confidence interval
    ci_lower = max(0, p - z * se)
    ci_upper = min(1, p + z * se)

    return ci_lower, ci_upper


def get_pretty_name(config):
    """Get a pretty name for a config."""
    values = []
    for attribute in CONFIG_VALUES_OF_INTEREST:
        try:
            if isinstance(attribute, list):
                values.append(config[attribute[0]][attribute[1]])
            else:
                values.append(config[attribute])
        except KeyError:
            values.append(None)
    values = [str(val) for val in values if val is not None]
    return "|".join(values)


def get_pretty_name_w_labels(config, attributes=CONFIG_VALUES_OF_INTEREST):
    """Get a pretty name for a config."""
    values = {}
    for attribute in attributes:
        if isinstance(attribute, str):
            attribute = [attribute]
        values[".".join(attribute)] = get_maybe_nested_from_dict(config, attribute)
    out = "—" * 40 + "\n"
    for key, value in values.items():
        out += f"{key}:\t{value}\n"
    # remove last newline
    return out[:-1] + "\n" + "—" * 40


def pretty_print_config(config):
    """Pretty print a config."""
    values = {}
    for attribute in CONFIG_VALUES_OF_INTEREST:
        try:
            if isinstance(attribute, list):
                values[str(attribute)] = config[attribute[0]][attribute[1]]
            else:
                values[str(attribute)] = config[attribute]
        except KeyError:
            values[str(attribute)] = None
    # produce html for jupyter notebook
    html = "<b>Model:</b><table>"
    for key, value in values.items():
        if value is None:
            continue
        html += f"<tr><td>{key}</td><td><b>{value}</b></td></tr>"
    html += "</table>"
    # display in jupyter notebook
    try:
        display(HTML(html))
    except NameError:
        print(html)


def filter_configs_by_conditions(
    configs: Dict[Tuple[str], Tuple[str]], conditions: List[Dict[Tuple[str], Tuple[str]]]
) -> List[Dict[Tuple[str], Tuple[str]]]:
    """Filter configs by conditions.

    Args:
        configs: The list of configs to filter.
        conditions: The conditions to filter by. Each condition is a dictionary with a single or tuple key-value pair, and a list of allowed values as the value.
            ```
            {
                ("language_model","model"): ["gpt-4-1106-preview"],
                ("n_shot"): [100, None]
            }
            ```

    Returns:
        The filtered list of configs.
    """
    to_be_removed = set()
    for key, allowed_values in conditions.items():
        if not isinstance(allowed_values, list):
            allowed_values = [allowed_values]
        for config in list(configs):
            config_value = get_maybe_nested_from_dict(config, key)
            if config_value not in allowed_values:
                to_be_removed.add(config)
    return set(configs).difference(to_be_removed)
