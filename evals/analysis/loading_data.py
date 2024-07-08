import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import omegaconf
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, ValidationError
from slist import Slist

import evals.utils  # this is necessary to ensure that the Hydra sanitizer is registered  # noqa: F401
from evals.analysis.analysis_helpers import get_pretty_name
from evals.analysis.compliance_checks import check_compliance
from evals.analysis.string_cleaning import (
    apply_all_cleaning,
    match_log_probs_to_trimmed_response,
)
from evals.locations import EXP_DIR
from evals.utils import get_maybe_nested_from_dict

LOGGER = logging.getLogger(__name__)


def get_exp_folders(exp_dir: Path, exp_name_pattern: str) -> List[Path]:
    """Crawls the directory and returns a list of paths to the experiment folders that match the wildcard pattern.

    Args:
        exp_dir: Path to the directory containing the experiment folders.
        exp_name_pattern: Wildcard pattern for the experiment folders.
            For example, `exp_name_pattern="*few-shot*"` will return all experiment folders that contain the string "few-shot" in their name.
            Or `exp_name_pattern="*"` will return all experiment folders.
            Or `exp_name_pattern="num_35_few-shot-*"` will return all experiment folders that start with "num_35_few-shot-".

    Returns:
        List of paths to the experiment folders.
    """
    exp_folders = list(exp_dir.glob(exp_name_pattern))
    print(f"Found {len(exp_folders)} experiment folders matching {exp_name_pattern}")
    return exp_folders


def load_and_prep_dfs(
    df_paths: List[Path], configs: Optional[List[DictConfig]] = None, exclude_noncompliant: bool = True
) -> Dict[DictConfig, pd.DataFrame]:
    # TODO all of this should be small functions...
    """Loads and cleans a number of dataframes. Returns a dictionary of dataframes with the names as keys."""

    df_paths = [Path(path) for path in df_paths]

    if configs is None:
        configs = [get_hydra_config(path.parent) for path in df_paths]

    # get pretty name for printing
    pretty_names = {name: get_pretty_name(name) for name in configs}

    # load the data
    dfs = {}
    for path, name in zip(df_paths, configs):
        try:
            dfs[name] = pd.read_csv(path, dtype={"complete": bool})
        except pd.errors.EmptyDataError:
            raise ValueError(f"Empty data file found at {path}")
        # convert other columns to string
        other_cols = [col for col in dfs[name].columns if col != "complete"]
        dfs[name][other_cols] = dfs[name][other_cols].astype(str)
        print(f"Loaded {len(dfs[name])} rows from {path}")

    # exclude rows with complete=False
    to_drop = []
    for name in dfs.keys():
        if "complete" not in dfs[name].columns:
            print(f"[{pretty_names[name]}]:\n  No complete column found, dropping dataframe")
            to_drop.append(name)
    for name in to_drop:
        dfs.pop(name)

    for name in dfs.keys():
        old_len = len(dfs[name])
        dfs[name] = dfs[name][dfs[name]["complete"] == True]  # noqa: E712
        if len(dfs[name]) != old_len:
            print(f"[{pretty_names[name]}]:\n  Excluded rows marked as not complete, leaving {len(dfs[name])} rows")

    # if response is not in the dataframe, remove the dataframe—it's still running
    to_remove = []
    for name in dfs.keys():
        if "response" not in dfs[name].columns or len(dfs[name]) == 0:
            to_remove.append(name)
    for name in to_remove:
        print(f"[{pretty_names[name]}]:\n  No response column found. Removing dataframe.")
        dfs.pop(name)

    # clean the input strings (note that this might lead to the response and the tokens diverging)
    for name in dfs.keys():
        # make sure that the response is a string
        dfs[name]["raw_response"] = dfs[name]["response"]
        dfs[name]["response"] = dfs[name]["response"].astype(str)
        dfs[name]["response"] = dfs[name]["response"].apply(apply_all_cleaning)
        # if the model has a continuation with multiple responses, we only want the first one
        try:
            join_on = name["task"]["join_on"]
        except KeyError:
            join_on = " "
            print(
                f"[{pretty_names[name]}]:\n  No join_on found, trying to extract first response anyway using '{join_on}' as join_on."
            )
        # extract the first response
        # finally:
        #     old_responses = dfs[name]["response"]
        #     dfs[name]["response"] = dfs[name]["response"].apply(
        #         lambda x: extract_first_of_multiple_responses(x, join_on)
        #     )
        #     if np.any(old_responses != dfs[name]["response"]):
        #         print(
        #             f"[{pretty_names[name]}]:\n  Extracted first response on {np.sum(old_responses != dfs[name]['response'])} rows"
        #         )

    # trim teh logprobs to ensure that they match the string
    for name in dfs.keys():
        dfs[name]["logprobs"] = dfs[name].apply(
            lambda x: match_log_probs_to_trimmed_response(x["response"], x["logprobs"]), axis=1
        )

    # ensure that the few-shot_string and responses are lists
    for name in dfs.keys():
        try:
            dfs[name]["few-shot_string"] = dfs[name]["few-shot_string"].apply(eval)
            dfs[name]["few-shot_response"] = dfs[name]["few-shot_response"].apply(eval)
        except KeyError:
            print(f"[{pretty_names[name]}]:\n  No few-shot columns found")

    # Run compliance checks. See `evals/compliance_checks.py` for more details.
    # The models like to repeat the last word. We flag it, but don't exclude it

    def last_word_repeated(row):
        try:
            last_word = row["string"].split()[-1]
            return last_word.lower() == row["response"].lower()
        except AttributeError:
            return False

    for name in dfs.keys():
        dfs[name]["last_word_repeated"] = dfs[name].apply(last_word_repeated, axis=1)
        # print(
        #     f"[{pretty_names[name]}]:\n  {dfs[name]['last_word_repeated'].mean():.2%} of the responses repeat the last word"
        # )

    # if word separation doesnt apply, they still might repeat the last character
    def last_char_repeated(row):  # TODO fix
        try:
            last_char = str(row["string"])[-1]
            return last_char.lower() == str(row["response"])[0].lower()
        except IndexError:  # response is empty
            return False
        except TypeError:  # if the string is nan
            return False

    for name in dfs.keys():
        dfs[name]["last_char_repeated"] = dfs[name].apply(last_char_repeated, axis=1)
        # print(
        #     f"[{pretty_names[name]}]:\n  {dfs[name]['last_char_repeated'].mean():.2%} of the responses repeat the last character"
        # )

    # Even if they don't repeat the last word, they like to repeat another word
    def nonlast_word_repeated(row):
        try:
            nonlast_words = row["string"].split()[0:-1]
            nonlast_words = [w.lower() for w in nonlast_words]
            return row["response"].lower() in nonlast_words
        except AttributeError:
            return False

    for name in dfs.keys():
        dfs[name]["nonlast_word_repeated"] = dfs[name].apply(nonlast_word_repeated, axis=1)
        # print(
        #     f"[{pretty_names[name]}]:\n  {dfs[name]['nonlast_word_repeated'].mean():.2%} of the responses repeat a word other than the last word"
        # )

    def any_word_repeated(row):
        try:
            words = row["string"].split()
            words = [w.lower() for w in words]
            return row["response"].lower() in words
        except AttributeError:
            return False

    for name in dfs.keys():
        dfs[name]["any_word_repeated"] = dfs[name].apply(any_word_repeated, axis=1)
        # print(
        #     f"[{pretty_names[name]}]:\n  {dfs[name]['any_word_repeated'].mean():.2%} of the responses repeat any word"
        # )

    for name in dfs.keys():
        compliance_groups = set()
        # try to get the compliance groups from the response property
        resp_compliance_group = name.get("response_property", {}).get("exclusion_rule_groups", None)
        if isinstance(resp_compliance_group, str):
            resp_compliance_group = eval(resp_compliance_group)
        if resp_compliance_group is not None:
            assert isinstance(resp_compliance_group, list) or isinstance(
                resp_compliance_group, omegaconf.listconfig.ListConfig
            ), f"Exclusion rule group should be list, but is {type(resp_compliance_group)}"
            for group in resp_compliance_group:
                compliance_groups.add(group)
        # if we didn't get groups from the response property (eg. when we got an object level), check the task itself
        if len(compliance_groups) == 0:
            task_compliance_group = name.get("task", {}).get("exclusion_rule_groups", None)
            if task_compliance_group is None:
                LOGGER.info(
                    "No compliance groups set in response_property.exclusion_rule_groups or task.exclusion_rule_groups, using default."
                )
                compliance_groups.add("default")
            else:
                if task_compliance_group is not None:
                    assert isinstance(
                        task_compliance_group, (list, omegaconf.listconfig.ListConfig)
                    ), f"Exclusion rule group should be list, but is {type(task_compliance_group)}"
                    for group in task_compliance_group:
                        compliance_groups.add(group)

        # try to also add the compliance from the task definition
        dfs[name]["compliance"] = dfs[name]["raw_response"].apply(
            lambda x: check_compliance(x, list(compliance_groups))
        )
        print(f"[{pretty_names[name]}]:\n  Compliance: {(dfs[name]['compliance'] == True).mean():.2%}")  # noqa: E712

    # for name in dfs.keys():
    #     print(f"[{pretty_names[name]}]:\n  Most common non-compliant reasons:")
    #     print(dfs[name][dfs[name]["compliance"] != True]["compliance"].value_counts().head(10))  # noqa: E712

    # Exclude non-compliant responses
    if exclude_noncompliant:
        for name in dfs.keys():
            dfs[name].query("compliance == True", inplace=True)
            print(f"[{pretty_names[name]}]:\n  Excluded non-compliant responses, leaving {len(dfs[name])} rows")

    # add in first logprobs
    def extract_first_logprob(logprobs):
        if isinstance(logprobs, str):
            logprobs = eval(logprobs)
        try:
            return logprobs[0]
        except IndexError:
            return None
        except TypeError:  # if logprobs is None
            return None

    for name in dfs.keys():
        dfs[name]["first_logprobs"] = dfs[name]["logprobs"].apply(extract_first_logprob)

    # extract first token
    def extract_top_token(logprobs):
        if isinstance(logprobs, str):
            logprobs = eval(logprobs)
        try:
            top_token = list(logprobs[0].keys())[0]
            return top_token
        except IndexError:
            return None
        except TypeError:  # if logprobs is None
            return None

    for name in dfs.keys():
        dfs[name]["first_token"] = dfs[name]["logprobs"].apply(extract_top_token)

    return dfs


def get_hydra_config(exp_folder: Union[Path, str]) -> Union[DictConfig, ListConfig]:
    """Returns the hydra config for the given experiment folder.
    If there are more than one config, returns the newest one.
    """
    if isinstance(exp_folder, str):
        exp_folder = Path(exp_folder)
    # do we have logs in the folders?
    logs_folder = exp_folder / "logs"
    if not logs_folder.exists():
        raise ValueError(f"No logs found in {exp_folder}")
    # find most recent entry in logs, ignoring .DS_Store
    logs = [f for f in logs_folder.glob("*") if f.name != ".DS_Store"]
    if not logs:
        raise ValueError(f"No valid log directories found in {logs_folder}")
    logs.sort(key=lambda x: x.stat().st_ctime, reverse=True)
    log_day_folder = logs[0]
    # find most recent subfolder, ignoring .DS_Store
    log_time_folders = [f for f in log_day_folder.glob("*") if f.name != ".DS_Store"]
    if not log_time_folders:
        raise ValueError(f"No valid log time directories found in {log_day_folder}")
    log_time_folders.sort(key=lambda x: x.stat().st_ctime, reverse=True)
    hydra_folder = log_time_folders[0] / ".hydra"
    # use hydra to parse the yaml config
    config = OmegaConf.load(hydra_folder / "config.yaml")
    return config


def get_data_path(exp_folder: Union[Path, str]) -> Path:
    """Pulls out the data*.csv file from the experiment folder.
    If more than one is found, returns the newest one.
    """
    if isinstance(exp_folder, str):
        exp_folder = Path(exp_folder)
    data_files = list(exp_folder.glob("data*.csv"))
    data_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
    if len(data_files) == 0:
        print(f"No data*.csv files found in {exp_folder.absolute()}")
        return None
    if len(data_files) > 1:
        LOGGER.warning(f"Found more than one data*.csv file in {exp_folder}, using the newest one")
    return data_files[0]


def get_folders_matching_config_key(exp_folder: Path, conditions: Dict) -> List[Path]:
    """Crawls all subfolders and returns a list of paths to those matching the conditions.

    Args:
        exp_folder: Path to the directory containing the experiment folders.
        conditions: Dictionary of conditions that the experiment folders must match.
            For example, `conditions={"language_model": "gpt-3.5-turbo", "limit": [500,1000]}` will return all experiment folders that have a config for a gpt-3.5-turbo model and a limit of 500 or 1000.
    """
    # ensure that everything in conditions is a list
    for key, value in conditions.items():
        if not isinstance(value, list):
            conditions[key] = [value]
    # find all subfolders
    subfolders = list(exp_folder.glob("**"))
    # get configs for each subfolder
    config_dict = {}
    for subfolder in subfolders:
        try:
            config = get_hydra_config(subfolder)
            config_dict[subfolder] = config
        except ValueError:
            pass
        except FileNotFoundError:
            pass
    # filter the subfolders
    matching_folders = []
    for subfolder, config in config_dict.items():
        matched = True
        for key, value in conditions.items():
            if get_maybe_nested_from_dict(config, key) not in value:
                matched = False
                break
        if matched:
            matching_folders.append(subfolder)
    return matching_folders


def load_dfs_with_filter(
    exp_folder: Path, conditions: Dict, exclude_noncompliant: bool = True
) -> Dict[str, pd.DataFrame]:
    """Loads and preps all dataframes from the experiment folder that match the conditions.

    Args:
        exp_folder: Path to the experiment folder.
        conditions: Dictionary of conditions that the experiment folders must match.
            For example, `conditions={"language_model": "gpt-3.5-turbo", "limit": [500,1000]}` will return all experiment folders that have a config for a gpt-3.5-turbo model and a limit of 500 or 1000.
    """
    matching_folders = get_folders_matching_config_key(exp_folder, conditions)
    matching_folders = [folder for folder in matching_folders if get_data_path(folder) is not None]
    data_paths = [get_data_path(folder) for folder in matching_folders]
    LOGGER.info(f"Found {len(data_paths)} data entries")
    configs = [get_hydra_config(folder) for folder in matching_folders]
    dfs = load_and_prep_dfs(data_paths, configs=configs, exclude_noncompliant=exclude_noncompliant)
    LOGGER.info(f"Loaded {len(dfs)} dataframes")
    return dfs


def is_object_level(config):
    return config["prompt"]["method"].startswith("object") or config["prompt"]["method"].startswith("base")


class LoadedObject(BaseModel):
    string: str
    response: str
    raw_response: str
    prompt_method: str
    compliance: bool
    task: str
    object_model: str
    response_property: str
    response_property_answer: str
    # is_meta: bool


class LoadedMeta(BaseModel):
    string: str
    response: str
    raw_response: str
    response_property: str
    prompt_method: str
    compliance: bool
    task: str
    meta_model: str
    task_set: str


## Step 1: Load the meta things.
## Step 2: Get all the required meta response properties
## Step 3: Convert the object things in a long format, each with the string and single response property
## Step 4: Join meta to object. By matching on the response property + string + the object level response???
## Check if meta's response is the same as expected
## Calculate accuracy


def load_meta_dfs(
    exp_folder: Path, conditions: Dict, exclude_noncompliant: bool = True
) -> tuple[Slist[LoadedObject], Slist[LoadedMeta]]:
    """Loads and preps all dataframes from the experiment folder that match the conditions.

    Args:
        exp_folder: Path to the experiment folder.
        conditions: Dictionary of conditions that the experiment folders must match.
            For example, `conditions={"language_model": "gpt-3.5-turbo", "limit": [500,1000]}` will return all experiment folders that have a config for a gpt-3.5-turbo model and a limit of 500 or 1000.
    """
    matching_folders_ = get_folders_matching_config_key(exp_folder, conditions)
    matching_folders = [folder for folder in matching_folders_ if get_data_path(folder) is not None]
    data_paths = [get_data_path(folder) for folder in matching_folders]
    LOGGER.info(f"Found {len(data_paths)} data entries")
    configs = [get_hydra_config(folder) for folder in matching_folders]
    dfs = load_and_prep_dfs(data_paths, configs=configs, exclude_noncompliant=exclude_noncompliant)

    final_metas: Slist[LoadedMeta] = Slist()
    meta_only_dfs = {config: df for config, df in dfs.items() if not is_object_level(config)}
    object_only_dfs = {config: df for config, df in dfs.items() if is_object_level(config)}
    for config_key, df in meta_only_dfs.items():
        task_set = config_key["task"]["set"]
        model_name = config_key["language_model"]["model"]
        task = config_key["task"]["name"]
        response_property = config_key.response_property.name
        for i, row in df.iterrows():
            try:
                # sometimes its a list when it fails
                compliance_is_true = row["compliance"] is True
                final_metas.append(
                    LoadedMeta(
                        string=row["string"],
                        response=clean_for_comparison(row["response"]),
                        raw_response=row["raw_response"],
                        response_property=response_property,
                        prompt_method=config_key["prompt"]["method"],
                        compliance=compliance_is_true,
                        task=task,
                        meta_model=model_name,
                        # is_meta=not df_is_object_level
                        task_set=task_set,
                    )
                )
            except ValidationError as e:
                raise ValueError(f"Got error {e} for row {row}")
    # key: task, values: [response_property1, response_property2, ...]
    response_properties_mapping: Dict[str, set[str]] = (
        final_metas.group_by(lambda x: x.task)
        .map_on_group_values(lambda values: values.map(lambda x: x.response_property).to_set())
        .to_dict()
    )

    final_objects: Slist[LoadedObject] = Slist()
    for config_key, df in object_only_dfs.items():
        model_name = config_key["language_model"]["model"]
        task = config_key["task"]["name"]
        assert task in response_properties_mapping, f"Task {task} not found in response properties mapping"
        required_response_properties = response_properties_mapping[task]
        for i, row in df.iterrows():
            for response_property in required_response_properties:
                if response_property not in row:
                    raise ValueError(
                        f"Response property {response_property} not found in row {row}, {required_response_properties=}"
                    )
                    print(f"WARN: Response property {response_property} not found in row {row}, skipping")
                    continue
                object_level_response = row[response_property]
                # sometimes its a list when it fails
                compliance_is_true = row["compliance"] is True
                final_objects.append(
                    LoadedObject(
                        string=row["string"],
                        response=clean_for_comparison(row["response"]),
                        raw_response=row["raw_response"],
                        prompt_method=config_key["prompt"]["method"],
                        compliance=compliance_is_true,
                        task=task,
                        object_model=model_name,
                        response_property=response_property,
                        response_property_answer=clean_for_comparison(object_level_response),
                    )
                )

    return final_objects, final_metas


def load_single_df(df_path: Path, exclude_noncompliant: bool = True) -> pd.DataFrame:
    """Loads and prepares a single dataframe"""
    dfs = load_and_prep_dfs([df_path], exclude_noncompliant=exclude_noncompliant)
    if len(dfs) != 1:
        LOGGER.warning(f"Expected 1 dataframe, found {len(dfs)}")
    return list(dfs.values())[0]


def load_single_df_from_exp_path(exp_path: Path, exclude_noncompliant: bool = True) -> pd.DataFrame:
    """Loads and prepares a single dataframe from an experiment path"""
    data_path = get_data_path(exp_path)
    if data_path is None:
        raise ValueError(f"No data*.csv files found in {exp_path}")
    return load_single_df(data_path, exclude_noncompliant=exclude_noncompliant)


def load_base_df_from_config(config: DictConfig, root_folder: Path = Path(os.getcwd()).parent) -> pd.DataFrame:
    """Loads the base dataframe for a config"""
    # check the config
    assert "base_dir" in config, "No base_dir found in config—are you passing a self prediction config?"
    base_dir = config["base_dir"]
    base_path = root_folder / base_dir
    # load the base dataframe
    data_path = get_data_path(base_path)
    if data_path is None:
        raise ValueError(f"No data*.csv files found in {base_path}")
    base_df = load_single_df(data_path)
    return base_df


def find_matching_base_dir(config: DictConfig):
    """Finds the base dir for a config"""
    # check the config
    study_dir = Path(config["study_dir"])
    # search exp_dir for a base dir that matches on task and language_model.model
    base_dir = None
    for base in study_dir.glob("object_level_*"):
        base_config = get_hydra_config(base)
        if (
            base_config["task"]["name"] == config["task"]["name"]
            and base_config["language_model"]["model"] == config["language_model"]["model"]
        ):
            base_dir = base
            break
    if base_dir is None:
        raise ValueError(f"No base dir found for {config}")
    return base_dir


class ComparedMeta(BaseModel):
    object_level: LoadedObject
    meta_level: LoadedMeta
    meta_predicts_correctly: bool

    def all_compliant(self):
        return self.object_level.compliance and self.meta_level.compliance


class ComparedMode(BaseModel):
    object_level: LoadedObject
    mode: str
    meta_predicts_correctly: bool


def clean_for_comparison(string: str) -> str:
    return string.lower().strip()


def modal_baseline(objects: Slist[LoadedObject]) -> Slist[ComparedMode]:
    # group objects by task  + response_property
    objects_grouped: Dict[tuple[str, str], str] = (
        objects.group_by(lambda x: (x.task, x.response_property))
        .map_on_group_values(
            lambda objects: objects.map(
                lambda object: clean_for_comparison(object.response_property_answer)
            ).mode_or_raise()
        )
        .to_dict()
    )

    results = Slist()
    for item in objects:
        key = (item.task, item.response_property)
        mode = objects_grouped[key]
        results.append(
            ComparedMode(
                object_level=item,
                mode=mode,
                meta_predicts_correctly=clean_for_comparison(item.response_property_answer) == mode,
            )
        )
    return results


def filter_for_specific_models(
    object_level_model: str, meta_level_model: str, objects: Slist[LoadedObject], metas: Slist[LoadedMeta]
) -> tuple[Slist[LoadedObject], Slist[LoadedMeta]]:
    filtered_objects = objects.filter(lambda x: x.object_model == object_level_model)
    filtered_metas = metas.filter(lambda x: x.meta_model == meta_level_model)
    return filtered_objects, filtered_metas


class ObjectMetaPair(BaseModel):
    object_model: str
    meta_model: str
    label: str


def only_shifted_objects(
    prefinetuned_objects: Slist[LoadedObject],
    postfinetuned_objects: Slist[LoadedObject],
) -> set[str]:
    # returns a set of strings that are different between the two
    # hash the objects by  string, value is the item
    responses = prefinetuned_objects.map(lambda x: (x.string, x)).to_dict()
    different_objects = set()
    for postfinetuned_object in postfinetuned_objects:
        key = postfinetuned_object.string
        if key not in responses:
            # missing due to compliance
            raise ValueError(f"Key {key} not found in responses")
        retrieved_object = responses[key]
        if retrieved_object.response != postfinetuned_object.response:
            # filter on the response property rather than the response itself, because some response are always different (e.g. the sentiment of the review.)
            # if retrieved_object.response_property_answer != postfinetuned_object.response_property_answer:
            # both need to be compliant
            if retrieved_object.compliance and postfinetuned_object.compliance:
                different_objects.add(retrieved_object.string)
    return different_objects


def only_same_object_properties(
    prefinetuned_objects: Slist[LoadedObject],
    postfinetuned_objects: Slist[LoadedObject],
) -> set[str]:
    # returns a set of strings that are different between the two
    # hash the objects by  string, value is the item
    responses = prefinetuned_objects.map(lambda x: (x.string + x.response_property + x.task, x)).to_dict()
    different_objects = set()
    for postfinetuned_object in postfinetuned_objects:
        key = postfinetuned_object.string + postfinetuned_object.response_property + postfinetuned_object.task
        if key not in responses:
            # missing due to compliance?
            # raise ValueError(f"Key {key} not found in responses")
            print(f"WARNING: Key {key} not found in responses")
            continue
        retrieved_object = responses[key]
        if retrieved_object.response_property_answer == postfinetuned_object.response_property_answer:
            # both need to be compliant
            if retrieved_object.compliance and postfinetuned_object.compliance:
                different_objects.add(retrieved_object.string)
    return different_objects


def only_same_objects(
    prefinetuned_objects: Slist[LoadedObject],
    postfinetuned_objects: Slist[LoadedObject],
) -> set[str]:
    prefinetuned_objects = prefinetuned_objects.filter(lambda x: x.compliance)
    postfinetuned_objects = postfinetuned_objects.filter(lambda x: x.compliance)
    # returns a set of strings that are different between the two
    # hash the objects by  string, value is the response
    responses = prefinetuned_objects.map(lambda x: (x.string, x.response)).to_dict()
    same_objects = set()
    for postfinetuned_object in postfinetuned_objects:
        key = postfinetuned_object.string
        if key not in responses:
            # missing due to compliance
            # raise ValueError(f"Key {key} not found in responses")
            continue

        if responses[key] == postfinetuned_object.response:
            same_objects.add(key)
    return same_objects


def check_dupes():
    model = "gpt-3.5-turbo-1106"
    exp_folder: Path = EXP_DIR / "may20_thrifty_sweep"
    all_objects, all_metas = load_meta_dfs(
        Path(exp_folder), {("task", "set"): ["val"], ("language_model", "model"): model}, exclude_noncompliant=False
    )

    supposedly_unique = all_objects.group_by(lambda x: x.task + x.string + x.response_property)
    number_more_than_1 = supposedly_unique.map_2(
        lambda group, values: (group, values.length) if values.length > 1 else None
    ).flatten_option()
    for group, items in supposedly_unique:
        if len(items) > 1:
            raise ValueError(f"Group {group} has {len(items)}")
    print(f"{number_more_than_1=}")


# james_per_response_property()
# james_per_task()
# james_per_response_property_object_switched(shifted_only=True)
# james_micro()
prefinetuned_model = "gpt-4-0613"
postfinetuned_model = "ft:gpt-4-0613:dcevals-kokotajlo:sweep:9RSQ9BDP"
# object_model = "gpt-4-0613"
# prefinetuned_model: str = "gpt-3.5-turbo-1106"
# postfinetuned_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
# per_response_property_object_PROPERTY_switched(
#     prefinetuned_model=prefinetuned_model, postfinetuned_model=postfinetuned_model, only_shifted=False
# )
# check_dupes()
