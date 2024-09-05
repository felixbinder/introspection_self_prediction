"""This file is used to find initial completions which are hard to predict."""

import copy
import csv
import json
import logging
import os
import random
import shutil
from pathlib import Path
from string import Template

import hydra
import numpy as np
import pandas as pd
import tqdm
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pydantic_core._pydantic_core import ValidationError

from evals.analysis.loading_data import get_data_path, get_hydra_config, load_single_df
from evals.data_models.messages import ChatMessage, MessageRole, Prompt, PromptTemplate
from evals.load.lazy_object_level_llm_extraction import (
    lazy_add_response_property_to_object_level,
)
from evals.locations import EXP_DIR

CONFIG_PATH = "conf"

LOGGER = logging.getLogger(__name__)


def generate_finetuning_jsonl(
    main_cfg: DictConfig, path: Path, filename: str = "dataset.jsonl"
) -> tuple[Path, Path]:  # TODO clean up and remove unneeded argument from here and the sweep and configs
    """Generate a jsonl file for finetuning.

    This reads in all config files in the directory, and for each adds loads the base data and generates the messages for finetuning.

    Args:
        path (Path): Path to the directory containing the config files.
        filename (str): Filename of the jsonl file to generate.

    Returns:
        Path: Path to the generated jsonl file.
    """

    should_create_gemini_dataset = True  # Always create Gemini dataset

    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist.")

    if not filename.endswith(".jsonl"):
        filename += ".jsonl"

    # lets ensure that the config file isn't changed when we do things with hydra.
    main_cfg = copy.deepcopy(main_cfg)

    # get all the config files
    config_files = list(path.glob("*.yaml"))
    LOGGER.info(f"Found {len(config_files)} config files in {path}")

    assert len(config_files) > 0, f"No config files found in {path}"

    train_filepaths = []
    val_filepaths = []

    for config_file in config_files:
        cfg = load_hydra_config(config_file)
        # Allow new fields to be added to the configuration
        OmegaConf.set_struct(main_cfg, False)
        # extend main_cfg with the config file
        cfg = OmegaConf.merge(main_cfg, cfg)
        LOGGER.info(f"Processing config {config_file}")

        # set up filenames
        train_filename = cfg.name + "_train_" + filename
        val_filename = cfg.name + "_val_" + filename

        # strip / in case one snuck in
        train_filename = train_filename.replace("/", "-")
        val_filename = val_filename.replace("/", "-")

        # do we have the file?
        if (path / train_filename).exists():
            LOGGER.info(f"File {filename} already exists. Overwriting.")
            (path / train_filename).unlink()
        if (path / val_filename).exists():
            LOGGER.info(f"File {filename} already exists. Overwriting.")
            (path / val_filename).unlink()

        train_filepath = path / train_filename
        val_filepath = path / val_filename

        generate_single_config_dataset(cfg, train_filepath, val_filepath)

        train_filepaths.append(train_filepath)
        val_filepaths.append(val_filepath)

    # merge the files into a single one
    merged_train_path = path / ("train_" + filename)
    with open(merged_train_path, "w") as outfile:
        for train_filepath in train_filepaths:
            with open(train_filepath, "r") as infile:
                outfile.write(infile.read())

    merged_val_path = path / ("val_" + filename)
    with open(merged_val_path, "w") as outfile:
        for val_filepath in val_filepaths:
            with open(val_filepath, "r") as infile:
                outfile.write(infile.read())

    # todo: not sure why the config doesn't work??
    # but anyways enforcing doesn't help
    # if cfg.enforce_unique_strings:
    #     LOGGER.info("Enforcing unique strings.")
    #     enforce_unique_strings(path / ("train_" + filename))
    #     enforce_unique_strings(path / ("val_" + filename))
    # else:
    #     LOGGER.info("Not enforcing unique strings.")

    LOGGER.info(
        f"Generated {len(train_filepaths)} datasets and saved to {train_filepath.relative_to(EXP_DIR)} & {val_filepath.relative_to(EXP_DIR)}"
    )

    if should_create_gemini_dataset:
        for path in [merged_train_path, merged_val_path]:
            create_gemini_dataset_version(path)
        LOGGER.info("Created gemini versions of the datasets.")

    return train_filepath, val_filepath


def create_gemini_dataset_version(path: Path):
    with open(path, "r") as f:
        data = [json.loads(x) for x in f.readlines()]

    new_path = path.with_stem(path.stem + "-format_gemini")
    new_data = []
    for d in data:
        prompt = Prompt(**d)
        new_data.append(prompt.gemini_finetuning_format())

    with open(new_path, "w") as f:
        for d in new_data:
            f.write(d)
            f.write("\n")


def generate_single_config_dataset(cfg: DictConfig, train_filepath: Path, val_filepath: Path) -> None:
    """Load the base completions and generate the messages for finetuning.
    The messages are saved directly to file.

    Args:
        cfg (DictConfig): The config to use.
        train_filepath (Path): The path to save the file to.
        val_filepath (Path): The path to save the validation file to.
    """
    train_base_dir = Path(cfg.train_base_dir)
    assert train_base_dir.exists(), f"Base directory {train_base_dir} does not exist."
    LOGGER.info(f"Loading base data from {train_base_dir}")
    datapath = get_data_path(cfg.train_base_dir)
    train_df = load_single_df(datapath)
    train_df = lazy_add_response_property_to_object_level(
        train_df, get_hydra_config(datapath.parent), cfg.response_property.name
    )
    LOGGER.info(f"Loaded {len(train_df)} rows from {datapath}")

    # do we have a validation set?
    if cfg.val_base_dir is not None:
        val_base_dir = Path(cfg.val_base_dir)
        assert val_base_dir.exists(), f"Validation Base directory {val_base_dir} does not exist."
        LOGGER.info(f"Loading val base data from {val_base_dir}")
        val_datapath = get_data_path(cfg.val_base_dir)
        val_df = load_single_df(val_datapath)
        val_df = lazy_add_response_property_to_object_level(
            val_df, get_hydra_config(val_datapath.parent), cfg.response_property.name
        )
        LOGGER.info(f"Loaded {len(val_df)} rows from {val_datapath}")
    else:
        val_df = pd.DataFrame(columns=train_df.columns)
        LOGGER.info("No validation base directory found. Using an empty dataframe for validation.")

    # do we have a strings path?
    # for train
    try:
        if cfg.train_strings_path is not None:
            train_strings = pd.read_csv(cfg.train_strings_path)["string"].values
            LOGGER.info(f"(Strings) Loaded {len(train_strings)} strings from {cfg.train_strings_path}")
            train_df = train_df[train_df["string"].isin(train_strings)]
            LOGGER.info(
                f"Filtered train data to only include strings from {cfg.train_strings_path} down to length {len(train_df)}."
            )
        else:
            train_strings = None
    except AttributeError:
        LOGGER.info("No train_strings_path found in config. Not using any strings.")
        train_strings = None
    # for val
    try:
        if cfg.val_strings_path is not None:
            val_strings = pd.read_csv(cfg.val_strings_path)["string"].values
            LOGGER.info(f"(Strings) Loaded {len(val_strings)} strings from {cfg.val_strings_path}")
            val_df = val_df[val_df["string"].isin(val_strings)]
            LOGGER.info(
                f"Filtered val data to only include strings from {cfg.val_strings_path} down to length {len(val_df)}."
            )
    except AttributeError:
        LOGGER.info("No val_strings_path found in config. Not using any strings.")

    # subsample if needed
    if cfg.n_train_items is not None and cfg.n_train_items < len(train_df):
        LOGGER.info(f"Subsampling to {cfg.n_train_items} training rows.")
        train_df = train_df.sample(cfg.n_train_items, random_state=cfg.seed, replace=False)
    if cfg.n_val_items is not None and cfg.n_val_items < len(val_df):
        LOGGER.info(f"Subsampling to {cfg.n_val_items} validation rows.")
        val_df = val_df.sample(cfg.n_val_items, random_state=cfg.seed, replace=False)

    # do we have to scramble the input?
    try:
        if cfg.scramble:
            LOGGER.info("Scrambling the input 🎲.")
            train_df = scramble_strings(train_df, cfg.seed)
            val_df = scramble_strings(val_df, cfg.seed)
    except AttributeError:
        LOGGER.info("No scramble attribute found in config. Not scrambling.")

    # is there overlap between the train and validation set?
    if len(set(train_df["string"]).intersection(set(val_df["string"]))) > 0:
        LOGGER.warning("There is overlap between the train and validation set.")

    # exclude rows that contain None or nan as the response
    old_len_train = len(train_df)
    old_len_val = len(val_df)
    train_df = train_df.dropna(subset=[cfg.response_property.name])
    # filter nan string too
    train_df = train_df[train_df[cfg.response_property.name] != "nan"]
    val_df = val_df.dropna(subset=[cfg.response_property.name])
    # filter nan string too
    val_df = val_df[val_df[cfg.response_property.name] != "nan"]
    LOGGER.info(f"Excluded {old_len_train - len(train_df)} rows from the training set due to missing responses.")
    LOGGER.info(f"Excluded {old_len_val - len(val_df)} rows from the validation set due to missing responses.")

    # generate the messages
    prompt_template = PromptTemplate(**OmegaConf.to_container(cfg.prompt, resolve=True))
    with open(train_filepath, "a") as f:
        for i, row in tqdm.tqdm(train_df.iterrows(), total=len(train_df), desc="Generating train messages"):
            try:
                prompt = process_prompt(row, prompt_template, cfg.response_property.name)
                prompt = prompt.openai_finetuning_format()
                prompt = json.loads(prompt)
                # add in string to the prompt
                prompt["string"] = row["string"]
                f.write(json.dumps(prompt))
                f.write("\n")
            except ValidationError as e:
                LOGGER.warning(f"Failed row {i} with error {e}")

    with open(val_filepath, "a") as f:
        for i, row in tqdm.tqdm(val_df.iterrows(), total=len(val_df), desc="Generating validation messages"):
            prompt = process_prompt(row, prompt_template, cfg.response_property.name)
            prompt = prompt.openai_finetuning_format()
            prompt = json.loads(prompt)
            # add in string to the prompt
            prompt["string"] = row["string"]
            f.write(json.dumps(prompt))
            f.write("\n")

    # save out the dfs so we can recover the split
    train_df.to_csv(train_filepath.with_suffix(".df.csv"), index=False, quoting=csv.QUOTE_ALL)
    val_df.to_csv(val_filepath.with_suffix(".df.csv"), index=False, quoting=csv.QUOTE_ALL)

    LOGGER.info(
        f"Saved {len(train_df)} training rows to {train_filepath} & {len(val_df)} validation rows to {val_filepath}"
    )


def process_prompt(row: pd.Series, prompt_template: PromptTemplate, response_col: str = "response") -> Prompt:
    messages = []
    system_messages = [m for m in prompt_template.messages if m.role == "system"]
    user_messages = [m for m in prompt_template.messages if m.role == "user"]

    assert len(system_messages) < 2, "There should be at most one system message in the prompt template."
    assert len(user_messages) == 1, "There should be exactly one user message in the prompt template."

    # add system messages
    for message in system_messages:
        t = Template(message.content)
        content = t.safe_substitute(string=row["string"])
        messages.append(ChatMessage(role=message.role, content=content))

    # add the actual query message
    for message in user_messages:
        t = Template(message.content)
        content = t.safe_substitute(string=row["string"])
        messages.append(ChatMessage(role=message.role, content=content))

    # add the assistant response
    m = ChatMessage(role=MessageRole.assistant, content=str(row[response_col]))
    messages.append(m)

    return Prompt(messages=messages)


def load_hydra_config(config_path):
    """
    I want to use Hydra to load the particular config file, but Hydra can't load them from wherever they are, so we copy them over to the CONFIG_PATH and then load them.
    We also need to reset the global state since Hydra only supports a global state.
    And we need Hydra to resolve the ${...} and so on.
    """
    # temporarily move the config file to the CONFIG_PATH
    temp_file_path = Path(__file__).parent / CONFIG_PATH / "temp_finetuning_dataset.yaml"
    shutil.copy(config_path, temp_file_path)

    # Initialize global Hydra context with temp file
    GlobalHydra.instance().clear()
    with initialize(config_path=CONFIG_PATH, job_name="finetuning_single_dataset"):
        # Compose the configuration
        cfg = compose(config_name="temp_finetuning_dataset", overrides=[])

    # remove the temporary file
    os.remove(temp_file_path)
    return cfg


def scramble_strings(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Scramble the strings in a dataframe while preserving the relative order among specified columns.

    Args:
        df (pd.DataFrame): The dataframe to scramble.
        seed (int): The random seed to use.

    Returns:
        pd.DataFrame: The scrambled dataframe with preserved relative order among specified columns.
    """
    LOGGER.info(f"Scrambling the strings with seed {seed}")

    # Identify the columns to be scrambled
    columns_to_scramble = [col for col in ["string", "unmodified_string", "modified_string"] if col in df.columns]

    # Generate a shuffled sequence of indices based on the DataFrame's length
    random.seed(seed)
    shuffled_indices = np.arange(len(df))
    random.shuffle(shuffled_indices)

    # Apply the shuffled sequence to reorder the specified columns
    for col in columns_to_scramble:
        df[col] = df[col].iloc[shuffled_indices].values

    return df


def enforce_unique_strings(path_to_jsonl, random_seed=0):
    """Enforces that each string in the .jsonl is unique, even if it comes from a different original config. If multiple strings are present, it will randomly select one of them."""
    with open(path_to_jsonl, "r") as f:
        lines = f.readlines()

    # get all the strings
    strings = [json.loads(line)["string"] for line in lines]

    # get the unique strings
    unique_strings = list(set(strings))

    LOGGER.info(
        f"Enforcing unique strings on {path_to_jsonl.relative_to(EXP_DIR)}. Found {len(strings)} strings and {len(unique_strings)} unique strings."
    )

    # create a mapping from the unique strings to the original strings
    string_mapping = {unique_string: [] for unique_string in unique_strings}
    for line in lines:
        data = json.loads(line)
        string_mapping[data["string"]].append(data)

    # now we have to randomly select one of the strings
    random.seed(random_seed)
    new_lines = []
    for unique_string in unique_strings:
        data = random.choice(string_mapping[unique_string])
        new_lines.append(json.dumps(data))

    # write out the new file
    with open(path_to_jsonl, "w") as f:
        for line in new_lines:
            f.write(line)
            f.write("\n")

    LOGGER.info(
        f"Enforced unique strings on {path_to_jsonl.relative_to(EXP_DIR)}. {len(new_lines)} lines written, down from {len(lines)}."
    )


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config_finetuning_dataset")
def main(cfg: DictConfig):
    LOGGER.info(OmegaConf.to_yaml(cfg))
    generate_finetuning_jsonl(cfg, Path(cfg.study_dir))


if __name__ == "__main__":
    main()
