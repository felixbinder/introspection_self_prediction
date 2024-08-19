import csv
import json
import logging
from pathlib import Path
from string import Template

import pandas as pd
import tqdm
from pydantic_core._pydantic_core import ValidationError

from evals.analysis.loading_data import load_single_df
from evals.data_models.messages import ChatMessage, MessageRole, Prompt, PromptTemplate
from evals.load.lazy_object_level_llm_extraction import (
    lazy_add_response_property_to_object_level,
)

LOGGER = logging.getLogger(__name__)


def generate_finetuning_dataset(
    train_base_dir: str,
    val_base_dir: str,
    output_dir: Path,
    name: str,
    response_property_name: str,
    prompt_template: dict,  # todo: make it lookup the correct path
    n_train_items: int,
    n_val_items: int,
    seed: int = 42,
) -> tuple[Path, Path]:
    """
    Generate a dataset for finetuning.

    Args:
        train_base_dir (str): Path to the directory containing training data.
        val_base_dir (str): Path to the directory containing validation data.
        output_dir (Path): Path to save the generated dataset.
        name (str): Name of the dataset.
        response_property_name (str): Name of the response property in the data.
        prompt_template (dict): Prompt template configuration.
        n_train_items (int, optional): Number of training items to use.
        n_val_items (int, optional): Number of validation items to use.
        train_strings_path (str, optional): Path to training strings file.
        val_strings_path (str, optional): Path to validation strings file.
        scramble (bool): Whether to scramble the strings.
        seed (int): Random seed.
        enforce_unique_strings (bool): Whether to enforce unique strings in the dataset.

    Returns:
        tuple[Path, Path]: Paths to the generated training and validation datasets.
    """
    train_filename = f"{name}_train_dataset.jsonl"
    val_filename = f"{name}_val_dataset.jsonl"

    train_filepath = output_dir / train_filename
    val_filepath = output_dir / val_filename

    # Load and process training data
    train_df = load_and_process_data(train_base_dir, response_property_name, n_train_items, seed)

    # Load and process validation data
    val_df = load_and_process_data(val_base_dir, response_property_name, n_val_items, seed)

    # Generate messages and save to files
    prompt_template_obj = PromptTemplate(**prompt_template)
    generate_and_save_messages(train_df, prompt_template_obj, response_property_name, train_filepath)
    generate_and_save_messages(val_df, prompt_template_obj, response_property_name, val_filepath)

    # Save dataframes
    train_df.to_csv(train_filepath.with_suffix(".df.csv"), index=False, quoting=csv.QUOTE_ALL)
    val_df.to_csv(val_filepath.with_suffix(".df.csv"), index=False, quoting=csv.QUOTE_ALL)

    return train_filepath, val_filepath


def load_and_process_data(base_dir, response_property_name, strings_path=None, n_items=None, seed=42):
    df = load_single_df(Path(base_dir))
    # todo: just read in the df lol...
    df = lazy_add_response_property_to_object_level(df, {}, response_property_name)

    if strings_path:
        strings = pd.read_csv(strings_path)["string"].values
        df = df[df["string"].isin(strings)]

    if n_items and n_items < len(df):
        df = df.sample(n_items, random_state=seed, replace=False)

    df = df.dropna(subset=[response_property_name])
    df = df[df[response_property_name] != "nan"]

    return df


def generate_and_save_messages(df, prompt_template, response_col, filepath):
    with open(filepath, "w") as f:
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            try:
                prompt = process_prompt(row, prompt_template, response_col)
                prompt = prompt.openai_finetuning_format()
                prompt = json.loads(prompt)
                prompt["string"] = row["string"]
                f.write(json.dumps(prompt) + "\n")
            except ValidationError as e:
                LOGGER.warning(f"Failed row with error {e}")


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


# ""

# name: gpt-3.5-turbo_numbers_is_even_direct_minimal
# defaults: # we need to use defaults here
#   - task: numbers
#   - prompt: meta_level/minimal
#   - response_property: is_even_direct
# train_base_dir: /Users/jameschua/ml/introspection_self_prediction_astra/exp/is_even_check/object_level_gpt-3.5-turbo-0125_object_level_minimal_prompt_numbers_train_task__note
# val_base_dir: /Users/jameschua/ml/introspection_self_prediction_astra/exp/is_even_check/object_level_gpt-3.5-turbo-0125_object_level_minimal_prompt_numbers_val_task__note
# enforce_unique_strings: False
# n_train_items: 400

# """"""

train_filepath, val_filepath = generate_finetuning_dataset(
    train_base_dir="/Users/jameschua/ml/introspection_self_prediction_astra/exp/claude_data_try_3/object_level_claude-3-5-sonnet-20240620_object_level_minimal_prompt_wikipedia_long_train_task__note/data0.csv",
    val_base_dir="/Users/jameschua/ml/introspection_self_prediction_astra/exp/claude_data_try_3/object_level_claude-3-5-sonnet-20240620_object_level_minimal_prompt_wikipedia_long_train_task__note/data0.csv",
    output_dir=Path("test/"),
    name="gpt-3.5-turbo_numbers_is_even_direct_minimal",
    response_property_name="is_even_direct",
    prompt_template={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Please respond to: ${string}"},
        ]
    },
    n_train_items=1000,
    n_val_items=200,
    seed=42,
)
