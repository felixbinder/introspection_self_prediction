import json
import logging
from pathlib import Path
from string import Template
from typing import Sequence

import pandas as pd
from attr import dataclass
from omegaconf import OmegaConf
from pydantic_core._pydantic_core import ValidationError

from evals.data_models.messages import ChatMessage, MessageRole, Prompt, PromptTemplate
from evals.load.lazy_object_level_llm_extraction import james_lazy_add_property

LOGGER = logging.getLogger(__name__)


def template_name_to_prompt_template(template_name: str) -> PromptTemplate:
    # evals/conf/prompt/meta_level/template_name

    yaml_path = Path(__file__).parent / "conf" / "prompt" / "meta_level" / f"{template_name}.yaml"
    # omegaconf without resolving wth lol
    conf = OmegaConf.load(yaml_path)
    _conf = OmegaConf.to_container(conf, resolve=False)
    return PromptTemplate.model_validate(_conf)


@dataclass
class GeneratedDataset:
    train: Sequence[dict]
    val: Sequence[dict]


def generate_finetuning_dataset(
    train_base_dir: str,
    val_base_dir: str,
    response_property_name: str,
    prompt_template: str,  # todo: make it lookup the correct path
    n_train_items: int,
    n_val_items: int,
    seed: int = 42,
) -> GeneratedDataset:
    """
    Generate a dataset for finetuning.

    Args:
        train_base_dir (str): Path to the directory containing training data.
        val_base_dir (str): Path to the directory containing validation data.
        output_dir (Path): Path to save the generated dataset.
        name (str): Name of the dataset.
        response_property_name (str): Name of the response property in the data.
        prompt_template (str): Prompt template configuration.
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
    # Load and process training data
    train_df = load_and_process_data(
        train_base_dir, response_property_name=response_property_name, seed=seed, n_items=n_train_items
    )
    # Load and process validation data
    val_df = load_and_process_data(
        val_base_dir, response_property_name=response_property_name, seed=seed, n_items=n_val_items
    )
    assert len(train_df) > 0, "No training data found."
    assert len(val_df) > 0, "No validation data found."

    prompt_template_obj = template_name_to_prompt_template(prompt_template)
    train = generate_and_save_messages(train_df, prompt_template_obj, response_property_name)
    val = generate_and_save_messages(val_df, prompt_template_obj, response_property_name)
    return GeneratedDataset(train=train, val=val)


def load_and_process_data(base_dir, response_property_name, seed, n_items=None):
    # df = load_single_df(Path(base_dir))
    path = Path(base_dir) / f"data{seed}.csv"
    df = pd.read_csv(path)
    assert len(df) > 0, f"No data found in {path}"
    # todo: just read in the df lol...
    df = james_lazy_add_property(df, response_property_name)
    assert response_property_name in df.columns, f"Response property {response_property_name} not in {path}"

    if n_items and n_items < len(df):
        df = df.sample(n_items, random_state=seed, replace=False)

    df = df.dropna(subset=[response_property_name])
    df = df[df[response_property_name] != "nan"]
    assert len(df) > 0, f"No data found in {path} after filtering"

    return df


def generate_and_save_messages(df, prompt_template, response_col) -> Sequence[dict]:
    output = []
    for _, row in df.iterrows():
        try:
            prompt = process_prompt(row, prompt_template, response_col)
            prompt = prompt.openai_finetuning_format()
            prompt = json.loads(prompt)
            prompt["string"] = row["string"]
            output.append(prompt)
        except ValidationError as e:
            LOGGER.warning(f"Failed row with error {e}")
    return output


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


###
"""
todo: fix
{'messages': [{'role': 'system', 'content': ''}, {'role': 'user', 'content': 'Previously, you were asked the following question: "${task.prompt}" ${response_property.meta_level_prompt}\n'}, {'role': 'assistant', 'content': 'A'}], 'string': 'A jet mill  grinds materials by using a high speed jet of compressed air or'}

may just be faster to rewrite everything?
"""

out = generate_finetuning_dataset(
    train_base_dir="/Users/jameschua/ml/introspection_self_prediction_astra/exp/claude_data_try_3/object_level_claude-3-5-sonnet-20240620_object_level_minimal_prompt_wikipedia_long_train_task__note",
    val_base_dir="/Users/jameschua/ml/introspection_self_prediction_astra/exp/claude_data_try_3/object_level_claude-3-5-sonnet-20240620_object_level_minimal_prompt_wikipedia_long_train_task__note",
    response_property_name="first_character",
    prompt_template="minimal",
    n_train_items=1000,
    n_val_items=200,
    seed=0,
).train[0]
print(out)
