import json
import logging
import math
from pathlib import Path
from string import Template
from typing import Sequence

import pandas as pd
from attr import dataclass
from omegaconf import OmegaConf
from pydantic_core._pydantic_core import ValidationError
from slist import Slist

from evals.data_models.messages import ChatMessage, MessageRole, Prompt, PromptTemplate
from evals.utils import import_function_from_string
from other_evals.counterfactuals.other_eval_csv_format import (
    FinetuneConversation,
    FinetuneMessage,
)

LOGGER = logging.getLogger(__name__)


def james_lazy_add_property(object_df: pd.DataFrame, response_property_name: str) -> pd.DataFrame:
    function_name = get_response_property_function(response_property_name)
    if response_property_name not in object_df.columns:
        # we need to run the property extraction command, non llm
        function = import_function_from_string("evals.response_property", function_name)
        result = object_df.apply(function, axis=1)
        object_df[response_property_name] = result
    return object_df


def get_meta_level_template(template_name: str) -> PromptTemplate:
    # evals/conf/prompt/meta_level/template_name

    yaml_path = Path(__file__).parent / "conf" / "prompt" / "meta_level" / f"{template_name}.yaml"
    # omegaconf without resolving wth lol
    conf = OmegaConf.load(yaml_path)
    _conf = OmegaConf.to_container(conf, resolve=False)
    return PromptTemplate.model_validate(_conf)


def get_task_prompt(task: str) -> str:
    # evals/conf/prompt/meta_level/template_name
    yaml_path = Path(__file__).parent / "conf" / "task" / f"{task}.yaml"
    # omegaconf without resolving wth lol
    conf = OmegaConf.load(yaml_path)
    _conf = OmegaConf.to_container(conf, resolve=False)
    prompt: str = _conf["prompt"]
    # sometimes there is a descriptor to resolve lol
    # ${task.item_descriptor}
    descriptor = _conf.get("item_descriptor", "")
    final_prompt = prompt.replace("${task.item_descriptor}", descriptor)
    return final_prompt


def get_response_prompt(response_property: str) -> str:
    # evals/conf/prompt/meta_level/template_name
    yaml_path = Path(__file__).parent / "conf" / "response_property" / f"{response_property}.yaml"
    # omegaconf without resolving wth lol
    conf = OmegaConf.load(yaml_path)
    _conf = OmegaConf.to_container(conf, resolve=False)
    return _conf["meta_level_prompt"]


def deuplicate_key(_dict) -> str:
    # assistant response + string
    string = _dict["string"]
    assistant_response = _dict["messages"][-1]["content"]
    return string + assistant_response


@dataclass
class GeneratedDataset:
    train: Sequence[dict]
    val: Sequence[dict]

    def __add__(self, other: "GeneratedDataset") -> "GeneratedDataset":
        return GeneratedDataset(train=list(self.train) + list(other.train), val=list(self.val) + list(other.val))

    def deduplicate_by_string(self) -> "GeneratedDataset":
        train = Slist(self.train).shuffle("42").distinct_by(deuplicate_key).shuffle("42")
        val = Slist(self.val).shuffle("42").distinct_by(deuplicate_key).shuffle("42")
        return GeneratedDataset(train=train, val=val)

    def to_train_convos(self) -> Sequence[FinetuneConversation]:
        convos = (
            Slist(self.train)
            .map(lambda x: [FinetuneMessage.model_validate(m) for m in x["messages"]])
            .map(lambda convo: FinetuneConversation(messages=convo))
        )
        return convos

    def to_val_convos(self) -> Sequence[FinetuneConversation]:
        convos = (
            Slist(self.val)
            .map(lambda x: [FinetuneMessage.model_validate(m) for m in x["messages"]])
            .map(lambda convo: FinetuneConversation(messages=convo))
        )
        return convos


def james_make_finetuning(
    train_base_dir: str,
    val_base_dir: str | None,
    task: str,
    response_property: str,
    prompt_template: str,  # todo: make it lookup the correct path
    n_train_items: int,
    n_val_items: int,
    seed: int = 0,
    # Threshold for the probability of the first token in the response to be used for training
    probability_threshold: float = 0.0,
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
        train_base_dir, response_property_name=response_property, seed=seed, n_items=n_train_items
    )

    # Load and process validation data
    val_df = (
        load_and_process_data(val_base_dir, response_property_name=response_property, seed=seed, n_items=n_val_items)
        if val_base_dir
        else None
    )

    if probability_threshold > 0:
        train_df = filter_for_threshold(
            df=train_df, response_property_name=response_property, probability_threshold=probability_threshold
        )
        val_df = (
            filter_for_threshold(
                df=val_df, response_property_name=response_property, probability_threshold=probability_threshold
            )
            if val_df is not None
            else None
        )
    assert len(train_df) > 0, "No training data found."
    if val_base_dir:
        assert len(val_df) > 0, "No validation data found."

    prompt_template_obj = get_meta_level_template(prompt_template)
    task_prompt = get_task_prompt(task)
    response_prompt = get_response_prompt(response_property)

    train = generate_and_save_messages(
        df=train_df,
        prompt_template=prompt_template_obj,
        task_prompt=task_prompt,
        response_prompt=response_prompt,
        response_col=response_property,
    )
    val = (
        generate_and_save_messages(
            df=val_df,
            prompt_template=prompt_template_obj,
            task_prompt=task_prompt,
            response_prompt=response_prompt,
            response_col=response_property,
        )
        if val_df is not None
        else []
    )
    return GeneratedDataset(train=train, val=val)


def get_response_property_function(response_property: str) -> str:
    # load the yaml
    yaml_path = Path(__file__).parent / "conf" / "response_property" / f"{response_property}.yaml"
    # get the response_property_function
    conf = OmegaConf.load(yaml_path)
    _conf = OmegaConf.to_container(conf, resolve=False)
    return _conf["python_function"]


def load_and_process_data(base_dir, response_property_name, seed, n_items=None):
    # df = load_single_df(Path(base_dir))
    path = Path(base_dir) / f"data{seed}.csv"
    # read as str
    df = pd.read_csv(path, dtype=str)
    # drop nan "["response"]"
    df = df.dropna(subset=["response"])
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


def get_highest_logprob(logprobs: str) -> float:
    # we should really just store things as json
    list_of_probs = eval(logprobs)
    # the first token
    first_token: dict = list_of_probs[0]
    # technically can sort if you want and using logprobs>1
    assert len(first_token) == 1, "Did you run with logprobs=1"
    logprob = list(first_token.values())[0]
    # nat base
    proba = math.exp(logprob)
    assert 0 <= proba <= 1, f"Proba is {proba}"
    return proba


def filter_for_threshold(df, response_property_name: str, probability_threshold: float) -> pd.DataFrame:
    """
    Filter the dataframe for rows where the probability of the first token is above the threshold.

    Args:
        df (pd.DataFrame): The dataframe to filter.
        response_property_name (str): The name of the response property column.
        probability_threshold (float): The threshold for the probability of the first token.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    df["first_token_proba"] = df["logprobs"].apply(lambda x: get_highest_logprob(x))
    df = df[df["first_token_proba"] > probability_threshold]
    return df


def generate_and_save_messages(
    df, prompt_template, response_col: str, task_prompt: str, response_prompt: str
) -> Sequence[dict]:
    output = []
    for _, row in df.iterrows():
        try:
            prompt = substitute_string(
                row=row,
                prompt_template=prompt_template,
                task_prompt=task_prompt,
                response_property_prompt=response_prompt,
                response_col=response_col,
            )
            prompt = prompt.openai_finetuning_format()
            prompt = json.loads(prompt)
            prompt["string"] = row["string"]
            output.append(prompt)
        except ValidationError as e:
            LOGGER.warning(f"Failed row with error {e}")
    return output


def substitute_string(
    row: pd.Series,
    prompt_template: PromptTemplate,
    task_prompt: str,
    response_property_prompt: str,
    response_col: str = "response",
) -> Prompt:
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
        sub_content = message.content.replace("${task.prompt}", task_prompt)
        sub_content = sub_content.replace("${response_property.meta_level_prompt}", response_property_prompt)
        t = Template(sub_content)

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

# out = james_make_finetuning(
#     train_base_dir="/Users/jameschua/ml/introspection_self_prediction_astra/exp/claude_data_try_3/object_level_claude-3-5-sonnet-20240620_object_level_minimal_prompt_wikipedia_long_train_task__note",
#     val_base_dir="/Users/jameschua/ml/introspection_self_prediction_astra/exp/claude_data_try_3/object_level_claude-3-5-sonnet-20240620_object_level_minimal_prompt_wikipedia_long_train_task__note",
#     response_property="first_character",
#     task="wikipedia_long",
#     prompt_template="minimal",
#     n_train_items=1000,
#     n_val_items=200,
#     seed=0,
# ).train
# print(out)
