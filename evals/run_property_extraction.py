"""This file is used to use an LLM to extract properties from object-level responses."""

import asyncio
import csv
import logging
import traceback
from pathlib import Path
from string import Template

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from evals.analysis.loading_data import get_data_path
from evals.apis.inference.api import InferenceAPI
from evals.apis.inference.cache_manager import CacheManager
from evals.data_models.inference import LLMParams
from evals.data_models.messages import ChatMessage, Prompt, PromptTemplate
from evals.utils import (
    async_function_with_retry,
    get_current_git_hash,
    import_function_from_string,
    setup_environment,
)

LOGGER = logging.getLogger(__name__)


class DatasetRunner:
    def __init__(
        self,
        cfg: DictConfig,
        prompt_template: PromptTemplate,
        llm_params: LLMParams,
        inference_api: InferenceAPI,
        print_prompt_and_response: bool = False,
        cache_manager: CacheManager = None,
    ):
        self.prompt_template = prompt_template
        self.property_name = cfg.response_property.name
        self.cfg = cfg
        self.llm_params = llm_params
        self.inference_api = inference_api
        self.print_prompt_and_response = print_prompt_and_response
        self.cache_manager = cache_manager

    async def run(self, index: int, row: pd.Series) -> dict:
        prompt = self.process_prompt(row)

        # load cache if available
        if self.cache_manager is not None:
            cache = self.cache_manager.maybe_load_cache(prompt, self.llm_params)
            if cache is not None:
                LOGGER.info(f"Loaded cache for row {index}")
                return {
                    "answer": cache.responses[0].completion,
                    "logprobs": cache.responses[0].logprobs,
                    f"{self.property_name}_complete": True,
                }

        try:
            responses = await self.inference_api(
                model_ids=self.llm_params.model,
                prompt=prompt,
                temperature=self.llm_params.temperature,
                max_tokens=self.llm_params.max_tokens,
                top_p=self.llm_params.top_p,
                num_candidates_per_completion=self.llm_params.num_candidates_per_completion,
                insufficient_valids_behaviour=self.llm_params.insufficient_valids_behaviour,
                is_valid=lambda x: True,  # len(x) > 0 and len(x) < 10 and " " not in x, # x should be a single word
                print_prompt_and_response=self.print_prompt_and_response,
                logprobs=self.llm_params.logprobs,
                seed=self.llm_params.seed,
            )
            # save successful prompt/response to file
            if self.cache_manager is not None:
                self.cache_manager.save_cache(prompt, self.llm_params, responses)

            answer = responses[0].completion
            logprobs = responses[0].logprobs
            complete = True
            self.inference_api.log_model_timings()
            LOGGER.info(f"Completed row {index}\tRunning cost: ${self.inference_api.running_cost:.3f}")
        except RuntimeError as e:
            complete = False
            answer = traceback.format_exc()
            logprobs = None
            LOGGER.warning(f"Failed row {index} with error {e}")
            LOGGER.warning(answer)
        return {
            "answer": answer,
            "logprobs": logprobs,
            f"{self.property_name}_complete": complete,
        }

    def process_prompt(self, row: pd.Series) -> Prompt:
        messages = []
        system_messages = [m for m in self.prompt_template.messages if m.role == "system"]
        user_messages = [m for m in self.prompt_template.messages if m.role == "user"]

        # add system messages
        for message in system_messages:
            t = Template(message.content)
            content = t.safe_substitute(response=row["response"])
            messages.append(ChatMessage(role=message.role, content=content))

        for message in user_messages:
            t = Template(message.content)
            content = t.safe_substitute(response=row["response"])
            messages.append(ChatMessage(role=message.role, content=content))

        return Prompt(messages=messages)


async def run_dataset(filename: str, property_name: str, dataset_runner: DatasetRunner, limit: int = None) -> bool:
    # load dataset and filter out completed rows
    full_df = pd.read_csv(filename)
    if property_name not in full_df.columns:
        full_df[property_name] = ""
    if f"{property_name}_complete" not in full_df.columns:
        full_df[f"{property_name}_complete"] = False
    df = full_df[~(full_df[f"{property_name}_complete"])]

    # remove repetive responses to prevent problems with the OpenAI API & Gemini having a stroke
    df = remove_repetive_responses(df)

    # run each question concurrently
    LOGGER.info(f"Processing {len(df)} rows")
    tasks = [dataset_runner.run(i, row) for i, row in df.iterrows()]
    results = await asyncio.gather(*tasks)

    # update dataframe with results
    completed = sum([bool(result[f"{property_name}_complete"]) for result in results])
    LOGGER.info(f"Processed {len(results)} rows. {completed} were complete.")
    df.update(
        pd.DataFrame(
            {
                property_name: [result["answer"] for result in results],
                f"{property_name}_complete": [result[f"{property_name}_complete"] for result in results],
            },
            index=df.index,
        )
    )
    full_df.update(df)
    full_df.to_csv(filename, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    # return whether all rows are complete
    if full_df[f"{property_name}_complete"].eq(True).all():
        LOGGER.info("All rows complete!")
        return True
    else:
        LOGGER.info("Not all rows complete. Retrying...")
        return False


async def async_main(cfg: DictConfig):
    LOGGER.info(f"Using experiment directory {cfg.exp_dir}")
    LOGGER.info(f"Using model {cfg.language_model.model}")
    LOGGER.info(f"Using property {cfg.response_property.name}")

    # we load the file from the dir and change it in place
    filepath = get_data_path(cfg.dir)

    # do we have a python function set? In that case, we bail on the LLM stuff and just run the function
    if cfg.response_property.python_function is not None and cfg.response_property.python_function != "none":
        LOGGER.info(f"Running python function {cfg.response_property.python_function}")
        apply_python_function(cfg.response_property, filepath)
        return True

    # setup api handler
    setup_environment(anthropic_tag=cfg.anthropic_tag, logging_level=cfg.logging)
    prompt_history_dir = Path(cfg.prompt_history_dir) if cfg.prompt_history_dir is not None else None
    inference_api = InferenceAPI(
        anthropic_num_threads=cfg.anthropic_num_threads,
        openai_fraction_rate_limit=cfg.openai_fraction_rate_limit,
        organization=cfg.organization,
        prompt_history_dir=prompt_history_dir,
    )
    # load configs
    prompt_parts = PromptTemplate(**OmegaConf.to_container(cfg.prompt, resolve=True))
    llm_params = LLMParams(**OmegaConf.to_container(cfg.language_model, resolve=True))
    cache_manager = CacheManager(Path(cfg.cache_dir)) if cfg.cache_dir is not None else None
    dataset_runner = DatasetRunner(
        cfg, prompt_parts, llm_params, inference_api, cfg.print_prompt_and_response, cache_manager
    )

    # run dataset (with retry)
    complete = await async_function_with_retry(
        run_dataset,
        filepath,
        cfg.response_property.name,
        dataset_runner,
        limit=None,
    )
    return complete


def apply_python_function(response_property: DictConfig, filepath: str):
    """Apply a python function to the dataset."""
    LOGGER.info(f"Applying python function {response_property.python_function} to {filepath}")
    try:
        function = import_function_from_string("evals.response_property", response_property.python_function)
    except ImportError:
        LOGGER.error(f"Could not import python function {response_property.python_function}")
        raise
    # load the dataset
    df = pd.read_csv(filepath)
    # ensure that all rows are strings
    for column in df.columns:
        if column not in ["complete"]:
            df[column] = df[column].astype(str)
    # apply the function
    df[response_property.name] = df.apply(lambda row: try_function(function, row), axis=1)
    # save the dataset
    df.to_csv(filepath, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    LOGGER.info(f"Applied python function {response_property.python_function} to {filepath}")


def try_function(function, row):
    try:
        return function(row)
    except Exception as e:
        LOGGER.error(f"Error running function {function.__name__} on row {row}")
        LOGGER.error(e)
        return np.nan


def remove_repetive_responses(df):
    """
    Sometimes models repeat the same word over and over. To prevent the OpenAI API, we truncate the responses/identity
    """

    # find responses that repeat the same word
    def is_repetitive(response):
        response = str(response)
        return len(set(response.split())) == 1 and len(response.split()) > 10

    repetitive_mask = df["response"].apply(is_repetitive)
    # truncate the responses
    df.loc[repetitive_mask, "response"] = df.loc[repetitive_mask, "response"].apply(
        lambda x: " ".join(x.split()[0:10]) + "<truncated repetive response>"
    )
    if "identity"  in df.columns:
        df.loc[repetitive_mask, "identity"] = df.loc[repetitive_mask, "identity"].apply(
            lambda x: " ".join(x.split()[0:10]) + "<truncated repetive response>"
        )
    if len(df[repetitive_mask]) > 0:
        LOGGER.warn(f"Truncated {len(df[repetitive_mask])} repetive responses")
    return df


@hydra.main(version_base=None, config_path="conf", config_name="config_property_extraction")
def main(cfg: DictConfig):
    print("Current git hash:", get_current_git_hash())
    print(cfg)
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
