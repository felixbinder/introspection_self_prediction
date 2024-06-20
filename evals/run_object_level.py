"""This file is used to find initial completions which are hard to predict."""

import asyncio
import csv
import logging
import traceback
from pathlib import Path
from string import Template
from typing import Optional

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tenacity import RetryError

from evals.apis.inference.api import InferenceAPI
from evals.apis.inference.cache_manager import CacheManager
from evals.data_models.inference import LLMParams
from evals.data_models.messages import ChatMessage, Prompt, PromptTemplate
from evals.load.load_dataset import create_data_file, load_dataset
from evals.utils import (
    async_function_with_retry,
    get_current_git_hash,
    setup_environment,
)

LOGGER = logging.getLogger(__name__)


class DatasetRunner:
    def __init__(
        self,
        prompt_template: PromptTemplate,
        llm_params: LLMParams,
        inference_api: InferenceAPI,
        print_prompt_and_response: bool = False,
        cache_manager: CacheManager = None,
    ):
        self.prompt_template = prompt_template
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
                    "complete": True,
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
                cais_path=self.llm_params.cais_path,
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
            "complete": complete,
        }

    def process_prompt(self, row: pd.Series) -> Prompt:
        messages = []
        for message in self.prompt_template.messages:
            t = Template(message.content)
            content = t.safe_substitute(string=row["string"])
            messages.append(ChatMessage(role=message.role, content=content))

        return Prompt(messages=messages)


async def run_dataset(filename: str, dataset_runner: DatasetRunner, limit: Optional[int] = None) -> bool:
    # load dataset and filter out completed rows
    full_df = pd.read_csv(filename)
    if limit is not None:
        full_df = full_df.head(limit)
    if "response" not in full_df.columns:
        full_df["response"] = ""
    if "complete" not in full_df.columns:
        full_df["complete"] = False
    if "logprobs" not in full_df.columns:
        full_df["logprobs"] = {}
    df = full_df[~(full_df["complete"])]

    # run each question concurrently
    LOGGER.info(f"Processing {len(df)} rows")
    tasks = [dataset_runner.run(i, row) for i, row in df.iterrows()]
    results = await asyncio.gather(*tasks)

    # update dataframe with results
    completed = sum([bool(result["complete"]) for result in results])
    LOGGER.info(f"Processed {len(results)} rows. {completed} were complete.")
    df.update(
        pd.DataFrame(
            {
                "response": [result["answer"] for result in results],
                "logprobs": [result["logprobs"] for result in results],
                "complete": [result["complete"] for result in results],
            },
            index=df.index,
        )
    )
    full_df.update(df)
    full_df.to_csv(filename, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    # return whether all rows are complete
    if full_df["complete"].eq(True).all():
        LOGGER.info("All rows complete!")
        return True
    else:
        LOGGER.info("Not all rows complete. Retrying...")
        return False


async def async_main(cfg: DictConfig):
    LOGGER.info(f"Using experiment directory {cfg.exp_dir}")
    LOGGER.info(f"Using model {cfg.language_model.model}")
    LOGGER.info(f"Using method {cfg.prompt.method}")

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
        prompt_parts, llm_params, inference_api, cfg.print_prompt_and_response, cache_manager
    )

    # load dataset and save to file
    exp_dir = Path(cfg.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    filename = exp_dir / f"data{cfg.seed}.csv"
    if not filename.exists() or cfg.reset:
        LOGGER.info(f"File {filename} does not exist. Creating...")
        data = load_dataset(
            path=cfg.task.dataset_path,
            seed=cfg.seed,
            shuffle=cfg.task.shuffle,
            n=cfg.task.num,
            n_samples=cfg.n_samples,
            filter_strings_path=cfg.task.get("filter_strings_path", None),
        )
        if cfg.strings_path is not None and cfg.strings_path != "none":
            LOGGER.info(f"Using strings from {cfg.strings_path}. Filtering csv to only include these strings.")
            strings_df = pd.read_csv(cfg.strings_path)
            before_len = len(data)
            allowed_strings = set(strings_df["string"])
            data = data[data["string"].isin(allowed_strings)]
            after_len = len(data)
            LOGGER.info(f"Filtered data from {before_len} to {after_len} based on strings.")
        create_data_file(data, filename)
    else:
        LOGGER.info(f"File {filename} exists. Will not recreate.")

    # run dataset (with retry)
    try:
        complete = await async_function_with_retry(
            run_dataset,
            filename,
            dataset_runner,
            limit=cfg.limit,
        )
    except RetryError as e: # make sure to reraise the proper error not the one from the async function
        LOGGER.error(f"Failed with error {e}")
        LOGGER.error(traceback.format_exc())
        LOGGER.error("Failed to complete dataset—at least one row is not completed.")
        complete = False
    print(exp_dir)  # print the experiment directory for scripting purposes
    return complete


@hydra.main(version_base=None, config_path="conf", config_name="config_object_level")
def main(cfg: DictConfig):
    print("Current git hash:", get_current_git_hash())
    print(cfg)
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
