"""This is used to run the self prediction task."""

import asyncio
import csv
import logging
import shutil
import traceback
from pathlib import Path
from string import Template

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from evals.analysis.loading_data import find_matching_base_dir
from evals.apis.inference.api import InferenceAPI
from evals.apis.inference.cache_manager import CacheManager
from evals.data_models.inference import LLMParams
from evals.data_models.messages import ChatMessage, MessageRole, Prompt, PromptTemplate
from evals.generate_few_shot import generate_few_shot_data
from evals.utils import (
    async_function_with_retry,
    collate_mode_of_n,
    gather_max_par,
    get_current_git_hash,
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
            # self.inference_api.log_model_timings()
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
        # extract the few-shot strings and responses
        # # this isn't safe, but then again, who would expect an AI model to not be safe?
        few_shot_strings = eval(row["few-shot_string"])
        few_shot_strings = [str(s) for s in few_shot_strings]
        few_shot_responses = eval(row["few-shot_response"])
        few_shot_responses = [str(s) for s in few_shot_responses]

        messages = []
        system_messages = [m for m in self.prompt_template.messages if m.role == "system"]
        user_messages = [m for m in self.prompt_template.messages if m.role == "user"]

        # add system messages
        for message in system_messages:
            t = Template(message.content)
            content = t.safe_substitute(string=row["string"])
            messages.append(ChatMessage(role=message.role, content=content))

        # add in few shot messages into the context
        assert len(few_shot_strings) == len(
            few_shot_responses
        ), f"Expected the same number of few-shot strings and responses, but got {len(row['few-shot_string'])} strings and {len(row['few-shot_response'])} responses."
        for few_shot_string, few_shot_response in zip(few_shot_strings, few_shot_responses):
            for message in user_messages:
                # add in the few-shot string for each user message (usually just one)
                t = Template(message.content)
                content = t.safe_substitute(string=few_shot_string)
                messages.append(ChatMessage(role=message.role, content=content))
                # add in the few-shot response for each user message (usually just one)
                m = ChatMessage(role=MessageRole.assistant, content=few_shot_response)
                messages.append(m)

        # add the actual query message
        for message in user_messages:
            t = Template(message.content)
            content = t.safe_substitute(string=row["string"])
            messages.append(ChatMessage(role=message.role, content=content))

        return Prompt(messages=messages)


async def run_dataset(filename: str, dataset_runner: DatasetRunner, limit: int = None, n_samples: int = 1) -> bool:
    # load dataset and filter out completed rows
    full_df = pd.read_csv(filename)
    if limit is not None:
        full_df = full_df.head(limit * n_samples)
    if dataset_runner.cfg.response_property.name not in full_df.columns:
        full_df[dataset_runner.cfg.response_property.name] = ""
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
    results = await gather_max_par(50, *tasks)

    # update dataframe with results
    completed = sum([bool(result["complete"]) for result in results])
    LOGGER.info(f"Processed {len(results)} rows. {completed} were complete.")
    df.update(
        pd.DataFrame(
            {
                "response": [result["answer"] for result in results],
                dataset_runner.cfg.response_property.name: [
                    result["answer"] for result in results
                ],  # also save the answer into the response property for later comparision
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
        cfg, prompt_parts, llm_params, inference_api, cfg.print_prompt_and_response, cache_manager
    )

    # load dataset and save to file
    exp_dir = Path(cfg.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    filename = exp_dir / f"raw_data{cfg.base_seed}.csv"
    if not filename.exists() or cfg.reset:
        # have to make the data_{seed}.csv file
        LOGGER.info(f"File {filename} does not exist. Creating...")
        new_filename = setup_data_file(cfg, exp_dir, filename)
        assert new_filename == filename, f"Expected {new_filename} to be {filename}"
        LOGGER.info(f"Generated data{cfg.base_seed}.csv at {filename}")
    else:
        LOGGER.info(f"File {filename} exists. Skipping generation.")
    model = cfg.language_model.model
    if "llama-" in model or "claude-3-5-sonnet-20240620" in model or "claude-3-haiku-20240307" in model:
        # If using llama, its not an moe, so no need to take mode
        print(f"Setting n_samples to 1 since using {cfg.language_model.model}")
        n_samples = 1
    else:
        print("Setting n_samples to cfg", cfg.n_samples)
        n_samples = cfg.n_samples

    # run dataset (with retry)
    complete = await async_function_with_retry(
        run_dataset,
        filename,
        dataset_runner,
        limit=cfg.limit,
        n_samples=n_samples,
    )
    if n_samples > 1:
        collate_mode_of_n(filename)  # collate the data to get modal response for each sample
    else:
        LOGGER.info("Only one sample, no need to collate mode")
        shutil.copy(filename, filename.parent / f"{filename.stem.replace('raw_data', 'data')}.csv")
    print(exp_dir)  # print the experiment directory for scripting purposes
    return complete


def setup_data_file(cfg, exp_dir, filename):
    # we have to create the data{seed}.csv file
    if cfg.strings_path != "none" and cfg.strings_path is not None:
        strings_path = Path(cfg.strings_path)
        assert (
            strings_path.exists()
        ), f"Strings file {strings_path} does not exist. Use evals/extract_[...].py to generate strings file"
    else:
        strings_path = None
        LOGGER.info("No strings file provided. Using the base data as the strings file.")
    if cfg.base_dir is None or cfg.base_dir == "none":
        LOGGER.warning(f"No base data directory provided. Trying to find one in {cfg.study_dir}")
        base_data_path = find_matching_base_dir(cfg)
    else:
        base_data_path = Path(cfg.base_dir)
    base_data_path = base_data_path / f"data{cfg.base_seed}.csv"
    new_filename = generate_few_shot_data(
        base_data_path=base_data_path,
        strings_path=strings_path,
        response_property_name=cfg.response_property.name,
        filter_strings_path=cfg.task.get("filter_strings_path", None),
        n_shot=cfg.n_shot,
        output_path=filename,
        seed=cfg.seed,
        how=cfg.get("n_shot_seeding", "true"),
        repeat=cfg.get("n_samples", 1),
    )
    return new_filename


@hydra.main(version_base=None, config_path="conf", config_name="config_meta_level")
def main(cfg: DictConfig):
    print("Current git hash:", get_current_git_hash())
    print(cfg)
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
