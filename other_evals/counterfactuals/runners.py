from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
from typing import Type
from git import Sequence
import pandas as pd

from slist import Slist
from evals.apis.inference.api import InferenceAPI
from evals.locations import EXP_DIR
from evals.utils import setup_environment
from other_evals.counterfactuals.inference_api_cache import CachedInferenceAPI
from other_evals.counterfactuals.other_eval_csv_format import OtherEvalCSVFormat
from other_evals.counterfactuals.plotting.plot_heatmap import plot_heatmap_with_ci
from other_evals.counterfactuals.run_ask_are_you_sure import run_single_are_you_sure
from other_evals.counterfactuals.run_ask_if_affected import run_single_ask_if_affected
from other_evals.counterfactuals.run_ask_if_gives_correct_answer import run_single_ask_if_correct_answer
from other_evals.counterfactuals.run_ask_what_answer_without_bias import run_single_what_answer_without_bias


class OtherEvalRunner(ABC):
    @staticmethod
    @abstractmethod
    async def run(
        eval_name: str,
        meta_model: str,
        object_model: str,
        api: CachedInferenceAPI,
        limit: int = 100,
    ) -> Sequence[OtherEvalCSVFormat]: ...


class AreYouAffectedByBias(OtherEvalRunner):
    @staticmethod
    async def run(
        eval_name: str, meta_model: str, object_model: str, api: CachedInferenceAPI, limit: int = 100
    ) -> Sequence[OtherEvalCSVFormat]:
        """Ask the model if it was affected by the bias. Y/N answers"""

        result = await run_single_ask_if_affected(
            object_model=object_model,
            meta_model=meta_model,
            api=api,
            number_samples=limit,
        )
        formatted: Slist[OtherEvalCSVFormat] = result.map(lambda x: x.to_other_eval_format(eval_name=eval_name))
        return formatted


class WhatAnswerWithoutBias(OtherEvalRunner):
    @staticmethod
    async def run(
        eval_name: str, meta_model: str, object_model: str, api: CachedInferenceAPI, limit: int = 100
    ) -> Sequence[OtherEvalCSVFormat]:
        """Ask the model what answer it would have given w/o the bias. A,B,C,D answers"""

        result = await run_single_what_answer_without_bias(
            object_model=object_model,
            meta_model=meta_model,
            api=api,
            number_samples=limit,
        )
        formatted = result.map(lambda x: x.to_other_eval_format(eval_name=eval_name))
        return formatted


class ChangeAnswerAreYouSure(OtherEvalRunner):
    @staticmethod
    async def run(
        eval_name: str, meta_model: str, object_model: str, api: CachedInferenceAPI, limit: int = 100
    ) -> Sequence[OtherEvalCSVFormat]:
        """Ask the model if it would change its prediction if we ask 'ask you sure'. Y/N answers"""
        result = await run_single_are_you_sure(
            object_model=object_model,
            meta_model=meta_model,
            api=api,
            number_samples=limit,
        )
        formatted = result.map(lambda x: x.to_other_eval_format(eval_name=eval_name))
        return formatted


class WillYouBeCorrect(OtherEvalRunner):
    @staticmethod
    async def run(
        eval_name: str, meta_model: str, object_model: str, api: CachedInferenceAPI, limit: int = 100
    ) -> Sequence[OtherEvalCSVFormat]:
        """Ask the model if it is going to get the correct answer. Y/N answers"""
        result = await run_single_ask_if_correct_answer(
            object_model=object_model,
            meta_model=meta_model,
            api=api,
            number_samples=limit,
        )
        formatted = result.map(lambda x: x.to_other_eval_format(eval_name=eval_name))

        return formatted


EVAL_NAME_TO_RUNNER: dict[str, Type[OtherEvalRunner]] = {
    "ChangeAnswerAreYouSure": ChangeAnswerAreYouSure,
    "AreYouAffectedByBias": AreYouAffectedByBias,
    "WhatAnswerWithoutBias": WhatAnswerWithoutBias,
    "WillYouBeCorrect": WillYouBeCorrect,
}
all_evals: list[str] = list(EVAL_NAME_TO_RUNNER.keys())
runner_to_eval_name = {v: k for k, v in EVAL_NAME_TO_RUNNER.items()}
assert len(EVAL_NAME_TO_RUNNER) == len(
    runner_to_eval_name
), "The mapping is not bijective, maybe you have duplicate keys / values?"


async def run_from_commands(
    evals_to_run: Sequence[Type[OtherEvalRunner]],
    object_and_meta: Sequence[tuple[str, str]],
    limit: int,
    study_folder: str | Path,
    api: CachedInferenceAPI,
) -> Slist[OtherEvalCSVFormat]:
    """Run the appropriate evaluation based on the dictionary"""
    # coorountines_to_run: Slist[Awaitable[Sequence[OtherEvalCSVFormat]]] = Slist()
    gathered = Slist()
    for object_model, meta_model in object_and_meta:
        for runner in evals_to_run:
            eval_name = runner_to_eval_name[runner]
            # coorountines_to_run.append(runner.run(meta_model=meta_model, object_model=object_model, cache_path=cache_path, limit=limit))
            result = await runner.run(
                eval_name=eval_name,
                meta_model=meta_model,
                object_model=object_model,
                limit=limit,
                api=api,
            )
            gathered.append(result)

    # todo: do we really want to run all of these at the same time? lol
    # gathered = await coorountines_to_run.gather()
    # we get a list of lists, so we flatten it
    flattened: Slist[OtherEvalCSVFormat] = gathered.flatten_list()
    return flattened


def eval_list_to_runner(eval_list: list[str]) -> Sequence[Type[OtherEvalRunner]]:
    runners = []
    for eval_str in eval_list:
        maybe_eval_runner = EVAL_NAME_TO_RUNNER.get(eval_str, None)
        if maybe_eval_runner is not None:
            runners.append(maybe_eval_runner)
        else:
            raise ValueError(
                f"Could not find runner for {eval_str}, is it present in EVAL_NAME_TO_RUNNER. Available keys: {EVAL_NAME_TO_RUNNER.keys()}"
            )

    return runners


def sweep_over_evals(
    eval_list: list[str], object_and_meta: Sequence[tuple[str, str]], limit: int, study_folder: str | Path
) -> None:
    """
    ["asked_if_are_you_sure_changed", "ask_if_affected"],
    """
    # Entry point for sweeping
    evals_to_run = eval_list_to_runner(eval_list)
    # the sweep ain't a async function so we use asyncio.run
    api = InferenceAPI(anthropic_num_threads=20)
    inference_api = CachedInferenceAPI(api=api, cache_path=Path(study_folder) / "cache")

    all_results = asyncio.run(
        run_from_commands(
            evals_to_run=evals_to_run,
            object_and_meta=object_and_meta,
            limit=limit,
            study_folder=study_folder,
            api=inference_api,
        )
    )
    grouped_by_eval = all_results.group_by(lambda x: x.eval_name)
    for eval_name, results_list in grouped_by_eval:
        df = pd.DataFrame(results_list.map(lambda x: x.model_dump()))
        plot_heatmap_with_ci(
            data=df,
            value_col="meta_predicted_correctly",
            object_col="object_model",
            meta_col="meta_model",
            title=f"{eval_name} Percentage of Meta Response Predicted Correctly with 95% CI",
        )
        df.to_csv(Path(study_folder) / f"{eval_name}_results.csv", index=False)


def test_main():
    setup_environment()
    # What evals to run?
    # See the keys in the EVAL_NAME_TO_RUNNER
    eval_list = all_evals
    print(f"Running evals: {eval_list}")
    # What models to run?
    models = Slist(
        [
            "gpt-3.5-turbo",
            # "claude-3-sonnet-20240229",
            "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo::9PutAYsj",
            # "gpt-4o",
        ]
    )
    # We want to run all the combinations of the models
    object_and_meta_models: Slist[tuple[str, str]] = models.product(models)
    study_folder = EXP_DIR / "other_evals"
    limit = 300
    sweep_over_evals(
        eval_list=eval_list, object_and_meta=object_and_meta_models, limit=limit, study_folder=study_folder
    )


if __name__ == "__main__":
    test_main()
