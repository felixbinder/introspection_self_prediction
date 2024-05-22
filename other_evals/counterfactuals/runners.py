from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
from typing import Type
from git import Sequence
import pandas as pd

from slist import Slist
from evals.locations import EXP_DIR
from evals.utils import setup_environment
from other_evals.counterfactuals.other_eval_csv_format import OtherEvalCSVFormat
from other_evals.counterfactuals.plotting.plot_heatmap import plot_heatmap_with_ci_flipped
from other_evals.counterfactuals.run_ask_are_you_sure import run_single_are_you_sure
from other_evals.counterfactuals.run_ask_if_affected import run_single_ask_if_affected


class OtherEvalRunner(ABC):
    @staticmethod
    @abstractmethod
    async def run(
        meta_model: str, object_model: str, cache_path: str | Path, limit: int = 100
    ) -> Sequence[OtherEvalCSVFormat]: ...


class AskIfAffectedRunner:
    @staticmethod
    async def run(
        meta_model: str, object_model: str, cache_path: str | Path, limit: int = 100
    ) -> Sequence[OtherEvalCSVFormat]:
        """Ask the model if it was affected by the bias. Y/N answers"""

        result = await run_single_ask_if_affected(
            object_model=object_model,
            meta_model=meta_model,
            cache_path=cache_path,
            number_samples=limit,
        )
        formatted = result.map(lambda x: x.to_other_eval_format(eval_name="asked_if_affected"))
        return formatted


class AreYouSureRunner(OtherEvalRunner):
    @staticmethod
    async def run(
        meta_model: str, object_model: str, cache_path: str | Path, limit: int = 100
    ) -> Sequence[OtherEvalCSVFormat]:
        """Ask the model if it would change its prediction if we ask 'ask you sure'. Y/N answers"""
        result = await run_single_are_you_sure(
            object_model=object_model,
            meta_model=meta_model,
            cache_path=cache_path,
            number_samples=limit,
        )
        formatted = result.map(lambda x: x.to_other_eval_format(eval_name="are_you_sure"))
        return formatted


async def run_from_commands(
    evals_to_run: Sequence[Type[OtherEvalRunner]],
    object_and_meta: Sequence[tuple[str, str]],
    limit: int,
    study_folder: str | Path,
) -> Sequence[OtherEvalCSVFormat]:
    """Run the appropriate evaluation based on the dictionary"""
    # coorountines_to_run: Slist[Awaitable[Sequence[OtherEvalCSVFormat]]] = Slist()
    gathered = Slist()
    cache_path = Path(study_folder) / "model_cache.jsonl"
    for object_model, meta_model in object_and_meta:
        for runner in evals_to_run:
            # coorountines_to_run.append(runner.run(meta_model=meta_model, object_model=object_model, cache_path=cache_path, limit=limit))
            result = await runner.run(
                meta_model=meta_model, object_model=object_model, cache_path=cache_path, limit=limit
            )
            gathered.append(result)

    # todo: do we really want to run all of these at the same time? lol
    # gathered = await coorountines_to_run.gather()
    # we get a list of lists, so we flatten it
    flattened: Slist[OtherEvalCSVFormat] = gathered.flatten_list()
    return flattened


def eval_dict_to_runner(eval_dict: dict[str, list[str]]) -> Sequence[Type[OtherEvalRunner]]:
    runners = []
    for key, value_list in eval_dict.items():
        for value in value_list:
            match key, value:
                case "biased_evals", "are_you_sure":
                    runners.append(AreYouSureRunner)
                case "biased_evals", "ask_if_affected":
                    runners.append(AskIfAffectedRunner)
                case _, _:
                    raise ValueError(f"Unexpected evaluation type: {key=} {value=}")
    return runners


async def main():
    eval_dict = {
        "biased_evals": ["are_you_sure", "ask_if_affected"],
    }
    models = Slist(
        [
            "gpt-3.5-turbo",
            "claude-3-sonnet-20240229",
        ]
    )
    setup_environment()
    object_and_meta_models: Slist[tuple[str, str]] = models.product(models)
    study_folder = EXP_DIR / "other_evals"
    limit = 500
    evals_to_run = eval_dict_to_runner(eval_dict)
    results = await run_from_commands(
        evals_to_run=evals_to_run, object_and_meta=object_and_meta_models, limit=limit, study_folder=study_folder
    )
    dicts = [result.model_dump() for result in results]
    df = pd.DataFrame(dicts)
    plot_heatmap_with_ci_flipped(
        data=df,
        value_col="meta_predicted_correctly",
        object_col="object_model",
        meta_col="meta_model",
        title="Percentage of Meta Predicted Correctly with 95% CI",
    )
    df.to_csv(study_folder / "other_evals_results.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main())
