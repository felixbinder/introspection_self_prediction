from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
from typing import Awaitable, Type
from git import Sequence
import pandas as pd

from pydantic import BaseModel
from slist import Slist
from evals.locations import EXP_DIR
from evals.utils import load_secrets, setup_environment
from other_evals.counterfactuals.other_eval_csv_format import OtherEvalCSVFormat
from other_evals.counterfactuals.run_ask_are_you_sure import run_single_are_you_sure


class OtherEvalRunner(ABC):
    @staticmethod
    @abstractmethod
    async def run(
        meta_model: str, object_model: str, cache_path: str | Path, limit: int = 100
    ) -> Sequence[OtherEvalCSVFormat]: ...


# class AskIfAffectedRunner:
#     @staticmethod
#     async def run(meta_model: str, object_model: str, limit: int = 100) -> str:
#         ...


class AreYouSureRunner(OtherEvalRunner):
    @staticmethod
    async def run(
        meta_model: str, object_model: str, cache_path: str | Path, limit: int = 100
    ) -> Sequence[OtherEvalCSVFormat]:
        """Ask the model if it would change its prediction if we ask 'ask you sure'"""
        # the model doesn't switch all the time, we need enough samples
        # TODO: less hacky way to do this
        result = await run_single_are_you_sure(
            object_model=object_model,
            meta_model=meta_model,
            cache_path=cache_path,
            number_samples=limit,
        )
        formatted = result.map(
            lambda x: x.to_other_eval_format(eval_name="are_you_sure")
        )
        return formatted



async def run_from_commands(
    evals_to_run: Sequence[Type[OtherEvalRunner]], object_and_meta: Sequence[tuple[str, str]], limit: int, study_folder: str | Path,
) -> Sequence[OtherEvalCSVFormat]:
    """Run the appropriate evaluation based on the dictionary"""
    # coorountines_to_run: Slist[Awaitable[Sequence[OtherEvalCSVFormat]]] = Slist()
    gathered = Slist()
    cache_path = Path(study_folder) / "model_cache.jsonl"
    for object_model, meta_model in object_and_meta:
        for runner in evals_to_run:
            # coorountines_to_run.append(runner.run(meta_model=meta_model, object_model=object_model, cache_path=cache_path, limit=limit))
            result = await runner.run(meta_model=meta_model, object_model=object_model, cache_path=cache_path, limit=limit)
            gathered.append(result)

    # todo: do we really want to run all of these at the same time? lol
    # gathered = await coorountines_to_run.gather()
    # we get a list of lists, so we flatten it
    flattened: Slist[OtherEvalCSVFormat] = gathered.flatten_list()
    return flattened


def eval_dict_to_runner(eval_dict: dict[str, str]) -> Sequence[Type[OtherEvalRunner]]:
    runners = []
    for key, value in eval_dict.items():
        match key, value:
            case "biased_evals", "are_you_sure":
                runners.append(AreYouSureRunner)
            case _, _:
                raise ValueError(f"Unexpected evaluation type: {key=} {value=}")
    return runners



async def main():
    eval_dict = {
        "biased_evals": "are_you_sure",
    }
    models = Slist(["gpt-3.5-turbo", "claude-3-sonnet-20240229"])
    setup_environment()
    object_and_meta_models: Slist[tuple[str, str]] = models.product(models)
    study_folder = EXP_DIR / "other_evals"
    limit = 200
    evals_to_run = eval_dict_to_runner(eval_dict)
    results = await run_from_commands(evals_to_run=evals_to_run, object_and_meta=object_and_meta_models, limit=limit, study_folder=study_folder)
    dicts = [result.model_dump() for result in results]
    df = pd.DataFrame(dicts)
    df.to_csv(study_folder / "other_evals_results.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())