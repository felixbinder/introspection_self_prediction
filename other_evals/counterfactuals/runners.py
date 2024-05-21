from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
from typing import Awaitable
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
        number_samples = limit * 10  
        result = await run_single_are_you_sure(
            object_model=object_model,
            meta_model=meta_model,
            cache_path=cache_path,
            number_samples=number_samples,
        )
        formatted = result.take(limit).map(
            lambda x: x.to_other_eval_format(eval_name="are_you_sure")
        )
        return formatted


async def run_from_commands(
    eval_dict: dict[str, str], meta_model: str, object_model: str, limit: int, study_folder: str | Path,
) -> Sequence[OtherEvalCSVFormat]:
    """Run the appropriate evaluation based on the dictionary"""
    coorountines_to_run: Slist[Awaitable[Sequence[OtherEvalCSVFormat]]] = Slist()
    cache_path = Path(study_folder) / "model_cache.jsonl"
    for key, value in eval_dict.items():
        match key, value:
            case "biased_evals", "are_you_sure":
                coroutine = AreYouSureRunner.run(meta_model, object_model, cache_path, limit)
                coorountines_to_run.append(coroutine)
            case _, _:
                raise ValueError(f"Unexpected evaluation type: {key=} {value=}")
    # asyncio.run but typed correctly
    gathered = await coorountines_to_run.par_map_async(lambda func: func)
    # we get a list of lists, so we flatten it
    flattened: Slist[OtherEvalCSVFormat] = gathered.flatten_list()
    return flattened


async def main():
    eval_dict = {
        "biased_evals": "are_you_sure",
    }
    meta_model = "gpt-3.5-turbo"
    object_model = "gpt-3.5-turbo"
    study_folder = EXP_DIR / "other_evals" / "other_evals"
    limit = 50
    setup_environment()
    results = await run_from_commands(eval_dict=eval_dict, meta_model=meta_model, object_model=object_model, limit=limit, study_folder=study_folder)
    dicts = [result.model_dump() for result in results]
    df = pd.DataFrame(dicts)
    df.to_csv(study_folder / "results.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())