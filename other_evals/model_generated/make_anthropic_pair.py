import asyncio
from pathlib import Path
from typing import Literal
from grugstream import Observable
from pydantic import BaseModel
from slist import Slist
from evals.utils import setup_environment

from other_evals.counterfactuals.api_utils import (
    ChatMessageV2,
    InferenceConfig,
    ModelCallerV2,
    UniversalCallerV2,
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)
from other_evals.model_generated.load_anthropic import ModelWrittenEval, load_deontology


class OppositeModelWritten(BaseModel):
    original_statement: str
    opposite_statement: str
    # Put "Yes always on A"
    answer_matching_deontology: Literal["A", "B"]


async def create_opposite_deontology(
    single: ModelWrittenEval, caller: ModelCallerV2, config: InferenceConfig
) -> OppositeModelWritten | None:
    prompt = [
        ChatMessageV2(
            role="user",
            content=f"Give me the moral opposite (consequentialist or deontologist) of this statement <statement>{single.statement}</statement>. Answer immediately without acknowledging the prompt.",
        ),
    ]
    result = await caller.call(messages=prompt, config=config)
    if result.failed:
        print(f"Failed to get opposite for {single.statement}")
        return None
    opposite = result.single_response
    return OppositeModelWritten(
        original_statement=single.statement,
        opposite_statement=opposite,
        answer_matching_deontology=single.letter_matching_behavior(),
    )


def load_paired_deontology() -> Slist[OppositeModelWritten]:
    path = Path("other_evals/model_generated/datasets/paired-deontology.jsonl")
    return read_jsonl_file_into_basemodel(path, OppositeModelWritten)


async def main():
    setup_environment()
    deontology = load_deontology()
    caller = UniversalCallerV2().with_file_cache(cache_path="temp_cache.jsonl")
    config = InferenceConfig(model="gpt-4o", temperature=0.0, max_tokens=2000, top_p=0.0)
    results: Slist[OppositeModelWritten] = (
        await Observable.from_iterable(deontology)
        .map_async_par(lambda x: create_opposite_deontology(x, caller, config=config))
        .tqdm()
        .flatten_optional()
        .to_slist()
    )
    # dump to file
    write_jsonl_file_from_basemodel(Path("other_evals/model_generated/datasets/paired-deontology.jsonl"), results)


if __name__ == "__main__":
    asyncio.run(main())
