from pathlib import Path

import openai
from evals.apis.finetuning.run import FineTuneHyperParams, FineTuneParams, run_finetune
from evals.apis.finetuning.syncer import WandbSyncer
from evals.apis.inference.api import InferenceAPI
from evals.locations import EXP_DIR
from evals.utils import load_secrets, setup_environment
from other_evals.counterfactuals.api_utils import read_jsonl_file_into_basemodel, write_jsonl_file_from_basemodel
from other_evals.counterfactuals.inference_api_cache import CachedInferenceAPI
from other_evals.counterfactuals.other_eval_csv_format import FinetuneConversation
from other_evals.counterfactuals.runners import ALL_EVAL_TYPES, OtherEvalRunner


from git import Sequence
from slist import Slist
import asyncio


from typing import Type

from other_evals.counterfactuals.yaml_compat_utils import read_model_id_from_model_config


async def get_finetuning_samples(
    evals_to_run: Sequence[Type[OtherEvalRunner]],
    object_model: str,
    api: CachedInferenceAPI,
    # Not all samples are successsful, and its not always a 50/50 balanced dataset. Because we balance, often you get 20% of the samples you ask for.
    try_n_samples: int = 500,
    # The maximum amount of samples to take from each eval.
    take_n_samples: int | None = 50,
) -> Slist[FinetuneConversation]:
    """Run the appropriate evaluation based on the dictionary"""
    gathered = Slist()
    for runner in evals_to_run:
        result = await runner.get_finetuning(
            object_model=object_model,
            limit=try_n_samples,
            api=api,
        )
        if take_n_samples is not None:
            result = Slist(result).shuffle("42").take(take_n_samples)
        print(f"Got {len(result)} samples for {runner.name()}")
        gathered.append(result)
    flattened: Slist[FinetuneConversation] = gathered.flatten_list()
    return flattened


def get_other_evals_finetuning_samples(
    evals_to_run: Sequence[Type[OtherEvalRunner]],
    object_model_config: str,
    # Not all samples are successsful, and its not always a 50/50 balanced dataset. Because we balance, often you get 20% of the samples you ask for.
    try_n_samples: int = 500,
    # The maximum amount of samples to take from each eval.
    limit_per_eval: int | None = 50,
    cache_path: str | Path = EXP_DIR / "other_evals" / "cache",
) -> Slist[FinetuneConversation]:
    # entry point from finetuning where we create the inferenceapi ourselves
    # sync function because the entry point is sync
    setup_environment()
    api = InferenceAPI(anthropic_num_threads=40)
    model_id = read_model_id_from_model_config(object_model_config)
    inference_api = CachedInferenceAPI(api=api, cache_path=cache_path)
    cooroutine = get_finetuning_samples(
        evals_to_run=evals_to_run,
        object_model=model_id,
        api=inference_api,
        try_n_samples=try_n_samples,
        take_n_samples=limit_per_eval,
    )
    return asyncio.run(cooroutine)


def add_new_samples_to_existing_jsonl_and_shuffle(
    existing_jsonl_path: Path,
    new_jsonl_path: Path,
    new_samples: Sequence[FinetuneConversation],
) -> None:
    existing_samples = (
        read_jsonl_file_into_basemodel(existing_jsonl_path, basemodel=FinetuneConversation)
        .add(Slist(new_samples))
        .shuffle("42")
    )
    write_jsonl_file_from_basemodel(new_jsonl_path, basemodels=existing_samples)


async def test_main():
    setup_environment()
    # the sweep ain't a async function so we use asyncio.run
    api = InferenceAPI(anthropic_num_threads=10)
    study_folder = "exp/finetuning"
    inference_api = CachedInferenceAPI(api=api, cache_path=Path(study_folder) / "cache")
    n_to_try = 30_000
    model = "gpt-3.5-turbo-0125"
    finetune_samples = await get_finetuning_samples(
        evals_to_run=ALL_EVAL_TYPES,
        object_model="gpt-3.5-turbo-0125",
        try_n_samples=n_to_try,
        take_n_samples=int(n_to_try * 0.1),
        api=inference_api,
    )
    print(f"Got {len(finetune_samples)} final finetuning samples")
    write_jsonl_file_from_basemodel("test_finetune_samples.jsonl", finetune_samples.shuffle("42"))
    # load secrets
    secrets = load_secrets("SECRETS")
    org = secrets["DEFAULT_ORG"]
    assert org is not None
    openai.organization = org
    syncer = WandbSyncer.create(
        project_name="introspection", notes="4 datasets gpt-3.5 datafinetuning on all other evals test"
    )

    hyper_params = FineTuneHyperParams(n_epochs=1, learning_rate_multiplier=1.6, batch_size=16)
    params = FineTuneParams(model=model, hyperparameters=hyper_params, seed=42)
    model_id = run_finetune(
        params=params,
        data_path=Path("test_finetune_samples.jsonl"),
        syncer=syncer,
        ask_to_validate_training=True,
        # val_data_path=val_data_path,
        organisation=org,
    )
    print(f"Model id is {model_id}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_main())
