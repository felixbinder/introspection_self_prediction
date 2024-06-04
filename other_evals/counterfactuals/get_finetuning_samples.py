from pathlib import Path

import openai
from evals.apis.finetuning.run import FineTuneHyperParams, FineTuneParams, run_finetune
from evals.apis.finetuning.syncer import WandbSyncer
from evals.apis.inference.api import InferenceAPI
from evals.utils import load_secrets, setup_environment
from other_evals.counterfactuals.api_utils import write_jsonl_file_from_basemodel
from other_evals.counterfactuals.inference_api_cache import CachedInferenceAPI
from other_evals.counterfactuals.other_eval_csv_format import FinetuneConversation
from other_evals.counterfactuals.runners import ALL_EVAL_TYPES, OtherEvalRunner


from git import Sequence
from slist import Slist


from typing import Type


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
        gathered.append(result)
    flattened: Slist[FinetuneConversation] = gathered.flatten_list()
    return flattened


async def test_main():
    setup_environment()
    # the sweep ain't a async function so we use asyncio.run
    api = InferenceAPI()
    study_folder = "exp/finetuning"
    inference_api = CachedInferenceAPI(api=api, cache_path=Path(study_folder) / "cache")
    n_to_try = 3000
    model = "gpt-3.5-turbo-0125"
    finetune_samples = await get_finetuning_samples(
        evals_to_run=ALL_EVAL_TYPES,
        object_model=model,
        try_n_samples=n_to_try,
        take_n_samples=int(n_to_try * 0.2),
        api=inference_api,
    )
    print(f"Got {len(finetune_samples)} final finetuning samples")
    write_jsonl_file_from_basemodel("test_finetune_samples.jsonl", finetune_samples.shuffle("42"))
    # load secrets
    secrets = load_secrets("SECRETS")
    org = secrets["OWAIN_ORG"]
    assert org is not None
    openai.organization = org
    syncer = WandbSyncer.create(project_name="introspection", notes="finetuning on all other evals test")

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



if __name__ == "__main__":
    import asyncio

    asyncio.run(test_main())
