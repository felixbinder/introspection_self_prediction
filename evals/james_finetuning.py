import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

from evals.apis.finetuning.run import FineTuneHyperParams, FineTuneParams, run_finetune
from evals.apis.finetuning.syncer import WandbSyncer
from evals.utils import load_secrets
from other_evals.counterfactuals.api_utils import write_jsonl_file_from_basemodel
from other_evals.counterfactuals.other_eval_csv_format import FinetuneConversation


def finetune_openai(
    model: str,
    notes: str,
    suffix: str,
    train_items: Sequence[FinetuneConversation],
    val_items: Sequence[FinetuneConversation],
    hyperparams: FineTuneHyperParams,
) -> str:
    # load secrets
    secrets = load_secrets("SECRETS")

    # Create a timestamp for the folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"train_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    # Write train and validation data to JSON files
    train_path = Path(folder_name) / "train.jsonl"
    val_path = Path(folder_name) / "val.jsonl"

    write_jsonl_file_from_basemodel(path=train_path, basemodels=train_items)
    write_jsonl_file_from_basemodel(path=val_path, basemodels=val_items)

    # Set up FineTuneParams
    params = FineTuneParams(
        model=model,
        hyperparameters=hyperparams,
        suffix=suffix,
    )

    # Set up WandbSyncer
    syncer = WandbSyncer.create(project_name="james-introspection", notes=notes)

    # Run fine-tuning
    return run_finetune(params=params, data_path=train_path, syncer=syncer, val_data_path=val_path)
