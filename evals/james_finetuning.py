import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

from evals.apis.finetuning.run import FineTuneHyperParams, FineTuneParams, run_finetune
from evals.apis.finetuning.syncer import WandbSyncer
from evals.create_finetuning_dataset import LOGGER
from evals.locations import CONF_DIR
from evals.run_finetuning import FT_CONFIG_TEMPLATE
from evals.utils import setup_environment
from other_evals.counterfactuals.api_utils import write_jsonl_file_from_basemodel
from other_evals.counterfactuals.other_eval_csv_format import FinetuneConversation


def create_model_config(study_name: str, ft_model_id: str, cais_path="~", overwrite=True) -> str:
    """Creates a model config file for the finetuned model in the config directory."""
    assert ft_model_id != "", "Model ID cannot be empty"
    safe_model_id = ft_model_id.replace(":", "_").replace("/", "_")
    directory = CONF_DIR / "language_model" / "finetuned" / study_name
    file_path = directory / f"{safe_model_id}.yaml"
    directory.mkdir(parents=True, exist_ok=True)
    if file_path.exists() and not overwrite:
        LOGGER.warning(f"File already exists at {file_path}. Not overwriting.")
        return f"finetuned/{study_name}/{safe_model_id}"
    with open(directory / file_path, "w") as f:
        f.write(FT_CONFIG_TEMPLATE.format(model=ft_model_id, cais_path=cais_path))
    return f"finetuned/{study_name}/{safe_model_id}"  # return the name of the config that can be loaded by Hydra


def finetune_openai(
    model: str,
    notes: str,
    suffix: str,
    train_items: Sequence[FinetuneConversation],
    val_items: Sequence[FinetuneConversation],
    hyperparams: FineTuneHyperParams,
) -> str:
    # load secrets
    setup_environment()

    # Create a timestamp for the folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"train_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    # Write train and validation data to JSON files
    train_path = Path(folder_name) / "train.jsonl"
    val_path = Path(folder_name) / "val.jsonl"

    write_jsonl_file_from_basemodel(path=train_path, basemodels=train_items)
    if val_items:
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
    return run_finetune(
        params=params, data_path=train_path, syncer=syncer, val_data_path=val_path if val_items else None
    )
