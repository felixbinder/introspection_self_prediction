import logging
import os
import random
import subprocess
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from evals.apis.finetuning.run import FineTuneHyperParams, FineTuneParams, run_finetune
from evals.apis.finetuning.syncer import WandbSyncer
from evals.locations import CONF_DIR
from evals.utils import (
    COMPLETION_MODELS,
    GEMINI_MODELS,
    GPT_CHAT_MODELS,
    get_current_git_hash,
    load_secrets,
    safe_model_name,
    setup_environment,
)

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent

FT_CONFIG_TEMPLATE = """
model: {model}
cais_path: {cais_path}
temperature: 0.0
top_p: 1.0
max_tokens: null
num_candidates_per_completion: 1
insufficient_valids_behaviour: "error"
"""


@hydra.main(config_path="conf", config_name="config_finetuning_run")
def main(cfg: DictConfig) -> str:
    print("Current git hash:", get_current_git_hash())
    assert " " not in cfg.notes, "Notes cannot have spaces, use underscores instead"
    # set working directory
    os.chdir(ROOT_DIR)

    setup_environment(openai_tag=cfg.openai_tag)
    params = FineTuneParams(
        model=cfg.language_model.model,
        hyperparameters=FineTuneHyperParams(
            n_epochs=cfg.epochs, learning_rate_multiplier=cfg.learning_rate, batch_size=cfg.batch_size
        ),
        suffix=cfg.notes,
        seed=cfg.seed,
    )

    # try to find the data files
    # defaults to cfg.study_dir / "train_dataset.jsonl"
    data_path = Path(cfg.train_path)
    val_data_path = Path(cfg.val_path)
    if cfg.language_model.model in GEMINI_MODELS:
        assert "-format_gemini" in str(data_path), "Path should be pointing at gemini formatted dataset."
        assert "-format_gemini" in str(val_data_path), "Path should be pointing at gemini formatted dataset."
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    if not val_data_path.exists():
        val_data_path = None
        LOGGER.warning(f"Validation file not found at {val_data_path}. Continuing without validation.")

    # load secrets
    secrets = load_secrets("SECRETS")
    try:
        org = secrets[cfg.organization]
    except KeyError:
        LOGGER.error(f"Organization {cfg.organization} not found in secrets")
        raise
    if params.model in (COMPLETION_MODELS | GPT_CHAT_MODELS | {"gemini-1.0-pro-002"}) or str(
        params.model
    ).lower().startswith("ft:gpt"):
        if cfg.use_wandb:
            syncer = WandbSyncer.create(project_name=safe_model_name(str(cfg.study_name))[0:127], notes=cfg.notes)
            # if more_config:
            #     more_config = {k: v for k, v in [x.split("=") for x in more_config.split(",")]}
            #     syncer.update_parameters_with_dict(params=more_config)
        else:
            syncer = None

        model_id = run_finetune(
            params=params,
            data_path=data_path,
            syncer=syncer,
            ask_to_validate_training=cfg.ask_to_validate_training,
            val_data_path=val_data_path,
            organisation=org,
        )
        # add dummy save path
        save_path = "~"

    else:
        LOGGER.info("Running HF finetuning")
        run_name = cfg.language_model.model + "_finetuned_" + cfg.notes
        save_path = f"{cfg.study_dir}/{run_name}"
        num_gpus = torch.cuda.device_count()
        if cfg.language_model.model == "llama-3-70b-instruct":
            batch_size = 64
            lr = 5e-4
            n_epochs = 5
        else:
            batch_size = cfg.batch_size or 32
            lr = cfg.learning_rate or 1e-3
            n_epochs = cfg.epochs or 5
        lora_rank = cfg.lora_rank or 8
        port = random.randint(10000, 20000)
        gradient_accumulation_steps = cfg.gradient_accumulation_steps or 8
        cmd = f"""accelerate launch --config_file evals/conf/accelerate_config.yaml \
--mixed_precision bf16 \
--main_process_port {port} \
--num_processes {num_gpus} \
--gradient_accumulation_steps {gradient_accumulation_steps} \
-m evals.apis.finetuning.hf_finetuning \
--config evals/conf/trl_config.yaml \
--output_dir {save_path} \
--run_name {run_name} \
--model_name_or_path {cfg.language_model.cais_path} \
--dataset_name {cfg.study_dir} \
--per_device_train_batch_size {(batch_size//num_gpus)//gradient_accumulation_steps} \
--gradient_accumulation_steps {gradient_accumulation_steps} \
--learning_rate {lr} \
--num_train_epochs {n_epochs} """
        if lora_rank is not None:
            cmd += f"--use_peft --lora_r={lora_rank} --lora_alpha=16"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output_lines = []
        for line in process.stdout:
            print(line, end="")  # stream the output to the command line
            output_lines.append(line.strip())
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        print(f"Successfully executed: {cmd}")
        model_id = run_name
        if lora_rank is not None:
            cmd = f"""python evals/apis/finetuning/merge_peft_adapter.py --adapter_model_name {save_path} --base_model_name {cfg.language_model.cais_path} --output_name {save_path}_merged"""
            save_path += "_merged"
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            output_lines = []
            for line in process.stdout:
                print(line, end="")  # stream the output to the command line
                output_lines.append(line.strip())
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
            print(f"Successfully executed: {cmd}")

    LOGGER.info(f"Done with model_id: {model_id}")
    # adding a config file for this finetuned model
    config_id = create_finetuned_model_config(cfg, model_id, cais_path=save_path)
    print(config_id)
    return model_id


def create_finetuned_model_config(cfg, ft_model_id, cais_path="~", overwrite=True):
    """Creates a model config file for the finetuned model in the config directory."""
    assert ft_model_id != "", "Model ID cannot be empty"
    safe_model_id = ft_model_id.replace(":", "_").replace("/", "_")
    directory = CONF_DIR / "language_model" / "finetuned" / cfg.study_name
    file_path = directory / f"{safe_model_id}.yaml"
    directory.mkdir(parents=True, exist_ok=True)
    if file_path.exists() and not overwrite:
        LOGGER.warning(f"File already exists at {file_path}. Not overwriting.")
        return f"finetuned/{cfg.study_name}/{safe_model_id}"
    with open(directory / file_path, "w") as f:
        f.write(FT_CONFIG_TEMPLATE.format(model=ft_model_id, cais_path=cais_path))
    return f"finetuned/{cfg.study_name}/{safe_model_id}"  # return the name of the config that can be loaded by Hydra


if __name__ == "__main__":
    main()
