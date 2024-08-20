"""This file holds helper functions to create the Hydra files for the finetuning dataset configs by sweeping over the different configurations."""

import argparse
from pathlib import Path

from evals.locations import EXP_DIR
from evals.utils import safe_model_name

TEMPLATE = """
name: $name

defaults: # we need to use defaults here
  - task: $task
  - prompt: $prompt
  - response_property: $response_property

train_base_dir: $train_base_dir

val_base_dir: $val_base_dir
enforce_unique_strings: $enforce_unique_strings

$overrides

"""


def create_finetuning_dataset_config(
    study_name: str,
    model_config: str,
    task_config: str,
    prompt_config: str,
    response_property_config: str,
    overrides: str,
    train_base_dir: str,
    val_base_dir: str,
    overwrite: bool = True,
    enforce_unique_strings: bool = True,
) -> Path:
    model_config = safe_model_name(model_config)
    # Finetuning folder—organized by source model. The folder then contains many different tasks etc.
    ft_exp_dir = EXP_DIR / "finetuning" / study_name / model_config
    ft_exp_dir.mkdir(parents=True, exist_ok=True)

    name = f"{model_config.replace('/', '-')}_{task_config.replace('/', '-')}_{response_property_config.replace('/', '-')}_{prompt_config.replace('/', '-')}"  # name of the config. We need to replace the / in the prompt config to avoid issues with the file path.

    if isinstance(overrides, list):
        overrides_str = "\n".join(overrides)
    else:
        overrides_str = overrides

    if "meta_level/" not in prompt_config:  # we need to load the meta level prompt
        prompt_config = f"meta_level/{prompt_config}"

    config = TEMPLATE.replace("$name", name)
    config = config.replace("$task", task_config)
    config = config.replace("$prompt", prompt_config)
    config = config.replace("$response_property", response_property_config)
    config = config.replace("$train_base_dir", train_base_dir)
    config = config.replace("$val_base_dir", val_base_dir)
    config = config.replace("$overrides", overrides_str)
    config = config.replace("$enforce_unique_strings", str(enforce_unique_strings).lower())

    # save to file
    config_path = ft_exp_dir / f"{name}.yaml"
    if config_path.exists() and not overwrite:
        print(f"File already exists at {config_path}. Not overwriting.")
        return config_path
    with open(config_path, "w") as f:
        f.write(config)
    return config_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--task_config", type=str, required=True)
    parser.add_argument("--prompt_config", type=str, required=True)
    parser.add_argument("--response_property_config", type=str, required=True)
    parser.add_argument("--train_base_dir", type=str, required=True)
    parser.add_argument("--val_base_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overrides", type=str, nargs="+", required=False, default=[])
    args = parser.parse_args()

    path = create_finetuning_dataset_config(
        args.study_name,
        args.model_config,
        args.task_config,
        args.prompt_config,
        args.response_property_config,
        args.overrides,
        args.train_base_dir,
        args.val_base_dir,
        args.overwrite,
    )
    print(f"Created config at {path}")
