"""This file runs the evaluation suite on a list of models. It get's written into `exp/evaluation_suite`.

Example usage:
python scripts/run_evalulation_suite.py
--model_configs="gpt-3.5-turbo,finetuned/study_name/model_name"
--study_name="study_name"
"""

import subprocess
from pathlib import Path

from evals import create_finetuning_dataset_configs
from evals.locations import CONF_DIR, EXP_DIR, REPO_DIR
from other_evals.counterfactuals.runners import run_sweep_over_other_evals

EVAL_SUITE = {
    "number_triplets": ["identity", "is_even"],
    "english_words": ["identity", "first_character"],
    "wikipedia": ["identity", "first_character"],
    "daily_dialog": ["identity", "first_character"],
    "dear_abbie": ["first_word", "sentiment", "dear_abbie/sympathetic_advice"],
    # "mmlu": ["matches_target"]
    # "writing_stories": ["writing_stories/good_ending", "writing_stories/main_character_name"], # inside/outside, main char male/female,
    # "jailbreak": ["jailbreak/jailbreak"],
    # "bias"
}

DIVERGENT_STRINGS = {
    "number_triplets": "exp/evaluation_suite/model_divergent_strings_number_triplets.csv",
    "english_words": "exp/evaluation_suite/model_divergent_strings_english_words.csv",
    "wikipedia": "exp/evaluation_suite/model_divergent_strings_wikipedia.csv",
    "daily_dialog": "exp/evaluation_suite/model_divergent_strings_daily_dialog.csv",
    "dear_abbie": "exp/evaluation_suite/model_divergent_strings_dear_abbie.csv",
    "mmlu": "exp/evaluation_suite/model_divergent_strings_mmlu.csv",
    # "writing_stories": "exp/evaluation_suite/model_divergent_strings_writing_stories.csv",
    # "jailbreak": "exp/evaluation_suite/model_divergent_strings_jailbreak.csv",
    # "bias
}

# Run other evals are aren't in the repo's format. See ALL_EVAL_TYPES
# OTHER_EVALS = [BiasDetectAddAreYouSure, BiasDetectAreYouAffected, BiasDetectWhatAnswerWithout, KwikWillYouBeCorrect]
OTHER_EVALS = []


N_TRAIN = 500
N_EVAL = 500
STUDY_NAME = "evaluation_suite"
PROMPT = "minimal"


def run_inference_only(models):
    """Run the models on the evaluation only.
    This is useful to see how well an introspection-trained model is doing."""
    for model in models:
        for task, response_properties in EVAL_SUITE.items():
            print(f"Running evaluation suite on Model: {model}\tTask: {task}")
            # run object level val
            command = f"python {REPO_DIR}/evals/run_object_level.py study_name={STUDY_NAME} language_model={model} prompt='object_level'/{PROMPT} task={task} task.set=val limit={N_EVAL} strings_path={DIVERGENT_STRINGS[task]}"
            run_command(command)
            # run meta level val
            for response_property in response_properties:
                command = f"python {REPO_DIR}/evals/run_meta_level.py study_name={STUDY_NAME} language_model={model} prompt='meta_level'/{PROMPT} task={task} response_property={response_property} task.set=val limit={N_EVAL} strings_path={DIVERGENT_STRINGS[task]}"
                run_command(command)
    print("Done running evaluation suite.")


def run_get_floor_ceiling_for_untrained_models(models):
    """Calculates the floor and ceiling for untrained models."""
    ft_models = run_finetuning(models)
    run_finetuned_models_on_their_task(ft_models)


def run_finetuning(models, ft_overrides=""):
    """Runs the models on the evaluation while also finetuning them on each task separately.
    This is used to create the ceiling and floor for a none-finetuned model so that we can compare the effect of introspection-training.
    THIS IS EXPENSIVE AND SLOW.
    ONLY PASS NON-FINETUNED MODELS HERE."""
    finetuned_models = {}
    for model in models:
        for task, response_properties in EVAL_SUITE.items():
            if get_finetuned_model(model, task) is not None:
                print(f"Model {model} already finetuned on task {task}—we found a config file. Skipping.")
                continue
            ft_study_name = f"{STUDY_NAME}/{task}"
            # generate object level train
            print(f"Generating object level training data for Model: {model}\tTask: {task}")
            command = f"python {REPO_DIR}/evals/run_object_level.py study_name={STUDY_NAME} language_model={model} prompt='object_level'/{PROMPT} task={task} task.set=train limit={N_TRAIN}"
            dataset_folder = run_command(command)
            # generate object level val
            command = f"python {REPO_DIR}/evals/run_object_level.py study_name={STUDY_NAME} language_model={model} prompt='object_level'/{PROMPT} task={task} task.set=val limit={N_EVAL} strings_path={DIVERGENT_STRINGS[task]}"
            val_dataset_folder = run_command(command)
            # generate finetuning data
            finetuning_config_paths = []
            for response_property in response_properties:
                print(
                    f"Generating finetuning data for Model: {model}\tTask: {task}\tResponse Property: {response_property}"
                )
                finetuning_config_path = create_finetuning_dataset_configs.create_finetuning_dataset_config(
                    study_name=ft_study_name,
                    model_config=model,
                    task_config=task,
                    prompt_config=PROMPT,
                    response_property_config=response_property,
                    overrides="",
                    train_base_dir=dataset_folder,
                    val_base_dir=val_dataset_folder,
                    overwrite=True,
                )
                finetuning_config_paths.append(finetuning_config_path)
            # create finetuning dataset
            command = f"python {REPO_DIR}/evals/create_finetuning_dataset.py study_name={ft_study_name} dataset_folder={model}"
            run_command(command)
            # run finetuning
            command = f"python {REPO_DIR}/evals/run_finetuning.py study_name={ft_study_name}/{model} language_model={model} notes=ES{task} {ft_overrides}"
            new_model_id = run_command(command)
            try:
                finetuned_models[task].append(new_model_id)
            except KeyError:
                finetuned_models[task] = [new_model_id]
    print("Done running evaluation suite and finetuning.")
    return finetuned_models


def run_finetuned_models_on_their_task(model: str):
    """Takes the input of `run_finetuning` and runs the finetuned models on the evaluation suite.
    This only runs each finetuned models on the task and response properties it was trained for.
    """
    for task, response_properties in EVAL_SUITE.items():
        # get finetuned model
        ft_model = get_finetuned_model(model, task)
        if ft_model is None:
            print(f"Finetuned model based on {model} for task {task} not found. Skipping.")
            continue
        for response_property in response_properties:
            print(
                f"Running evaluation suite on Model: {ft_model}\tTask: {task}\tResponse Property: {response_property}"
            )
            # run object level val
            command = f"python {REPO_DIR}/evals/run_object_level.py study_name={STUDY_NAME} language_model={ft_model} prompt='object_level'/{PROMPT} task={task} task.set=val limit={N_EVAL} strings_path={DIVERGENT_STRINGS[task]}"
            run_command(command)
            # run meta level val
            command = f"python {REPO_DIR}/evals/run_meta_level.py study_name={STUDY_NAME} language_model={ft_model} prompt='meta_level'/{PROMPT} task={task} response_property={response_property} task.set=val limit={N_EVAL} strings_path={DIVERGENT_STRINGS[task]}"
            run_command(command)


def generate_model_divergent_string():
    """Run this to generate the model divergent strings in the evaluation suite folder.
    This should only need to be run once to populate `DIVERGENT_STRINGS`.
    This runs the models specified below and creates a .csv files that contain the strings for each task for which the response property of the object-level  diverges.
    """
    DIVERGENCE_MODELS = ["gpt-3.5-turbo", "claude-3-sonnet"]
    for task, _ in EVAL_SUITE.items():
        folders = []
        # generate object level validations
        for model in DIVERGENCE_MODELS:
            print(f"Generating object level validation for task: {task}")
            command = f"python {REPO_DIR}/evals/run_object_level.py study_name={STUDY_NAME} language_model={model} prompt='object_level'/{PROMPT} task={task} task.set=val limit={N_EVAL * 3}"
            folder = run_command(command)
            folders.append(folder)
        # compute model divergenced strings
        print(f"Generating model divergent strings for task: {task}")
        filename = Path(f"{REPO_DIR}/exp/{STUDY_NAME}/model_divergent_strings_{task}.csv")
        if filename.exists():
            print(f"File {filename} already exists. Skipping.")
            continue
        # run the extraction
        command = f"python {REPO_DIR}/evals/extract_model_divergent_strings.py {' '.join(folders)} --output {filename} --response_properties {' '.join(EVAL_SUITE[task])}"
        run_command(command)
    print("Done generating model divergent strings.")


def run_command(command):
    """Execute the given command in the shell, stream the output, and return the last line."""
    print(f"Executing: {command}")
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        output_lines = []
        for line in process.stdout:
            print(line, end="")  # stream the output to the command line
            output_lines.append(line.strip())

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        last_line = output_lines[-1] if output_lines else ""
        print(f"Successfully executed: {command}")
        return last_line

    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e}")
        raise e


def get_finetuned_model(model: str, task: str):
    """Returns the finetuned model for the given model, task, and response property. Return None if not found."""
    path = Path(CONF_DIR) / "language_model" / "finetuned" / STUDY_NAME / task / model
    if not path.exists():
        return None
    # is there a single yaml file?
    yaml_files = list(path.glob("*.yaml"))
    if len(yaml_files) == 1:
        return yaml_files[0].relative_to(CONF_DIR / "language_model").with_suffix("")
    elif len(yaml_files) > 1:
        raise ValueError(f"Multiple yaml files found in {path}. Expected only one.")
    elif len(yaml_files) == 0:
        return None


if __name__ == "__main__":
    # which non-fted models?
    models = ["gpt-3.5-turbo", "claude-3-sonnet-20240229"]
    # generate divergent strings from fixed models
    generate_model_divergent_string()
    # run the evaluation suite
    run_inference_only(models)
    # finetune the model for each task individually
    run_finetuning(models)
    # see how well the finetuned models do on their task
    for model in models:
        run_finetuned_models_on_their_task(model)
    run_get_floor_ceiling_for_untrained_models(models)

    # Run the other evals every combination of object and meta models
    object_and_meta_models = [(object_model, meta_model) for object_model in models for meta_model in models]
    if OTHER_EVALS:
        run_sweep_over_other_evals(
            object_and_meta=object_and_meta_models,
            eval_list=OTHER_EVALS,
            limit=N_EVAL,
            show_plot=True,
            study_folder=EXP_DIR / STUDY_NAME,
        )
