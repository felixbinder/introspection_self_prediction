"""This file runs the evaluation suite on a list of models. It get's written into `exp/evaluation_suite`.

Example usage:
python scripts/run_evalulation_suite.py
--model_configs="gpt-3.5-turbo,finetuned/study_name/model_name"
--study_name="study_name"
"""

import subprocess
from pathlib import Path

from evals.locations import REPO_DIR

EVAL_SUITE = {
    "number_triplets": ["identity", "is_even"],
    "wealth_seeking": ["identity", "matches_wealth_seeking"],
    "power_seeking": ["identity", "matches_power_seeking"],
    "survival_instinct": ["identity", "matches_survival_instinct"],
    # "english_words": ["identity", "first_character"],
    # "wikipedia": ["identity", "first_character"],
    # "daily_dialog": ["identity", "first_character"],
    # "dear_abbie": ["first_word", "sentiment", "dear_abbie/sympathetic_advice"],
    # "mmlu": ["matches_target"]
    # "writing_stories": ["writing_stories/good_ending", "writing_stories/main_character_name"], # inside/outside, main char male/female,
    # "jailbreak": ["jailbreak/jailbreak"],
    # "bias"
}


# Run other evals are aren't in the repo's format. See ALL_EVAL_TYPES
# OTHER_EVALS = [BiasDetectAddAreYouSure, BiasDetectAreYouAffected, BiasDetectWhatAnswerWithout, KwikWillYouBeCorrect]
OTHER_EVALS = []


N_TRAIN = 50
N_EVAL = 100
STUDY_NAME = "evaluation_suite"
PROMPT = "minimal"

DIVERGENT_STRINGS = {
    "number_triplets": "exp/evaluation_suite/model_divergent_strings_number_triplets.csv",
    "wealth_seeking": "exp/evaluation_suite/model_divergent_strings_wealth_seeking.csv",
    "power_seeking": "exp/evaluation_suite/model_divergent_strings_power_seeking.csv",
    "survival_instinct": "exp/evaluation_suite/model_divergent_strings_survival_instinct.csv",
    # "english_words": "exp/evaluation_suite/model_divergent_strings_english_words.csv",
    # "wikipedia": "exp/evaluation_suite/model_divergent_strings_wikipedia.csv",
    # "daily_dialog": "exp/evaluation_suite/model_divergent_strings_daily_dialog.csv",
    # "dear_abbie": "exp/evaluation_suite/model_divergent_strings_dear_abbie.csv",
    # "mmlu": "exp/evaluation_suite/model_divergent_strings_mmlu.csv",
    # "writing_stories": "exp/evaluation_suite/model_divergent_strings_writing_stories.csv",
    # "jailbreak": "exp/evaluation_suite/model_divergent_strings_jailbreak.csv",
    # "bias
}


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


if __name__ == "__main__":
    # which non-fted models?
    models = ["gpt-3.5-turbo", "claude-3-sonnet"]
    # run the evaluation suite
    generate_model_divergent_string()
    run_inference_only(models)
