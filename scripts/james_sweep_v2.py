"""
James modification of the sweep
namely
- no divergent strings
- no enforcement of unique strings when training (theres really no need to, and theres a bug where it'll sample more for later response properties
- Max 2 processes so you don't DDOS openai

- generate `10000` object level completions with `train`
        - to be used as the training set for finetuning
- generate `2500` object level completions with `val`
        - for comparing the object/meta level
        - as validation set during finetuning
        - overgenerating to find model disagreement here
- generate finetuning datasets, sweeping across `response_property`, `task`, other args(?)
        - using the above directories
        - sweeping across other configs (response property, task)
        - => new model codes need to be kept track of
- run finetuning
- generate `500` meta-level completions on `val`, sweeping across `response_property`, `task`
        - use `2500` val generated ones to do model_divergence filtering down to 500
        - this also needs to include the newly generated models from above

Since not all response properties make sense for all tasks, we pass a list of response properties for every task as a JSON string. The name of the task is the key and the list of response properties is the value.

Example usage:
```bash
python -m scripts.sweep_full_study
--study_name="full_sweep_demo"
--model_configs="gpt-3.5-turbo"
--val_only_model_configs="gpt-4"
--doubly_trained_model_configs='{"finetuned/jun20_training_on_everything/gpt-3.5-turbo/ft_gpt-3.5-turbo-0125_dcevals-kokotajlo__9da15ENS": ["gpt-3.5-turbo","gpt-4"]}'
--tasks='{"wikipedia": ["identity", "sentiment"], "dear_abbie": ["identity", "sentiment", "dear_abbie/sympathetic_advice"]}'
--val_tasks='{"number_triplets": ["identity", "is_even"], "english_words": ["identity", "first_character"]}'
--other_evals='["BiasDetectAddAreYouSure", "BiasDetectAreYouAffected", "BiasDetectWhatAnswerWithout", "KwikWillYouBeCorrect"]'
--val_other_evals='["BiasDetectAddAreYouSure", "BiasDetectAreYouAffected", "BiasDetectWhatAnswerWithout", "KwikWillYouBeCorrect"]'
--prompt_configs='minimal'
--n_object_train=1000
--n_object_val=250
--n_meta_val=50
--skip_finetuning
--skip_finetuned_models
--finetuning_overrides='{"gpt-3.5-turbo":{"epochs":1,"learning_rate":5,"batch_size":5},"gpt-4":{"epochs":1,"learning_rate":5,"batch_size":5},"gemini-1.0-pro":{"epochs":1}}'
--inference_overrides='{"gpt-3.5-turbo":{"n_samples":5},"gpt-4":{"n_samples":5}'
```
"""

import argparse
import atexit
import json
import subprocess
from collections import defaultdict
from functools import partial
from multiprocessing import Manager, Pool, managers
from pathlib import Path
from typing import Dict, Sequence, Type, Union

from evals.apis.finetuning.run import FineTuneHyperParams
from evals.james_create_dataset import GeneratedDataset, james_make_finetuning
from evals.james_finetuning import create_model_config, finetune_openai
from evals.locations import EXP_DIR
from evals.utils import get_current_git_hash
from other_evals.counterfactuals.get_finetuning_samples import (
    get_other_evals_finetuning_samples,
)
from other_evals.counterfactuals.other_eval_csv_format import FinetuneConversation
from other_evals.counterfactuals.runners import (
    OtherEvalRunner,
    eval_list_to_runner,
    run_sweep_over_other_evals,
)
from other_evals.counterfactuals.yaml_compat_utils import (
    read_model_id_from_model_config,
)


def json_string(arg_value):
    """Attempt to parse a JSON string, raise an error if parsing fails."""
    try:
        return json.loads(arg_value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"The argument must be a valid JSON string: {e}")


def combine_dicts_of_lists(dicts: list[Dict]):
    out_dict = {}
    for cur_dict in dicts:
        for key in cur_dict:
            if key not in out_dict:
                out_dict[key] = cur_dict[key]
                out_dict[key] = list(set(out_dict[key] + cur_dict[key]))
    return out_dict


class StudyRunner:
    def __init__(self):
        self.parse_arguments()
        self.parse_args_into_lists_and_dicts()  # Updated to handle JSON strings
        self.manager = Manager()
        self.state = self.manager.dict()
        self.state["ft_configs"] = self.manager.list()
        self.state_lock = self.manager.Lock()
        # We validate the other evals here, so that we raise an error if the user tries to run an eval that doesn't exist
        # We don't overwrite self.args.other_evals because we want to keep the original string for the state file
        self.validated_other_evals = validate_other_evals(self.args.other_evals)
        self.validated_val_other_evals = validate_other_evals(self.args.val_other_evals)
        self.load_or_create_state_file()
        atexit.register(self.write_state_file)

    # Updated to parse JSON string arguments into dictionaries
    def parse_args_into_lists_and_dicts(self):
        string_args = [
            "model_configs",
            "val_only_model_configs",
            "prompt_configs",
            "skip_finetuning_for_models",
        ]
        dict_args = ["tasks", "val_tasks", "doubly_trained_model_configs"]
        if getattr(self.args, "finetuning_overrides") and getattr(self.args, "finetuning_overrides").strip().startswith(
            "{"
        ):
            dict_args.append("finetuning_overrides")
        else:
            string_args.append("finetuning_overrides")

        if getattr(self.args, "inference_overrides") and getattr(self.args, "inference_overrides").strip().startswith(
            "{"
        ):
            dict_args.append("inference_overrides")
        else:
            string_args.append("inference_overrides")

        for arg in string_args:
            setattr(
                self.args,
                arg,
                [x.strip() for x in getattr(self.args, arg).split(",")] if getattr(self.args, arg) else [],
            )
        # Handling JSON string arguments for tasks and validation tasks
        for arg in dict_args:
            if getattr(self.args, arg):
                setattr(self.args, arg, json_string(getattr(self.args, arg)))
            else:
                setattr(self.args, arg, {})

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Run a full study sweeping over the following configs.")
        parser.add_argument("--study_name", type=str, help="The name of the study. Defines the output directory.")
        parser.add_argument(
            "--model_configs", type=str, help="Comma-separated list of model configurations to sweep over."
        )
        parser.add_argument(
            "--val_only_model_configs",
            type=str,
            help="Comma-separated list of model configurations for validation only.",
            default="",
        )
        parser.add_argument(
            "--doubly_trained_model_configs",
            type=str,
            help="JSON string of doubly trained model configurations. e.g. {'ft_model_config': ['model1', 'model2']}. We only train these combinations rather than all possible combinations.",
            default="{}",
        )
        parser.add_argument("--tasks", type=str, help="JSON string of tasks configuration")
        parser.add_argument("--val_tasks", type=str, help="JSON string of validation tasks configuration", default="{}")
        parser.add_argument(
            "--other_evals",
            type=str,
            help="List of other evals to train on. e.g. ['BiasDetectAddAreYouSure']. See ALL_EVAL_TYPES",
            default="[]",
        )
        parser.add_argument(
            "--val_other_evals",
            type=str,
            help="List of other evals to evaluate on. e.g. ['BiasDetectAddAreYouSure']. See ALL_EVAL_TYPES",
            default="[]",
        )
        parser.add_argument("--prompt_configs", type=str, help="Comma-separated list of prompt configurations.")
        parser.add_argument(
            "--inference_overrides", type=str, help="JSON formated list of Hydra configuration overrides.", default=""
        )
        parser.add_argument(
            "--finetuning_overrides",
            type=str,
            help="JSON formated list of Hydra configuration overrides. e.g. {'gpt-3.5-turbo': {'epochs': 1}}",
            default="{}",
        )
        parser.add_argument(
            "--n_object_train", type=int, help="Number of object level completions for training.", default=1000
        )
        parser.add_argument(
            "--n_object_val", type=int, help="Number of object level completions for validation.", default=100
        )
        parser.add_argument(
            "--n_finetuning", type=int, help="Number of finetuning completions to generate.", default=400
        )
        parser.add_argument(
            "--n_meta_val", type=int, help="Number of meta level completions for validation.", default=100
        )
        parser.add_argument("--skip_finetuning", action="store_true", help="Skip the finetuning step.", default=False)
        parser.add_argument(
            "--skip_finetuned_models", action="store_true", help="Do not run finetuned models.", default=False
        )
        parser.add_argument(
            "--skip_finetuning_for_models",
            type=str,
            help="Comma-separated list of models to skip finetuning for.",
            default="",
        )
        self.args = parser.parse_args()

    def run_command(self, command, n_try: int = 0):
        """Execute the given command in the shell, stream the output, and return the last line."""
        if n_try > 2:
            raise Exception(f"Failed to run {command} after {n_try} tries.")

        try:
            self.state["commands"].append(command)  # log the command
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            output_lines = []
            for line in process.stdout:
                print(line, end="")  # stream the output to the command line
                output_lines.append(line.strip())

            process.wait()

            # handle errors
            if process.returncode == 137:
                print(f"❌ Memory error (137) executing {command}. Retrying...")
                return self.run_command(command, n_try + 1)
            elif process.returncode == 139:
                print(f"❌ Segmentation fault (139) executing {command}. Retrying...")
                return self.run_command(command, n_try + 1)
            elif process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)

            last_line = output_lines[-1] if output_lines else ""
            print(f"✅ Successfully executed: {command}")
            return last_line

        except subprocess.CalledProcessError as e:
            print(f"❌ Error executing {command}: {e}")
            raise e

    def parse_args_into_lists(self):
        for arg in [
            "model_configs",
            "val_only_model_configs",
            "prompt_configs",
            "finetuning_overrides",
            "skip_finetuning_for_models",
        ]:
            setattr(
                self.args, arg, getattr(self.args, arg).replace(", ", ",").split(",") if getattr(self.args, arg) else []
            )

    def load_or_create_state_file(self):
        state_file_path = Path(EXP_DIR / self.args.study_name / "state.json")
        state_file_path.parent.mkdir(parents=True, exist_ok=True)
        if state_file_path.exists():
            with open(state_file_path, "r") as f:
                state_dict = json.load(f)
                state_dict = self.turn_nested_dictionary_into_multiprocessing_dict(state_dict)
                self.state.update(state_dict)
            print(f"Existing state file loaded from {state_file_path}")
        else:
            self.state.update(
                {
                    "args": vars(self.args),
                    "object_train_runs": self.manager.dict(),
                    "object_val_runs": self.manager.dict(),
                    "finetuning_dataset_creation": self.manager.dict(),
                    "finetuning_runs": self.manager.dict(),
                    "ft_object_val_runs": self.manager.dict(),
                    "meta_val_runs": self.manager.dict(),
                    "commands": self.manager.list(),
                    "current_git_hash": get_current_git_hash(),
                }
            )
            self.write_state_file()
            print(f"New state file created at {state_file_path}")

    def write_state_file(self):
        state_file = Path(EXP_DIR / self.args.study_name / "state.json")
        with self.state_lock:
            state_dict = self.turn_nested_multiprocessing_dict_into_normal_dict(self.state)
            with open(state_file, "w") as f:
                json.dump(dict(state_dict), f, indent=4)
        print(f"State file written to {state_file}")

    def get_folders_by_task(self, task, set="val", block="object_val_runs"):
        """Get the folders for the object level completions by task and set."""
        return [
            v["folder"]
            for k, v in self.state[block].items()
            if v["task"] == task and v["set"] == set and v["status"] == "complete"
        ]

    def get_finetuned_model_configs(self):
        # return []
        return list(self.state["ft_configs"])

    def get_object_level_command(self, model, task, prompt, limit, set, overrides=""):
        if isinstance(overrides, dict):
            if model not in overrides:
                overrides = ""
            else:
                overrides = overrides[model]
                overrides = [f"{k}={v}" for k, v in overrides.items()]
        overrides = "\n".join(overrides)
        command = f"python -m evals.run_object_level study_name={self.args.study_name} language_model={model} language_model.logprobs=1 task={task} task.set={set} prompt=object_level/{prompt} limit={limit} {overrides}"
        return command

    def get_meta_level_command(
        self, model, task, response_property, prompt, limit, set, strings_path="~", overrides=[]
    ):
        if isinstance(overrides, dict):
            if model not in overrides:
                overrides = ""
            else:
                overrides = overrides[model]
                overrides = [f"{k}={v}" for k, v in overrides.items()]
        overrides = "\n".join(overrides)
        command = f"python -m evals.run_meta_level study_name={self.args.study_name} language_model={model} task={task} response_property={response_property} task.set={set} prompt=meta_level/{prompt} limit={limit} strings_path={strings_path} {overrides}"

        return command

    def get_finetuning_command(
        self, model, ft_study, notes, val_path: Path, train_path: Path, overrides: Union[dict, str] = ""
    ):
        if isinstance(overrides, dict):
            if model not in overrides:  # use empty overrides if not provided for a model
                overrides = ""
            else:
                overrides = overrides[model]
                overrides = [f"{k}={v}" for k, v in overrides.items()]
        override_str = " ".join(overrides)
        return f"python -m evals.run_finetuning study_name={ft_study} train_path={train_path.as_posix()} val_path={val_path.as_posix()} language_model={model} notes={notes} {override_str}"

    def run_study(self):
        SHIFT_DATA: bool = False
        pool = Pool(1)  # No clue why there is a race condition on the same CSV wth?

        #### run object level completions on train ####
        object_train_commands = []
        with self.state_lock:
            for model in self.args.model_configs + list(self.args.doubly_trained_model_configs.keys()):
                for task in self.args.tasks.keys():
                    for prompt in self.args.prompt_configs:
                        command = self.get_object_level_command(model, task, prompt, self.args.n_object_train, "train")
                        # check if we need to run this command and set up the state
                        if command not in self.state["object_train_runs"]:
                            self.state["object_train_runs"].update(
                                self.turn_nested_dictionary_into_multiprocessing_dict(
                                    {command: {"status": "incomplete"}}
                                )
                            )
                        elif self.state["object_train_runs"][command]["status"] == "complete":
                            print(f"Skipping {command} because it is already complete.")
                        # save other args to the state file
                        self.state["object_train_runs"][command].update({"model": model, "task": task, "set": "train"})
                        object_train_commands.append(command)
        self.write_state_file()

        pool.map(partial(run_object_train_command, state=self.state, state_lock=self.state_lock), object_train_commands)
        self.write_state_file()

        #### run object level completions on val ####
        object_val_commands = []
        # including validation only models here for the divergence calculation
        with self.state_lock:
            for model in (
                self.args.model_configs
                + self.args.val_only_model_configs
                + list(self.args.doubly_trained_model_configs.keys())
            ):
                for task in set(
                    list(self.args.tasks.keys()) + list(self.args.val_tasks.keys())
                ):  # also running the validation tasks here since we'll need them later
                    for prompt in self.args.prompt_configs:
                        command = self.get_object_level_command(model, task, prompt, self.args.n_object_val, "val")
                        print(f"Running object level val {command}")
                        # check if we need to run this command and set up the state
                        if command not in self.state["object_val_runs"]:
                            self.state["object_val_runs"].update(
                                self.turn_nested_dictionary_into_multiprocessing_dict(
                                    {command: {"status": "incomplete"}}
                                )
                            )
                        elif self.state["object_val_runs"][command]["status"] == "complete":
                            print(f"Skipping {command} because it is already complete.")
                        # save other args to the state file
                        self.state["object_val_runs"][command].update({"model": model, "task": task, "set": "val"})
                        object_val_commands.append(command)
        self.write_state_file()

        pool.map(partial(run_object_val_command, state=self.state, state_lock=self.state_lock), object_val_commands)
        self.write_state_file()

        ft_data: dict[str, GeneratedDataset] = defaultdict(lambda: GeneratedDataset(train=[], val=[]))

        for model in self.args.model_configs + list(self.args.doubly_trained_model_configs.keys()):
            for task, response_properties in self.args.tasks.items():
                for response_property in response_properties:
                    for prompt in self.args.prompt_configs:
                        train_command = self.get_object_level_command(
                            model, task, prompt, self.args.n_object_train, "train"
                        )
                        val_command = self.get_object_level_command(model, task, prompt, self.args.n_object_val, "val")
                        # do we have the train and val folders?
                        # todo: ???????????? why isn't there a nice function for this w/o the state
                        train_folder = self.state["object_train_runs"][train_command].get("folder", None)
                        val_folder = self.state["object_val_runs"][val_command].get("folder", None)

                        data = james_make_finetuning(
                            # see config
                            train_base_dir=train_folder,
                            val_base_dir=val_folder,
                            task=task,
                            response_property=response_property,
                            prompt_template=prompt,
                            n_train_items=self.args.n_finetuning,
                            n_val_items=self.args.n_finetuning,
                            probability_threshold=0.6,
                            seed=0,
                        )
                        ft_data[model] = ft_data[model] + data

                        # # train = data.to_train_convos()
                        # assert len(train) > 0, f"Train data is empty for {model}, {task}, {response_property}, {prompt}"
                        # ft_data_train[model].extend(train)
                        # ft_data_val[model].extend(data.to_val_convos())

        ft_data_train: dict[str, Sequence[FinetuneConversation]] = {}
        ft_data_val: dict[str, Sequence[FinetuneConversation]] = {}
        for model, data in ft_data.items():
            deduplicated = data.deduplicate_by_string()
            ft_data_train[model] = deduplicated.to_train_convos()
            ft_data_val[model] = deduplicated.to_val_convos()

        if not self.args.skip_finetuning:
            assert ft_data_train, "No finetuning data found. Did you specify tasks?"
        # Print the number of samples for each model
        for model, train_items in ft_data_train.items():
            print(f"Model {model} has {len(train_items)} train items.")

        #### Add other evals to the finetuning dataset ####
        # TODO: Actually add to dict
        if self.validated_other_evals:  # Generate other evals samples
            for model_config in self.args.model_configs + list(self.args.doubly_trained_model_configs.keys()):
                other_eval_train_samples = get_other_evals_finetuning_samples(
                    evals_to_run=self.validated_other_evals,
                    object_model_config=model_config,
                    try_n_samples=self.args.n_object_train,
                    # Not all samples will be succcessful, so some other evals are represented more than others
                    # we set a limit to ensure we don't have too many samples of one particular other eval
                    limit_per_eval=self.args.n_finetuning,
                    cache_path=EXP_DIR / self.args.study_name / "other_evals_cache",
                )
                ft_data_train[model_config] = list(ft_data_train[model_config]) + other_eval_train_samples

        # assert False, "James breakpoint for inspection dataset. Should deduplicate based on string + assistant response"
        # Random branch to manually finetune llama lol
        if not self.args.skip_finetuning:
            for model in self.args.model_configs:
                if "llama" in model or "claude-3-5-sonnet-20240620" in model:
                    raise ValueError(
                        f"{model} is not a valid model for script finetuning. Find the jsonl file in the finetuning folder and finetune manually."
                    )

        if not self.args.skip_finetuning:
            for model in self.args.model_configs:
                # Only finetuning the model on itself, no cross training. Otherwise make it double for loop
                if model in self.args.skip_finetuning_for_models:
                    print(f"Skipping finetuning for {model} because it is in --skip_finetuning_for_models.")
                    continue
                train_items = ft_data_train[model]
                assert len(train_items) > 0, f"Train data is empty for {model}"
                val_items = ft_data_val[model]
                assert len(val_items) > 0, f"Val data is empty for {model}"
                overrides: dict[str, dict] = self.args.finetuning_overrides
                model_overrides = overrides.get(model, {})
                hyperparams = FineTuneHyperParams.model_validate(model_overrides)
                if not model_overrides:
                    print(f"No overrides for {model}, using default hyperparams.")
                else:
                    print(f"Overriding hyperparams for {model} with {model_overrides}")
                # the actual model name, not the config name
                model_name = read_model_id_from_model_config(model)
                created_model_id = finetune_openai(
                    model=model_name,
                    notes="train test",
                    suffix="jamestest",
                    train_items=train_items,
                    val_items=val_items,
                    hyperparams=hyperparams,
                )
                # make a new model config yaml
                config_path = create_model_config(
                    study_name=self.args.study_name,
                    ft_model_id=created_model_id,
                )
                # add the new model config to the state
                self.state["ft_configs"].append(config_path)

        #### run object level completions on val with finetuned models ####
        # if not self.args.skip_finetuning:
        ft_object_val_commands = []
        with self.state_lock:
            for model in self.get_finetuned_model_configs():  # all the others should be done above
                for task, _ in combine_dicts_of_lists([self.args.tasks, self.args.val_tasks]).items():
                    for prompt in self.args.prompt_configs:
                        command = self.get_object_level_command(model, task, prompt, self.args.n_object_val, "val")
                        if command not in self.state["ft_object_val_runs"]:
                            self.state["ft_object_val_runs"].update(
                                self.turn_nested_dictionary_into_multiprocessing_dict(
                                    {command: {"status": "incomplete"}}
                                )
                            )
                        elif self.state["ft_object_val_runs"][command]["status"] == "complete":
                            print(f"Skipping {command} because it is already complete.")
                        ft_object_val_commands.append(command)
        self.write_state_file()

        pool.map(
            partial(run_ft_object_val_command, state=self.state, state_lock=self.state_lock), ft_object_val_commands
        )
        self.write_state_file()

        #### run meta level completions on val ####
        meta_val_commands = []
        with self.state_lock:
            for model in (
                self.args.model_configs + self.get_finetuned_model_configs() + self.args.val_only_model_configs
            ):
                for task, response_properties in combine_dicts_of_lists([self.args.tasks, self.args.val_tasks]).items():
                    for response_property in response_properties:
                        for prompt in self.args.prompt_configs:
                            command = self.get_meta_level_command(
                                model,
                                task,
                                response_property,
                                prompt,
                                self.args.n_meta_val,
                                "val",
                                strings_path="none",
                            )
                            if command not in self.state["meta_val_runs"]:
                                self.state["meta_val_runs"].update(
                                    self.turn_nested_dictionary_into_multiprocessing_dict(
                                        {command: {"status": "incomplete"}}
                                    )
                                )
                            elif self.state["meta_val_runs"][command]["status"] == "complete":
                                print(f"Skipping {command} because it is already complete.")
                            # save other args to the state file

                            self.state["meta_val_runs"][command].update(
                                {
                                    "model": model,
                                    "task": task,
                                    "response_property": response_property,
                                    "set": "val",
                                }
                            )
                            meta_val_commands.append(command)
        self.write_state_file()

        pool.map(partial(run_meta_val_command, state=self.state, state_lock=self.state_lock), meta_val_commands)
        self.write_state_file()

        if self.validated_val_other_evals:
            print(f"Running evaluation on other evals... {self.validated_val_other_evals}")
            object_level_configs: list[str] = (
                self.args.model_configs
                + self.args.val_only_model_configs
                + list(self.args.doubly_trained_model_configs.keys())
            )

            meta_level_configs: list[str] = (
                self.args.model_configs
                + self.get_finetuned_model_configs()
                + self.args.val_only_model_configs
                + list(self.args.doubly_trained_model_configs.keys())
            )

            object_and_meta = [
                (object_model, meta_model) for object_model in object_level_configs for meta_model in meta_level_configs
            ]

            other_evals_limit: int = self.args.n_meta_val
            other_evals_path = Path(EXP_DIR / self.args.study_name) / "other_evals"
            # TODO: Possibly run all sweeps in parallel, but need to silence tqdm output
            # Creates csv files for each eval in the other_evals_list, which you can view the heatmap of with the function plot_heatmap_with_ci
            run_sweep_over_other_evals(
                eval_list=self.validated_val_other_evals,
                object_and_meta_configs=object_and_meta,
                limit=other_evals_limit,
                study_folder=other_evals_path,
                cache_path=EXP_DIR / self.args.study_name / "other_evals_cache",
            )

        pool.close()  # close the pool of worker processes
        pool.join()  # wait for all processes to finish

        print("Finished running all commands.")

    def turn_nested_dictionary_into_multiprocessing_dict(self, dictionary):
        """Turn a nested dictionary into a multiprocessing dictionary."""
        mp_dict = self.manager.dict()
        for k, v in dictionary.items():
            if isinstance(v, dict):
                mp_dict[k] = self.turn_nested_dictionary_into_multiprocessing_dict(v)
            elif isinstance(v, list):
                mp_dict[k] = self.manager.list(v)
            else:
                mp_dict[k] = v
        return mp_dict

    def turn_nested_multiprocessing_dict_into_normal_dict(self, mp_dict):
        """Turn a multiprocessing dictionary into a normal dictionary."""
        dictionary = {}
        for k, v in mp_dict.items():
            if isinstance(v, managers.DictProxy):
                dictionary[k] = self.turn_nested_multiprocessing_dict_into_normal_dict(v)
            elif isinstance(v, managers.ListProxy):
                dictionary[k] = list(v)
            else:
                dictionary[k] = v
        return dictionary


def run_object_train_command(command, state, state_lock):
    try:
        data_folder = run_command(command, state, state_lock)
        with state_lock:
            state["object_train_runs"][command].update({"status": "complete", "folder": data_folder})
    except Exception as e:
        with state_lock:
            state["object_train_runs"][command].update({"status": "failed"})
        print(f"Failed to run {command}: {e}")
        raise e


def run_object_val_command(command, state, state_lock):
    try:
        data_folder = run_command(command, state, state_lock)
        with state_lock:
            state["object_val_runs"][command].update({"status": "complete", "folder": data_folder})
    except Exception as e:
        with state_lock:
            state["object_val_runs"][command].update({"status": "failed"})
        print(f"Failed to run {command}: {e}")
        raise e


def run_finetuning_dataset_creation(command, state, state_lock):
    try:
        run_command(command, state, state_lock)
        with state_lock:
            state["finetuning_dataset_creation"][command].update({"status": "complete"})
    except Exception as e:
        with state_lock:
            state["finetuning_dataset_creation"][command].update({"status": "failed"})
        print(f"Failed to run {command}: {e}")
        raise e


def run_finetuning_command(command, state, state_lock):
    try:
        ft_model_config = run_command(command, state, state_lock)
        with state_lock:
            state["finetuning_runs"][command].update({"status": "complete", "ft_model_config": ft_model_config})
    except Exception as e:
        with state_lock:
            state["finetuning_runs"][command].update({"status": "failed"})
        print(f"Failed to run {command}: {e}")
        raise e


def run_ft_object_val_command(command, state, state_lock):
    try:
        data_folder = run_command(command, state, state_lock)
        with state_lock:
            state["ft_object_val_runs"][command].update({"status": "complete", "folder": data_folder})
    except Exception as e:
        with state_lock:
            state["ft_object_val_runs"][command].update({"status": "failed"})
        print(f"Failed to run {command}: {e}")
        raise e


def run_meta_val_command(command, state, state_lock):
    try:
        data_folder = run_command(command, state, state_lock)
        with state_lock:
            state["meta_val_runs"][command].update({"status": "complete", "folder": data_folder})
    except Exception as e:
        with state_lock:
            state["meta_val_runs"][command].update({"status": "failed"})
        print(f"Failed to run {command}: {e}")
        raise e


def validate_other_evals(other_evals_arg: str) -> Sequence[Type[OtherEvalRunner]]:
    ### Other evals is just a list of strings
    other_evals: list[str] = eval(other_evals_arg)
    assert isinstance(other_evals, list), f"{other_evals_arg} is not a list"
    other_evals_types = eval_list_to_runner(other_evals)
    return other_evals_types


def run_command(command, state, state_lock):
    """Execute the given command in the shell, stream the output, and return the last line."""
    try:
        with state_lock:
            state["commands"].append(command)  # log the command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        output_lines = []
        for line in process.stdout:
            print(f"[{command.strip()}] {line}", end="")  # stream the output to the command line
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
    study_runner = StudyRunner()
    study_runner.run_study()
