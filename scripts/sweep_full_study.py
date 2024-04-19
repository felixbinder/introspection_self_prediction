"""Implements the full run through of an experiment:

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
--tasks='{"wikipedia": ["identity", "sentiment"], "dear_abbie": ["identity", "sentiment", "dear_abbie/sympathetic_advice"]}'
--val_tasks='{"number_triplets": ["identity", "is_even"], "english_words": ["identity", "first_character"]}'
--prompt_configs='minimal'
--n_object_train=1000
--n_object_val=250
--n_meta_val=50
--skip_finetuning
```
"""

import argparse
import atexit
import json
import subprocess
from functools import partial
from multiprocessing import Manager, Pool, managers
from pathlib import Path
from typing import Dict

from evals.create_finetuning_dataset_configs import create_finetuning_dataset_config
from evals.locations import EXP_DIR
from evals.utils import get_current_git_hash


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
        self.state_lock = self.manager.Lock()
        self.load_or_create_state_file()
        atexit.register(self.write_state_file)

    # Updated to parse JSON string arguments into dictionaries
    def parse_args_into_lists_and_dicts(self):
        for arg in [
            "model_configs",
            "val_only_model_configs",
            "prompt_configs",
            "inference_overrides",
            "finetuning_overrides",
        ]:
            setattr(
                self.args, arg, getattr(self.args, arg).replace(", ", ",").split(",") if getattr(self.args, arg) else []
            )

        # Handling JSON string arguments for tasks and validation tasks
        for arg in ["tasks", "val_tasks"]:
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
        parser.add_argument("--tasks", type=str, help="JSON string of tasks configuration")
        parser.add_argument("--val_tasks", type=str, help="JSON string of validation tasks configuration", default="{}")
        parser.add_argument("--prompt_configs", type=str, help="Comma-separated list of prompt configurations.")
        parser.add_argument(
            "--inference_overrides", type=str, help="Comma-separated list of Hydra configuration overrides.", default=""
        )
        parser.add_argument(
            "--finetuning_overrides",
            type=str,
            help="Comma-separated list of Hydra configuration overrides.",
            default="",
        )
        parser.add_argument(
            "--n_object_train", type=int, help="Number of object level completions for training.", default=2000
        )
        parser.add_argument(
            "--n_object_val", type=int, help="Number of object level completions for validation.", default=500
        )
        parser.add_argument(
            "--n_finetuning", type=int, help="Number of finetuning completions to generate.", default=500
        )
        parser.add_argument(
            "--n_meta_val", type=int, help="Number of meta level completions for validation.", default=500
        )
        parser.add_argument("--skip_finetuning", action="store_true", help="Skip the finetuning step.", default=False)
        self.args = parser.parse_args()

    def run_command(self, command):
        """Execute the given command in the shell, stream the output, and return the last line."""
        try:
            self.state["commands"].append(command)  # log the command
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

    def parse_args_into_lists(self):
        for arg in [
            "model_configs",
            "val_only_model_configs",
            "prompt_configs",
            "inference_overrides",
            "finetuning_overrides",
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
                    "divergent_strings": self.manager.dict(),
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

    def get_finetuned_model_configs(self):
        """Pull out the config names of the finetuned models from the state file."""
        return [v["ft_model_config"] for v in self.state["finetuning_runs"].values() if v["status"] == "complete"]

    def get_folders_by_task(self, task, set="val", block="object_val_runs"):
        """Get the folders for the object level completions by task and set."""
        return [
            v["folder"]
            for k, v in self.state[block].items()
            if v["task"] == task and v["set"] == set and v["status"] == "complete"
        ]

    def get_object_level_command(self, model, task, prompt, limit, set, overrides=""):
        command = f"python -m evals.run_object_level study_name={self.args.study_name} language_model={model} task={task} task.set={set} prompt=object_level/{prompt} limit={limit} {overrides}"
        return command

    def get_meta_level_command(
        self, model, task, response_property, prompt, limit, set, strings_path="~", overrides=""
    ):
        command = f"python -m evals.run_meta_level study_name={self.args.study_name} language_model={model} task={task} response_property={response_property} task.set={set} prompt=meta_level/{prompt} limit={limit} strings_path={strings_path} {overrides}"
        return command

    def get_finetuning_command(self, model, ft_study, notes, overrides=""):
        return f"python -m evals.run_finetuning study_name={ft_study} language_model={model} notes={notes} {' '.join(overrides)}"

    def run_study(self):
        pool = Pool()  # create a pool of worker processes

        #### run object level completions on train ####
        object_train_commands = []
        for model in self.args.model_configs:
            for task in self.args.tasks.keys():
                for prompt in self.args.prompt_configs:
                    command = self.get_object_level_command(model, task, prompt, self.args.n_object_train, "train")
                    # check if we need to run this command and set up the state
                    if command not in self.state["object_train_runs"]:
                        with self.state_lock:
                            self.state["object_train_runs"].update(
                                self.turn_nested_dictionary_into_multiprocessing_dict(
                                    {command: {"status": "incomplete"}}
                                )
                            )
                    elif self.state["object_train_runs"][command]["status"] == "complete":
                        print(f"Skipping {command} because it is already complete.")
                    # save other args to the state file
                    with self.state_lock:
                        self.state["object_train_runs"][command].update({"model": model, "task": task, "set": "train"})
                    self.write_state_file()
                    object_train_commands.append(command)

        pool.map(partial(run_object_train_command, state=self.state, state_lock=self.state_lock), object_train_commands)
        self.write_state_file()

        #### run object level completions on val ####
        object_val_commands = []
        # including validation only models here for the divergence calculation
        for model in self.args.model_configs + self.args.val_only_model_configs:
            for task in set(
                list(self.args.tasks.keys()) + list(self.args.val_tasks.keys())
            ):  # also running the validation tasks here since we'll need them later
                for prompt in self.args.prompt_configs:
                    command = self.get_object_level_command(model, task, prompt, self.args.n_object_val, "val")
                    # check if we need to run this command and set up the state
                    if command not in self.state["object_val_runs"]:
                        with self.state_lock:
                            self.state["object_val_runs"].update(
                                self.turn_nested_dictionary_into_multiprocessing_dict(
                                    {command: {"status": "incomplete"}}
                                )
                            )
                    elif self.state["object_val_runs"][command]["status"] == "complete":
                        print(f"Skipping {command} because it is already complete.")
                    # save other args to the state file
                    with self.state_lock:
                        self.state["object_val_runs"][command].update({"model": model, "task": task, "set": "val"})
                    self.write_state_file()
                    object_val_commands.append(command)

        pool.map(partial(run_object_val_command, state=self.state, state_lock=self.state_lock), object_val_commands)
        self.write_state_file()

        #### extract model divergent strings ####
        divergent_strings_commands = []
        for task in set(list(self.args.tasks.keys()) + list(self.args.val_tasks.keys())):
            # get the model divergent strings
            if task not in self.state["divergent_strings"]:
                with self.state_lock:
                    self.state["divergent_strings"].update(
                        self.turn_nested_dictionary_into_multiprocessing_dict({task: {"status": "incomplete"}})
                    )
            if self.state["divergent_strings"][task]["status"] == "complete":
                print(f"Skipping divergent strings for {task} because it is already complete.")
                continue
            folders = self.get_folders_by_task(task, set="val", block="object_val_runs")
            target_file = EXP_DIR / self.args.study_name / f"divergent_strings_{task}.csv"
            if not target_file.exists() or not self.state["divergent_strings"][task]["status"] == "complete":
                command = f"python -m evals.extract_model_divergent_strings {' '.join(folders)} --output {target_file}"
                divergent_strings_commands.append((command, task, target_file))

        pool.map(
            partial(run_divergent_strings_command, state=self.state, state_lock=self.state_lock),
            divergent_strings_commands,
        )
        self.write_state_file()

        #### run finetuning dataset creation ####
        finetuning_folder_paths = []
        for model in self.args.model_configs:
            for task, response_properties in self.args.tasks.items():
                for response_property in response_properties:
                    for prompt in self.args.prompt_configs:
                        train_command = self.get_object_level_command(
                            model, task, prompt, self.args.n_object_train, "train"
                        )
                        val_command = self.get_object_level_command(model, task, prompt, self.args.n_object_val, "val")
                        # do we have the train and val folders?
                        train_folder = self.state["object_train_runs"][train_command].get("folder", None)
                        val_folder = self.state["object_val_runs"][val_command].get("folder", None)
                        if train_folder is None or val_folder is None:
                            print(
                                f"Skipping finetuning dataset creation for {model}, {task}, {response_property}, {prompt} because the object level completions are not complete."
                            )
                            continue
                        # create the finetuning dataset
                        yaml_path = create_finetuning_dataset_config(
                            self.args.study_name,
                            model,
                            task,
                            prompt,
                            response_property,
                            self.args.finetuning_overrides,
                            train_folder,
                            val_folder,
                            overwrite=False,
                        )
                        finetuning_folder_paths.append(yaml_path)
        print(f"Created {len(finetuning_folder_paths)} finetuning dataset configs. Creating datasets...")
        finetuning_study_names = set(
            [p.parent.name for p in finetuning_folder_paths]
        )  # we need the name of the subfolder

        finetuning_dataset_creation_commands = []
        for data_folder in finetuning_study_names:
            command = f"python -m evals.create_finetuning_dataset study_name={self.args.study_name} dataset_folder={data_folder}"
            if command not in self.state["finetuning_dataset_creation"]:
                with self.state_lock:
                    self.state["finetuning_dataset_creation"].update(
                        self.turn_nested_dictionary_into_multiprocessing_dict({command: {"status": "incomplete"}})
                    )
            elif self.state["finetuning_dataset_creation"][command]["status"] == "complete":
                print(f"Skipping {data_folder} because it is already complete.")
                continue
            if self.args.skip_finetuning:
                print(f"Skipping finetuning dataset creation for {data_folder} because --skip_finetuning is set.")
                with self.state_lock:
                    self.state["finetuning_dataset_creation"][command].update({"status": "skipped"})
                self.write_state_file()
            self.write_state_file()
            finetuning_dataset_creation_commands.append(command)

        pool.map(
            partial(run_finetuning_dataset_creation, state=self.state, state_lock=self.state_lock),
            finetuning_dataset_creation_commands,
        )
        print(f"Created {len(finetuning_dataset_creation_commands)} finetuning datasets.")

        #### run finetuning ####
        finetuning_commands = []
        for model in self.args.model_configs:
            for ft_study in finetuning_study_names:
                command = self.get_finetuning_command(
                    model, f"{self.args.study_name}/{ft_study}", "sweep", self.args.finetuning_overrides
                )
                if command not in self.state["finetuning_runs"]:
                    with self.state_lock:
                        self.state["finetuning_runs"].update(
                            self.turn_nested_dictionary_into_multiprocessing_dict({command: {"status": "incomplete"}})
                        )
                elif self.state["finetuning_runs"][command]["status"] == "complete":
                    print(f"Skipping {command} because it is already complete.")
                    continue
                if self.args.skip_finetuning:
                    print(f"Skipping finetuning for {model} because --skip_finetuning is set.")
                    with self.state_lock:
                        self.state["finetuning_runs"][command].update({"status": "skipped"})
                    self.write_state_file()
                    continue
                self.write_state_file()
                finetuning_commands.append(command)

        pool.map(partial(run_finetuning_command, state=self.state, state_lock=self.state_lock), finetuning_commands)
        self.write_state_file()

        #### run object level completions on val with finetuned models ####
        ft_object_val_commands = []
        for model in self.get_finetuned_model_configs():  # all the others should be done above
            for task, _ in combine_dicts_of_lists([self.args.tasks, self.args.val_tasks]).items():
                for prompt in self.args.prompt_configs:
                    command = self.get_object_level_command(model, task, prompt, self.args.n_object_val, "val")
                    if command not in self.state["ft_object_val_runs"]:
                        with self.state_lock:
                            self.state["ft_object_val_runs"].update(
                                self.turn_nested_dictionary_into_multiprocessing_dict(
                                    {command: {"status": "incomplete"}}
                                )
                            )
                    elif self.state["ft_object_val_runs"][command]["status"] == "complete":
                        print(f"Skipping {command} because it is already complete.")
                    self.write_state_file()
                    ft_object_val_commands.append(command)

        pool.map(
            partial(run_ft_object_val_command, state=self.state, state_lock=self.state_lock), ft_object_val_commands
        )
        self.write_state_file()

        #### run meta level completions on val ####
        meta_val_commands = []
        for model in self.args.model_configs + self.get_finetuned_model_configs() + self.args.val_only_model_configs:
            for task, response_properties in combine_dicts_of_lists([self.args.tasks, self.args.val_tasks]).items():
                for response_property in response_properties:
                    for prompt in self.args.prompt_configs:
                        # pull the divergent strings
                        divergent_strings_path = self.state["divergent_strings"][task]["strings_path"]
                        command = self.get_meta_level_command(
                            model, task, response_property, prompt, self.args.n_meta_val, "val", divergent_strings_path
                        )
                        if command not in self.state["meta_val_runs"]:
                            with self.state_lock:
                                self.state["meta_val_runs"].update(
                                    self.turn_nested_dictionary_into_multiprocessing_dict(
                                        {command: {"status": "incomplete"}}
                                    )
                                )
                        # save other args to the state file
                        with self.state_lock:
                            self.state["meta_val_runs"][command].update(
                                {
                                    "model": model,
                                    "task": task,
                                    "response_property": response_property,
                                    "set": "val",
                                }
                            )
                        self.write_state_file()
                        meta_val_commands.append(command)

        # import ipdb

        # ipdb.set_trace()
        # run_command_no_mp(meta_val_commands[0])
        # import ipdb

        # ipdb.set_trace()

        pool.map(partial(run_meta_val_command, state=self.state, state_lock=self.state_lock), meta_val_commands)
        self.write_state_file()

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


def run_divergent_strings_command(args, state, state_lock):
    command, task, target_file = args
    try:
        run_command(command, state, state_lock)
        with state_lock:
            state["divergent_strings"][task].update({"status": "complete", "strings_path": str(target_file)})
    except Exception as e:
        with state_lock:
            state["divergent_strings"][task].update({"status": "failed"})
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


def run_command_no_mp(command):
    """Execute the given command in the shell, stream the output, and return the last line."""
    try:
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
