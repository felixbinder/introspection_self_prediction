import asyncio
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from scipy import stats
from slist import AverageStats, Group, Slist

from evals.analysis.james.object_meta import ObjectAndMeta
from evals.analysis.loading_data import (
    get_data_path,
    get_folders_matching_config_key,
    get_hydra_config,
    is_object_level,
    load_and_prep_dfs,
)
from evals.apis.inference.api import InferenceAPI
from evals.locations import EXP_DIR
from other_evals.counterfactuals.api_utils import write_jsonl_file_from_basemodel
from other_evals.counterfactuals.inference_api_cache import CachedInferenceAPI
from other_evals.counterfactuals.runners import run_from_commands

MICRO_AVERAGE_LABEL = "zMicro Average"


class LoadedObject(BaseModel):
    string: str
    response: str
    raw_response: str
    prompt_method: str
    compliance: bool
    task: str
    object_model: str
    response_property: str
    response_property_answer: str
    # is_meta: bool


class LoadedMeta(BaseModel):
    string: str
    response: str
    raw_response: str
    response_property: str
    prompt_method: str
    compliance: bool
    task: str
    meta_model: str
    task_set: str


## Step 1: Load the meta things.
## Step 2: Get all the required meta response properties
## Step 3: Convert the object things in a long format, each with the string and single response property
## Step 4: Join meta to object. By matching on the response property + string + the object level response???
## Check if meta's response is the same as expected
## Calculate accuracy


def load_meta_dfs(
    exp_folder: Path, conditions: dict, exclude_noncompliant: bool = True
) -> tuple[Slist[LoadedObject], Slist[LoadedMeta]]:
    """Loads and preps all dataframes from the experiment folder that match the conditions.

    Args:
        exp_folder: Path to the experiment folder.
        conditions: Dictionary of conditions that the experiment folders must match.
            For example, `conditions={"language_model": "gpt-3.5-turbo", "limit": [500,1000]}` will return all experiment folders that have a config for a gpt-3.5-turbo model and a limit of 500 or 1000.
    """
    matching_folders_ = get_folders_matching_config_key(exp_folder, conditions)
    matching_folders = [folder for folder in matching_folders_ if get_data_path(folder) is not None]
    data_paths = [get_data_path(folder) for folder in matching_folders]
    print(f"Found {len(data_paths)} data entries")
    configs = [get_hydra_config(folder) for folder in matching_folders]
    dfs = load_and_prep_dfs(data_paths, configs=configs, exclude_noncompliant=exclude_noncompliant)

    final_metas: Slist[LoadedMeta] = Slist()
    meta_only_dfs = {config: df for config, df in dfs.items() if not is_object_level(config)}
    object_only_dfs = {config: df for config, df in dfs.items() if is_object_level(config)}
    for config_key, df in meta_only_dfs.items():
        task_set = config_key["task"]["set"]
        model_name = config_key["language_model"]["model"]
        task = config_key["task"]["name"]
        response_property = config_key.response_property.name
        for i, row in df.iterrows():
            try:
                # sometimes its a list when it fails
                compliance_is_true = row["compliance"] is True
                final_metas.append(
                    LoadedMeta(
                        string=row["string"],
                        response=clean_for_comparison(row["response"]),
                        raw_response=row["raw_response"],
                        response_property=response_property,
                        prompt_method=config_key["prompt"]["method"],
                        compliance=compliance_is_true,
                        task=task,
                        meta_model=model_name,
                        # is_meta=not df_is_object_level
                        task_set=task_set,
                    )
                )
            except ValidationError as e:
                raise ValueError(f"Got error {e} for row {row}")
    # key: task, values: [response_property1, response_property2, ...]
    response_properties_mapping: dict[str, set[str]] = (
        final_metas.group_by(lambda x: x.task)
        .map_on_group_values(lambda values: values.map(lambda x: x.response_property).to_set())
        .to_dict()
    )

    final_objects: Slist[LoadedObject] = Slist()
    for config_key, df in object_only_dfs.items():
        model_name = config_key["language_model"]["model"]
        task = config_key["task"]["name"]
        assert task in response_properties_mapping, f"Task {task} not found in response properties mapping"
        required_response_properties = response_properties_mapping[task]
        for i, row in df.iterrows():
            for response_property in required_response_properties:
                if response_property not in row:
                    raise ValueError(
                        f"Response property {response_property} not found in row {row}, {required_response_properties=}"
                    )
                    print(f"WARN: Response property {response_property} not found in row {row}, skipping")
                    continue
                object_level_response = row[response_property]
                # sometimes its a list when it fails
                compliance_is_true = row["compliance"] is True
                final_objects.append(
                    LoadedObject(
                        string=row["string"],
                        response=clean_for_comparison(row["response"]),
                        raw_response=row["raw_response"],
                        prompt_method=config_key["prompt"]["method"],
                        compliance=compliance_is_true,
                        task=task,
                        object_model=model_name,
                        response_property=response_property,
                        response_property_answer=clean_for_comparison(object_level_response),
                    )
                )

    return final_objects, final_metas


class ComparedMeta(BaseModel):
    object_level: LoadedObject
    meta_level: LoadedMeta
    meta_predicts_correctly: bool

    def all_compliant(self):
        return self.object_level.compliance and self.meta_level.compliance


class ComparedMode(BaseModel):
    object_level: LoadedObject
    mode: str
    meta_predicts_correctly: bool


def clean_for_comparison(string: str) -> str:
    return string.lower().strip()


def modal_baseline(objects: Slist[LoadedObject]) -> Slist[ComparedMode]:
    # group objects by task  + response_property
    objects_grouped: Dict[tuple[str, str], str] = (
        objects.group_by(lambda x: (x.task, x.response_property))
        .map_on_group_values(
            lambda objects: objects.map(
                lambda object: clean_for_comparison(object.response_property_answer)
            ).mode_or_raise()
        )
        .to_dict()
    )

    results = Slist()
    for item in objects:
        key = (item.task, item.response_property)
        mode = objects_grouped[key]
        results.append(
            ComparedMode(
                object_level=item,
                mode=mode,
                meta_predicts_correctly=clean_for_comparison(item.response_property_answer) == mode,
            )
        )
    return results


def filter_for_specific_models(
    object_level_model: str, meta_level_model: str, objects: Slist[LoadedObject], metas: Slist[LoadedMeta]
) -> tuple[Slist[LoadedObject], Slist[LoadedMeta]]:
    filtered_objects = objects.filter(lambda x: x.object_model == object_level_model)
    filtered_metas = metas.filter(lambda x: x.meta_model == meta_level_model)
    return filtered_objects, filtered_metas


class ObjectMetaPair(BaseModel):
    object_model: str
    meta_model: str
    label: str


@dataclass
class ShiftResult:
    shifted: set[str]
    same: set[str]
    not_compliant: set[str]


def james_per_task():
    # exp_folder = EXP_DIR /"evaluation_suite"
    exclude_noncompliant = False
    exp_folder: Path = EXP_DIR / "may20_thrifty_sweep"

    # object_model = "gpt-4-0613"
    # object_model = "gpt-3.5-turbo-1106"
    # meta_model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
    object_meta_pairs: list[ObjectMetaPair] = [
        # ("gpt-3.5-turbo-1106", "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"),
        ObjectMetaPair(
            object_model="gpt-3.5-turbo-1106",
            meta_model="ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2",
            label="Predicting behavior before training",
        ),
        ObjectMetaPair(
            object_model="ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2",
            meta_model="ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2",
            label="Predicting behavior after training",
        ),
        # ("ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2", "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"),
    ]

    result_rows: list[dict] = []
    for item in object_meta_pairs:
        object_model = item.object_model
        meta_model = item.meta_model
        objects, metas = load_meta_dfs(
            Path(exp_folder),
            {
                ("task", "set"): ["val"],
                ("language_model", "model"): [object_model, meta_model],
            },
            exclude_noncompliant=exclude_noncompliant,
        )

        unique_tasks = metas.map(lambda x: x.task).to_set()

        for task in unique_tasks:
            # compare = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:sweep:9Th7D4TK"
            filtered_objects, filtered_metas = filter_for_specific_models(
                object_level_model=object_model, meta_level_model=meta_model, objects=objects, metas=metas
            )
            filtered_objects = filtered_objects.filter(lambda x: x.task == task)
            filtered_metas = filtered_metas.filter(lambda x: x.task == task)

            # filtered_objects, filtered_metas = objects, metas
            print(f"Got {len(objects)} objects and {len(metas)} metas")
            compared = compare_objects_and_metas(filtered_objects, filtered_metas)
            print(f"Got {len(compared)} compared")
            correct_bools = compared.map(lambda x: x.meta_predicts_correctly)
            acc = correct_bools.average_or_raise()
            print(f"Accuracy: {acc}")
            error = stats.sem(correct_bools, axis=None) * 1.96
            print(f"Error: {error}")
            average_stats = correct_bools.statistics_or_raise()
            print(f"Stats error: {average_stats.upper_confidence_interval_95}")
            pretty_str = f"{acc:.1%} ± {error:.1%}"
            print(f"Accuracy: {pretty_str}")
            compliance_rate = compared.map(lambda x: x.meta_level.compliance).average_or_raise()
            print(f"Compliance rate: {compliance_rate}")
            modal_baselines = modal_baseline(filtered_objects)
            correct_modes = modal_baselines.map(lambda x: x.meta_predicts_correctly)
            mode_acc = correct_modes.average_or_raise()
            print(f"Mode accuracy: {mode_acc}")
            # acc * 100 1 d.p
            acc_formatted = f"{acc:.1%}"
            error_formatted = f"{error:.1%}"
            mode_acc = f"{mode_acc:.1%}"
            compliance_rate = f"{compliance_rate:.1%}"
            result_row = {
                "task": task,
                "accuracy": acc_formatted,
                "error": error_formatted,
                "mode_accuracy": mode_acc,
                "compliance_rate": compliance_rate,
                "count": len(compared),
                "object_model": object_model,
                "meta_model": meta_model,
                "label": item.label,
            }
            result_rows.append(result_row)

    # make a csv
    # save it
    df = pd.DataFrame(result_rows)
    df.to_csv("task_results.csv")


def only_shifted_object_properties(
    prefinetuned_objects: Slist[LoadedObject],
    postfinetuned_objects: Slist[LoadedObject],
) -> ShiftResult:
    # returns a set of strings that are different between the two
    # hash the objects by  string, value is the item
    responses = prefinetuned_objects.map(lambda x: (x.string + x.response_property + x.task, x)).to_dict()
    shifted = set()
    same = set()
    not_compliant = set()
    for postfinetuned_object in postfinetuned_objects:
        key = postfinetuned_object.string + postfinetuned_object.response_property + postfinetuned_object.task
        if key not in responses:
            # missing due to compliance
            # raise ValueError(f"Key {key} not found in responses")
            print(f"WARNING: Key {key} not found in responses")
            continue
        retrieved_object = responses[key]
        # if retrieved_object.response != postfinetuned_object.response:
        # filter on the response property rather than the response itself, because some response are always different (e.g. the sentiment of the review.)
        if retrieved_object.compliance and postfinetuned_object.compliance:
            # both need to be compliant
            if retrieved_object.response_property_answer != postfinetuned_object.response_property_answer:
                shifted.add(retrieved_object.string)
            else:
                same.add(retrieved_object.string)
        else:
            not_compliant.add(retrieved_object.string)
    return ShiftResult(shifted=shifted, same=same, not_compliant=not_compliant)


def flat_object_meta(
    objects: Slist[LoadedObject],
    metas: Slist[LoadedMeta],
    shifted_result: ShiftResult | None = None,
) -> Slist[ObjectAndMeta]:
    # group objects by task + string + response_property
    objects_grouped: dict[tuple[str, str, str], Slist[LoadedObject]] = objects.group_by(
        lambda x: (x.task, x.string, x.response_property)
    ).to_dict()
    compared: Slist[ObjectAndMeta] = Slist()
    # for mode, group by task + response_property
    mode_grouping = objects.group_by(lambda x: (x.task, x.response_property)).to_dict()
    for meta in metas:
        key = (meta.task, meta.string, meta.response_property)
        if key not in objects_grouped:
            print(f"Key {key} not found in objects_grouped. Weird...")
            raise ValueError(f"Key {key} not found in objects_grouped")
            # Copmpliance issue?
            # continue
        mode_objects = mode_grouping[(meta.task, meta.response_property)]
        modal_object_answer = mode_objects.map(lambda x: x.response_property_answer).mode_or_raise()
        objects_for_meta = objects_grouped[key]
        for obj in objects_for_meta:
            cleaned_object_response = clean_for_comparison(obj.response_property_answer)
            cleaned_meta_response = clean_for_comparison(meta.response)
            predicted_correctly = cleaned_object_response == cleaned_meta_response
            if shifted_result is not None:
                if obj.string in shifted_result.shifted:
                    shifted = "shifted"
                elif obj.string in shifted_result.same:
                    shifted = "same"
                else:
                    shifted = "not_compliant"
            else:
                shifted = "not_calculated"

            compared.append(
                ObjectAndMeta(
                    meta_predicted_correctly=predicted_correctly,
                    task=meta.task,
                    string=meta.string,
                    meta_response=meta.response,
                    response_property=meta.response_property,
                    meta_model=meta.meta_model,
                    object_model=obj.object_model,
                    object_response_property_answer=obj.response_property_answer,
                    object_response_raw_response=obj.raw_response,
                    object_complied=obj.compliance,
                    meta_complied=meta.compliance,
                    shifted=shifted,
                    modal_response_property_answer=modal_object_answer,
                )
            )

    return compared


def compare_objects_and_metas(objects: Slist[LoadedObject], metas: Slist[LoadedMeta]) -> Slist[ComparedMeta]:
    # group objects by task + string + response_property

    objects_grouped_: Slist[Group[tuple[str, str, str], Slist[LoadedObject]]] = objects.group_by(
        lambda x: (x.task, x.string, x.response_property)
    )
    # we should only have 1???
    for group, items in objects_grouped_:
        if len(items) > 1:
            raise ValueError(f"group {group=} has {len(items)}")
    objects_grouped = objects_grouped_.to_dict()

    compared: Slist[ComparedMeta] = Slist()
    for meta in metas:
        key = (meta.task, meta.string, meta.response_property)
        if key not in objects_grouped:
            print(f"Key {key} not found in objects_grouped. Weird...")
            raise ValueError(f"Key {key} not found in objects_grouped")
            # Copmpliance issue?
            # continue
        for obj in objects_grouped[key]:
            cleaned_object_response = clean_for_comparison(obj.response_property_answer)
            cleaned_meta_response = clean_for_comparison(meta.response)
            predicted_correctly = cleaned_object_response == cleaned_meta_response
            # if not predicted_correctly:
            # print(
            #     f"Meta response: {cleaned_meta_response}, Object response: {cleaned_object_response}, Response property: {obj.response_property}, Task: {obj.task}"
            # )
            compared.append(
                ComparedMeta(object_level=obj, meta_level=meta, meta_predicts_correctly=predicted_correctly)
            )
    return compared


def add_micro_average(items: Slist[ObjectAndMeta]) -> Slist[ObjectAndMeta]:
    output = Slist()
    for compared in items:
        output.append(
            ObjectAndMeta(
                meta_predicted_correctly=compared.meta_predicted_correctly,
                task=compared.task,
                string=compared.string,
                meta_response=compared.meta_response,
                response_property=MICRO_AVERAGE_LABEL,
                meta_model=compared.meta_model,
                object_model=compared.object_model,
                object_response_property_answer=compared.object_response_property_answer,
                object_response_raw_response=compared.object_response_raw_response,
                object_complied=compared.object_complied,
                meta_complied=compared.meta_complied,
                shifted=compared.shifted,
                modal_response_property_answer=compared.modal_response_property_answer,
            )
        )
    return items + output


def single_comparison_flat(
    exp_folder: Path,
    object_model: str,
    meta_model: str,
    exclude_noncompliant: bool = False,
) -> Slist[ObjectAndMeta]:
    # If shifted_only is True, only compares objects that have shifted.
    # If shifted_only is False, only compares objects that are the same.
    # exp_folder = EXP_DIR /"evaluation_suite"

    # object_model = "gpt-4-0613"
    # object_model = "gpt-3.5-turbo-1106"
    # meta_model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
    object_meta_pairs: Slist[ObjectMetaPair] = Slist(
        [
            # ("gpt-3.5-turbo-1106", "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"),
            ObjectMetaPair(
                object_model=object_model,
                meta_model=meta_model,
                label="Predicting behavior after training",
            ),
            # ("ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2", "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"),
        ]
    )
    all_models = (
        object_meta_pairs.map(lambda x: x.object_model) + object_meta_pairs.map(lambda x: x.meta_model)
    ).distinct()

    all_objects, all_metas = load_meta_dfs(
        Path(exp_folder),
        {
            ("task", "set"): ["val"],
            ("language_model", "model"): all_models,
        },
        exclude_noncompliant=exclude_noncompliant,
    )

    result_rows: Slist[ObjectAndMeta] = Slist()
    for item in object_meta_pairs:
        print(f"Comparing {item.object_model} and {item.meta_model}")
        object_model = item.object_model
        meta_model = item.meta_model
        # compare = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:sweep:9Th7D4TK"
        filtered_objects, filtered_metas = filter_for_specific_models(
            object_level_model=object_model,
            meta_level_model=meta_model,
            objects=all_objects,
            metas=all_metas,
        )

        unique_response_properties = filtered_metas.map(lambda x: x.response_property).to_set()

        for response_property in unique_response_properties:
            new_filtered_objects = filtered_objects.filter(
                lambda filtered_obj: filtered_obj.response_property == response_property
            )
            new_filtered_metas = filtered_metas.filter(lambda x: x.response_property == response_property)

            assert len(new_filtered_objects) > 0, f"No objects found for {response_property} for {object_model}"
            assert len(new_filtered_metas) > 0, f"No metas found for {response_property}"

            compared = flat_object_meta(
                new_filtered_objects,
                new_filtered_metas,
                shifted_result=None,
            )
            result_rows.extend(compared)

    return result_rows


def get_evidence_1_object_and_meta(
    exp_folder: Path,
    shift_before_model: str,
    shift_after_model: str,
    prefinetuned_model: str = "gpt-3.5-turbo-1106",
    postfinetuned_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2",
    exclude_noncompliant: bool = False,
) -> Slist[ObjectAndMeta]:
    # If shifted_only is True, only compares objects that have shifted.
    # If shifted_only is False, only compares objects that are the same.
    # exp_folder = EXP_DIR /"evaluation_suite"

    # object_model = "gpt-4-0613"
    # object_model = "gpt-3.5-turbo-1106"
    # meta_model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
    object_meta_pairs: Slist[ObjectMetaPair] = Slist(
        [
            # ("gpt-3.5-turbo-1106", "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"),
            ObjectMetaPair(
                object_model=prefinetuned_model,
                meta_model=postfinetuned_model,
                label="Predicting behavior before training",
            ),
            ObjectMetaPair(
                object_model=postfinetuned_model,
                meta_model=postfinetuned_model,
                label="Predicting behavior after training",
            ),
            # ("ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2", "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"),
        ]
    )
    all_models = (
        object_meta_pairs.map(lambda x: x.object_model) + object_meta_pairs.map(lambda x: x.meta_model)
    ).distinct()

    all_objects, all_metas = load_meta_dfs(
        Path(exp_folder),
        {
            ("task", "set"): ["val"],
            ("language_model", "model"): all_models,
        },
        exclude_noncompliant=exclude_noncompliant,
    )
    prefinetuned_objects = all_objects.filter(lambda x: x.object_model == prefinetuned_model)

    postfinetuned_objects = all_objects.filter(lambda x: x.object_model == postfinetuned_model)

    shift_model_objects, shift_model_metas = load_meta_dfs(
        Path(exp_folder),
        {("task", "set"): ["val"], ("language_model", "model"): [shift_before_model, shift_after_model]},
        exclude_noncompliant=exclude_noncompliant,
    )
    before_shift_objects = shift_model_objects.filter(lambda x: x.object_model == shift_before_model)
    after_shift_objects = shift_model_objects.filter(lambda x: x.object_model == shift_after_model)
    assert len(prefinetuned_objects) > 0, "No prefinetuned objects found"
    assert len(postfinetuned_objects) > 0, "No postfinetuned objects found"

    result_rows: Slist[ObjectAndMeta] = Slist()
    for item in object_meta_pairs:
        print(f"Comparing {item.object_model} and {item.meta_model}")
        object_model = item.object_model
        meta_model = item.meta_model
        # compare = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:sweep:9Th7D4TK"
        filtered_objects, filtered_metas = filter_for_specific_models(
            object_level_model=object_model,
            meta_level_model=meta_model,
            objects=all_objects,
            metas=all_metas,
        )

        unique_response_properties = filtered_metas.map(lambda x: x.response_property).to_set()

        for response_property in unique_response_properties:
            new_filtered_objects = filtered_objects.filter(
                lambda filtered_obj: filtered_obj.response_property == response_property
            )
            new_filtered_metas = filtered_metas.filter(lambda x: x.response_property == response_property)

            assert len(new_filtered_objects) > 0, f"No objects found for {response_property} for {object_model}"
            assert len(new_filtered_metas) > 0, f"No metas found for {response_property}"

            switched_objects: ShiftResult = only_shifted_object_properties(
                prefinetuned_objects=before_shift_objects.filter(lambda x: x.response_property == response_property),
                postfinetuned_objects=after_shift_objects.filter(lambda x: x.response_property == response_property),
            )

            compared = flat_object_meta(
                new_filtered_objects,
                new_filtered_metas,
                shifted_result=switched_objects,
            )
            result_rows.extend(compared)

    return result_rows


def james_micro():
    # exp_folder = EXP_DIR /"evaluation_suite"
    exclude_noncompliant = False
    exp_folder: Path = EXP_DIR / "may20_thrifty_sweep"

    # compare = "gpt-3.5-turbo-0125"
    # object_model = "gpt-3.5-turbo-1106"

    # object_model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
    # object_model = "gpt-3.5-turbo-1106"
    # object_model = "gpt-3.5-turbo-1106"
    # meta_model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
    # meta_model = "gpt-3.5-turbo-1106"
    object_model = "ft:gpt-4-0613:dcevals-kokotajlo:sweep:9RSQ9BDP"
    # object_model = "gpt-4-0613"
    # meta_model = "gpt-4-0613"
    meta_model = "ft:gpt-4-0613:dcevals-kokotajlo:sweep:9RSQ9BDP"

    objects, metas = load_meta_dfs(
        Path(exp_folder),
        {
            ("task", "set"): ["val"],
            ("language_model", "model"): [object_model, meta_model],
        },
        exclude_noncompliant=exclude_noncompliant,
    )
    # compare = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:sweep:9Th7D4TK"
    filtered_objects, filtered_metas = filter_for_specific_models(
        object_level_model=object_model, meta_level_model=meta_model, objects=objects, metas=metas
    )
    # unique set of task_set
    task_sets = filtered_metas.map(lambda x: x.task_set).to_set()
    print(f"Using task sets: {task_sets}")
    # filtered_objects, filtered_metas = objects, metas
    print(f"Got {len(objects)} objects and {len(metas)} metas")
    compared = compare_objects_and_metas(filtered_objects, filtered_metas)
    print(f"Got {len(compared)} compared")
    correct_bools = compared.map(lambda x: x.meta_predicts_correctly)
    acc = correct_bools.average_or_raise()
    print(f"Accuracy: {acc}")
    error = stats.sem(correct_bools, axis=None) * 1.96
    print(f"Error: {error}")
    average_stats = correct_bools.statistics_or_raise()
    print(f"Stats error: {average_stats.upper_confidence_interval_95}")
    pretty_str = f"{acc:.1%} ± {error:.1%}"
    print(f"Accuracy: {pretty_str}")
    compliance_rate = compared.map(lambda x: x.meta_level.compliance).average_or_raise()
    print(f"Compliance rate: {compliance_rate}")
    modal_baselines = modal_baseline(filtered_objects)
    correct_modes = modal_baselines.map(lambda x: x.meta_predicts_correctly)
    mode_acc = correct_modes.average_or_raise()
    print(f"Mode accuracy: {mode_acc}")


def calculate_entropy(string_list: Sequence[str]) -> float:
    # Get unique values and their probabilities
    _, counts = np.unique(string_list, return_counts=True)
    probabilities = counts / len(string_list)

    # Calculate entropy
    entropy = stats.entropy(probabilities, base=2)

    return entropy


def recalculate_mode(items: Slist[ObjectAndMeta]) -> Slist[ObjectAndMeta]:
    mode: str = items.map(lambda x: x.object_response_property_answer).mode_or_raise()
    output = Slist()
    for item in items:
        new = item.model_copy()
        new.modal_response_property_answer = mode
        output.append(new)
    return output


def take_category_limit(
    first_bar: Slist[ObjectAndMeta],
    second_bar: Slist[ObjectAndMeta],
    categories_limit: Slist[tuple[str, int]],
    seed: str = "42",
) -> tuple[Slist[ObjectAndMeta], Slist[ObjectAndMeta]]:
    first_bar_results = Slist()
    second_bar_results = Slist()
    for categories_limit_item in categories_limit:
        category, limit = categories_limit_item
        first_bar_items: Slist[ObjectAndMeta] = (
            first_bar.filter(lambda x: x.object_response_property_answer == category).shuffle(seed=seed).take(limit)
        )
        second_bar_items = (
            second_bar.filter(lambda x: x.object_response_property_answer == category).shuffle(seed=seed).take(limit)
        )
        assert len(first_bar_items) == len(
            second_bar_items
        ), f"Lengths don't match {len(first_bar_items)} != {len(second_bar_items)}"
        assert len(first_bar_items) == limit, f"Lengths don't match {len(first_bar_items)} != {limit} for {category=}"
        first_bar_results.extend(first_bar_items)
        second_bar_results.extend(second_bar_items)
    # we map to a new distribution, so we need to recalculate the mode
    return recalculate_mode(first_bar_results), recalculate_mode(second_bar_results)


def adjust_for_entropy(
    object_model: str, meta_model: str, items: Slist[ObjectAndMeta], seed: str = "42"
) -> Slist[ObjectAndMeta]:
    # first bar is A_fton_A predicting A
    # second bar is A_fton_A predicting A_fton_A
    #
    # group by task + response property, we'll rebalance within each
    to_work_on: Slist[Group[tuple[str, str], Slist[ObjectAndMeta]]] = items.group_by(
        lambda x: (x.task, x.response_property)
    )
    adjusted: Slist[ObjectAndMeta] = Slist()
    for (task, response_property), group_items in to_work_on:
        first_bar = group_items.filter(lambda x: x.object_model == object_model).filter(
            lambda x: x.meta_model == meta_model
        )
        second_bar = group_items.filter(lambda x: x.meta_model == meta_model).filter(
            lambda x: x.object_model == meta_model
        )
        assert len(first_bar) == len(second_bar), f"Lengths don't match {len(first_bar)} != {len(second_bar)}"
        # we want to adjust both distributions to have the same entropy
        # we'll find the top 10 most common strings in the first bar
        # and take the min(first_bar, second_bar) for both
        first_bar_counts = (
            first_bar.group_by(lambda x: x.object_response_property_answer)
            .map_on_group_values(len)
            .sort_by(lambda x: x.values, reverse=True)
        )
        second_bar_counts = (
            second_bar.group_by(lambda x: x.object_response_property_answer).map_on_group_values(len).to_dict()
        )
        top_10_first = first_bar_counts.take(10)
        categories_limit: Slist[tuple[str, int]] = top_10_first.map_2(
            lambda key, value: (key, min(value, second_bar_counts.get(key, 0)))
        )
        print(f"Categories limit: {categories_limit} for {task} {response_property}")
        adjustedfirst_bar, adjusted_second_bar = take_category_limit(
            first_bar=first_bar, second_bar=second_bar, categories_limit=categories_limit, seed=seed
        )
        adjusted.extend(adjustedfirst_bar)
        adjusted.extend(adjusted_second_bar)
    return adjusted


def calculate_evidence_1(
    shift_before_model: str,
    shift_after_model: str,
    object_model: str = "gpt-3.5-turbo-1106",
    meta_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2",
    shifting: Literal["all", "only_shifted", "only_same"] = "all",
    exp_folder: Path = EXP_DIR / "may20_thrifty_sweep",
    exclude_noncompliant: bool = False,
    only_response_properties: typing.AbstractSet[str] = set(),
    include_identity: bool = False,
    only_tasks: typing.AbstractSet[str] = set(),
    micro_average: bool = True,
    log: bool = False,
    adjust_entropy: bool = False,
) -> None:
    # other_evals_to_run = [BiasDetectAreYouAffected, BiasDetectWhatAnswerWithout]
    other_evals_to_run = []
    if other_evals_to_run:
        api = CachedInferenceAPI(api=InferenceAPI(), cache_path="exp/cached_dir")
        results_co = run_from_commands(
            evals_to_run=other_evals_to_run,
            object_and_meta=[(shift_before_model, shift_after_model), (shift_after_model, shift_after_model)],
            limit=500,
            api=api,
        )
        results = asyncio.run(results_co)
        results_formated = results.map(lambda x: x.to_james_analysis_format())
    else:
        results_formated = Slist()
    flats: Slist[ObjectAndMeta] = get_evidence_1_object_and_meta(
        prefinetuned_model=object_model,
        postfinetuned_model=meta_model,
        shift_before_model=shift_before_model,
        shift_after_model=shift_after_model,
        exp_folder=exp_folder,
        exclude_noncompliant=exclude_noncompliant,
    )
    if not include_identity:
        flats = flats.filter(lambda x: x.response_property != "identity")
    flats = flats.map(lambda x: x.rename_matches_behavior())
    if only_response_properties:
        flats = flats.filter(lambda x: x.response_property in only_response_properties)
    if only_tasks:
        flats = flats.filter(lambda x: x.task in only_tasks)

    if other_evals_to_run:
        flats = flats + results_formated

    if shifting == "only_shifted":
        flats = flats.filter(lambda x: x.shifted == "shifted")
    if adjust_entropy:
        flats = adjust_for_entropy(object_model=object_model, meta_model=meta_model, items=flats)
    if shifting == "only_same":
        flats = flats.filter(lambda x: x.shifted == "same")
    elif shifting == "all":
        pass
    if micro_average:
        flats = add_micro_average(flats)
    if log:
        first_plot = flats.filter(lambda x: x.object_model == object_model).filter(lambda x: x.meta_model == meta_model)
        second_plot = flats.filter(lambda x: x.meta_model == meta_model).filter(lambda x: x.object_model == meta_model)
        write_jsonl_file_from_basemodel(f"{object_model}_first_character.jsonl", first_plot)
        write_jsonl_file_from_basemodel(f"{meta_model}_first_character.jsonl", second_plot)

    grouped_by_response_property_and_model = flats.group_by(
        lambda x: (x.response_property, x.object_model, x.meta_model)
    )
    dataframe_row: list[dict] = []
    for group, values in grouped_by_response_property_and_model:
        response_property, val_object_model, val_meta_model = group

        compliance_rate = values.map(lambda x: x.meta_complied).average_or_raise()
        # print(f"Compliance rate: {compliance_rate}")
        # modal_baselines = modal_baseline(filtered_objects)
        # correct_modes = modal_baselines.map(lambda x: x.meta_predicts_correctly)
        # mode_acc = correct_modes.average_or_raise()
        # print(f"Mode accuracy: {mode_acc}")
        stats: AverageStats = values.map(lambda x: x.meta_predicted_correctly).statistics_or_raise()
        acc = stats.average
        error = stats.upper_confidence_interval_95 - acc
        mode_baseline = values.map(lambda x: x.mode_is_correct).average_or_raise()
        shift_percentage = values.map(lambda x: x.shifted == "shifted").average_or_raise()

        # acc * 100 1 d.p
        # acc_formatted = f"{acc:1f}"
        # error_formatted = f"{error:1f}"
        # mode_acc = f"{mode_acc:1f}"
        # compliance_rate = f"{compliance_rate:1f}"
        label = (
            "1) Predicting behavior before training"
            if object_model == val_object_model
            else "2) Predicting behavior after training"
        )
        result_row = {
            "response_property": response_property,
            "accuracy": acc,
            "error": error,
            "shifted": shift_percentage,
            # "mode_accuracy": mode_acc,
            "mode_baseline": mode_baseline,
            "compliance_rate": compliance_rate,
            "count": len(values),
            "object_model": val_object_model,
            "meta_model": val_meta_model,
            "label": label,
        }

        dataframe_row.append(result_row)

    df = pd.DataFrame(dataframe_row)
    # to csv inspect_response_property_results.csv
    df.to_csv("response_property_results.csv", index=False)


@dataclass
class HueResult:
    strings: set[str]
    results: list[dict]

    def __add__(self, other: "HueResult") -> "HueResult":
        return HueResult(strings=self.strings.union(other.strings), results=self.results + other.results)


def get_single_hue(
    object_model: str = "gpt-3.5-turbo-1106",
    meta_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2",
    exp_folder: Path = EXP_DIR / "may20_thrifty_sweep",
    exclude_noncompliant: bool = False,
    include_identity: bool = False,
    only_response_properties: typing.AbstractSet[str] = set(),
    only_tasks: typing.AbstractSet[str] = set(),
    only_strings: typing.AbstractSet[str] = set(),
    label: str = "Accuracy",
    only_mode_not_answer: bool = False,
    log: bool = False,
) -> HueResult:
    flats: Slist[ObjectAndMeta] = single_comparison_flat(
        object_model=object_model,
        meta_model=meta_model,
        exp_folder=exp_folder,
        exclude_noncompliant=exclude_noncompliant,
    )
    if not include_identity:
        flats = flats.filter(lambda x: x.response_property != "identity")
    flats = flats.map(lambda x: x.rename_matches_behavior())
    if only_response_properties:
        flats = flats.filter(lambda x: x.response_property in only_response_properties)
    if only_tasks:
        flats = flats.filter(lambda x: x.task in only_tasks)
    # other_evals_to_run = [BiasDetectAreYouAffected, BiasDetectWhatAnswerWithout]
    other_evals_to_run = []
    if other_evals_to_run:
        api = CachedInferenceAPI(api=InferenceAPI(), cache_path="exp/cached_dir")
        results_co = run_from_commands(
            evals_to_run=other_evals_to_run,
            object_and_meta=[(object_model, meta_model)],
            limit=1_000,
            api=api,
        )
        results = asyncio.run(results_co)
        results_formated = results.map(lambda x: x.to_james_analysis_format())
        flats = flats + results_formated
    flats = add_micro_average(flats)
    grouped_by_response_property = flats.group_by(lambda x: (x.response_property, x.object_model, x.meta_model))
    dataframe_row: list[dict] = []
    all_strings = set()
    for group, values in grouped_by_response_property:
        response_property, object_model, meta_model = group
        if only_strings:
            values = values.filter(lambda x: x.string in only_strings)
        if only_mode_not_answer:
            values: Slist[ObjectAndMeta] = values.filter(lambda x: not x.mode_is_correct)
        compliance_rate = values.map(lambda x: x.meta_complied).average_or_raise()
        if response_property == "first_character" and log:
            # dump
            write_jsonl_file_from_basemodel(f"{object_model}_first_character.jsonl", values)
        # print(f"Compliance rate: {compliance_rate}")
        # modal_baselines = modal_baseline(filtered_objects)
        # correct_modes = modal_baselines.map(lambda x: x.meta_predicts_correctly)
        # mode_acc = correct_modes.average_or_raise()
        # print(f"Mode accuracy: {mode_acc}")
        stats: AverageStats = values.map(lambda x: x.meta_predicted_correctly).statistics_or_raise()
        acc = stats.average
        error = stats.upper_confidence_interval_95 - acc
        mode_baseline = values.map(lambda x: x.mode_is_correct).average_or_raise()
        strings = values.map(lambda x: x.string).to_set()
        all_strings.update(strings)

        # acc * 100 1 d.p
        # acc_formatted = f"{acc:1f}"
        # error_formatted = f"{error:1f}"
        # mode_acc = f"{mode_acc:1f}"
        # compliance_rate = f"{compliance_rate:1f}"
        result_row = {
            "response_property": response_property,
            "accuracy": acc,
            "error": error,
            "mode_baseline": mode_baseline,
            # "mode_accuracy": mode_acc,
            "compliance_rate": compliance_rate,
            "count": len(values),
            "object_model": object_model,
            "meta_model": meta_model,
            "label": label,
        }

        dataframe_row.append(result_row)
    return HueResult(strings=all_strings, results=dataframe_row)


# prefinetune_model: str = "gpt-3.5-turbo-1106"
# postfinetune_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
# prefinetune_model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
# postfinetune_model = "projects/351298396653/locations/us-central1/endpoints/8583876282930954240"
# exp_folder = EXP_DIR / "may20_thrifty_sweep"


# half heldout
# prefinetune_model: str = "gpt-3.5-turbo-0125"
# postfinetune_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z"
# # postfinetune_model: str = "gpt-3.5-turbo-0125"
# exp_folder = EXP_DIR / "jun25_leave_out_repsonse_prop"
# show_only: set[str] = {"first_word", "writing_stories/main_character_name", "is_even", "matches_survival_instinct", "matches_myopic_reward", MICRO_AVERAGE_LABEL}

# prefinetune_model = "gpt-4-0613"
# postfinetune_model = "ft:gpt-4-0613:dcevals-kokotajlo:sweep:9RSQ9BDP"
# exp_folder = EXP_DIR / "may20_thrifty_sweep"


# prefinetune_model = "gpt-3.5-turbo-0125"
# postfinetune_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9bMqzmwS"  # gpt_35
# postfinetune_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9bMneYk4" # claude
# exp_folder = EXP_DIR / "jun17_training_on_everything"


# prefinetune_model = "gpt-4o"
# postfinetune_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9bwjNtwH"
# exp_folder = EXP_DIR / "jun17_training_on_everything"
# calculate_shift_results(
#     # shifting="only_same",
#     to_compare_before=prefinetune_model,
#     to_compare_after=postfinetune_model,
#     shifting="only_shifted",
#     prefinetuned_model=prefinetune_model,
#     postfinetuned_model=postfinetune_model,
#     exp_folder=exp_folder,
#     # only_response_properties=show_only,
# )

# object_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
# object_model = "gpt-3.5-turbo-1106"
# postfinetune_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
# meta_model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
# meta_model = "ft:gpt-4-0613:dcevals-kokotajlo:sweep:9RSQ9BDP"
# meta_model = "ft:gpt-4-0613:dcevals-kokotajlo:sweep:9RSPjTJF" # gpt4_on_gpt3.5 predicting gpt3.5
# postfinetune_model = "projects/351298396653/locations/us-central1/endpoints/8583876282930954240"
# exp_folder = EXP_DIR / "may20_thrifty_sweep"


# # half heldout
# object_model: str = "gpt-3.5-turbo-0125"
# meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z"
# # postfinetune_model: str = "gpt-3.5-turbo-0125"
# exp_folder = EXP_DIR / "jun25_leave_out_repsonse_prop"
# show_only: set[str] = {"first_word", "writing_stories/main_character_name", "is_even", "matches_survival_instinct", "matches_myopic_reward", MICRO_AVERAGE_LABEL}


# half heldout evidence 2 gpt-4o
# object_model: str = "gpt-3.5-turbo-0125"
# meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:james-gpt4o-sweep:9eFW626d"
# # postfinetune_model: str = "gpt-3.5-turbo-0125"
# exp_folder = EXP_DIR / "jun25_gpt4o_inference_only"
# show_only: set[str] = {"first_word", "writing_stories/main_character_name", "is_even", "matches_survival_instinct", "matches_myopic_reward", MICRO_AVERAGE_LABEL}

# investigate if baseline works w/o idenity ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:resp-blin:9bMpuarB
# object_model: str = "gpt-3.5-turbo-0125"
# meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:resp-blin:9bMpuarB"
# exp_folder = EXP_DIR / "may20_thrifty_sweep"
# include_identity=False


# investigate finetune strength of precting claude
# object_model: str = "claude-3-sonnet-20240229"
# meta_model = "ft:gpt-4-0613:dcevals-kokotajlo:sweep:9RSQHCmp"
# exp_folder = EXP_DIR / "may20_thrifty_sweep"
# include_identity=True

## held out evidence 2
# # half heldout
# object_model: str = "gpt-3.5-turbo-0125"
# meta_model = "gpt-3.5-turbo-0125"
# exp_folder = EXP_DIR / "jun25_leave_out_repsonse_prop"
# show_only: set[str] = {"first_word", "writing_stories/main_character_name", "is_even", "matches_survival_instinct", "matches_myopic_reward", MICRO_AVERAGE_LABEL}
# include_identity = False

## gpt-4o
# object_model: str = "gpt-3.5-turbo-0125"
# meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:james-gpt4o-sweep:9eFW626d"
# # postfinetune_model: str = "gpt-3.5-turbo-0125"
# exp_folder = EXP_DIR / "jun25_gpt4o_inference_only"


# object_model: str = "gpt-3.5-turbo-0125"
# meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z"
# # postfinetune_model: str = "gpt-3.5-turbo-0125"
# exp_folder = EXP_DIR / "jun25_leave_out_repsonse_prop"
# show_only: set[str] = {"first_word", "writing_stories/main_character_name", "is_even", "matches_survival_instinct", "matches_myopic_reward", MICRO_AVERAGE_LABEL}
# include_identity = True

# object_model: str = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z"
# meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z"
# # postfinetune_model: str = "gpt-3.5-turbo-0125"
# exp_folder = EXP_DIR / "jun25_leave_out_repsonse_prop"
# show_only: set[str] = {
#     "first_word",
#     "writing_stories/main_character_name",
#     "is_even",
#     "matches_survival_instinct",
#     "matches_myopic_reward",
#     MICRO_AVERAGE_LABEL,
# }
# include_identity = False

## gpt-4o predicting A_ft_A
# object_model: str = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z"
# meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:gpt4o-on-ftedgpt35:9g5qGBji"
# # postfinetune_model: str = "gpt-3.5-turbo-0125"
# exp_folder = EXP_DIR / "jun25_leave_out_repsonse_prop"
# include_identity = False
# show_only: set[str] = {
#     "first_word",
#     "writing_stories/main_character_name",
#     "is_even",
#     "matches_survival_instinct",
#     "matches_myopic_reward",
#     MICRO_AVERAGE_LABEL,
# }
# plot_without_comparison(
#     object_model=object_model,
#     meta_model=meta_model,
#     exp_folder=exp_folder,
#     exclude_noncompliant=False,
#     include_identity=include_identity,
#     only_response_properties=show_only,
# )


## gpt-3.5 predicting itself
# object_model: str = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z"
# meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z"
# # postfinetune_model: str = "gpt-3.5-turbo-0125"
# exp_folder = EXP_DIR / "jun25_leave_out_repsonse_prop"
# include_identity = False
# show_only: set[str] = {"first_word", "writing_stories/main_character_name", "is_even", "matches_survival_instinct", "matches_myopic_reward", MICRO_AVERAGE_LABEL}


# 2x more samples ev 1
# object_model: str = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eMKxx3y"
# object_model = "gpt-3.5-turbo-0125"
# object_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eMKxx3y"
# meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eMKxx3y"
# exp_folder = EXP_DIR / "jun26_leave_out_response_prop_2k_samples"
# include_identity = False
# show_only: set[str] = {
#     "first_word",
#     "writing_stories/main_character_name",
#     "is_even",
#     "matches_survival_instinct",
#     "matches_myopic_reward",
#     MICRO_AVERAGE_LABEL,
# }


# investigate may 20
# object_model: str = "gpt-3.5-turbo-1106"
# object_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
# meta_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
# # prefinetune_model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
# # postfinetune_model = "projects/351298396653/locations/us-central1/endpoints/8583876282930954240"
# exp_folder = EXP_DIR / "may20_thrifty_sweep"
# include_identity = False
# show_only: set[str] = set()


## shift with baseline

# prefinetune_model = "gpt-3.5-turbo-1106"
# postfinetune_model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
# exp_folder = EXP_DIR / "may20_thrifty_sweep"
# calculate_shift_results(
#     to_compare_before=prefinetune_model,
#     to_compare_after=postfinetune_model,
#     shifting="only_shifted",
#     prefinetuned_model=prefinetune_model,
#     postfinetuned_model=postfinetune_model,
#     exp_folder=exp_folder,
#     # only_response_properties=show_only,
# )


# evidence 1 on 20 june
# object_model = "gpt-3.5-turbo-0125"
# meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS"
# exp_folder = EXP_DIR / "jun20_training_on_everything"
# first_bar = get_single_hue(
#     object_model="gpt-3.5-turbo-0125",
#     meta_model="gpt-3.5-turbo-0125",
#     exp_folder=exp_folder,
#     include_identity=False,
#     # only_response_properties={},
#     # only_tasks={},
#     label="GPT 3.5 predicting GPT 3.5",
# )
# second_bar  = get_single_hue(
#     object_model=object_model,
#     meta_model=meta_model,
#     exp_folder=exp_folder,
#     include_identity=False,
#     # only_response_properties={},
#     # only_tasks={},
#     label="GPT 3.5_fton_GPT3.5 predicting GPT 3.5 fted on GPT3.5",
# )
# items = (first_bar + second_bar).results
# df = pd.DataFrame(items)
# df.to_csv("response_property_results.csv", index=False)
# calculate_shift_results(
#     shift_before_model=object_model,
#     shift_after_model=meta_model,
#     shifting="all",
#     # include_identity=True,
#     include_identity=False,
#     object_model=object_model,
#     meta_model=meta_model,
#     exp_folder=exp_folder,
# )


def gpt35_on_better_responses():
    exp_folder = EXP_DIR / "jun20_training_on_everything"
    only_response_properties = {
        "first_character",
        "first_word",
        "last_character",
        "is_either_b_or_d",
        "matches behavior",
    }
    object_model = "gpt-3.5-turbo-0125"
    meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS"
    # first_bar = get_single_hue(
    #     object_model="gpt-3.5-turbo-0125",
    #     meta_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS",
    #     exp_folder=exp_folder,
    #     include_identity=False,
    #     only_response_properties=only_response_properties,
    #     # only_tasks={},
    #     label="1) GPT3.5_fton_GPT3.5 predicting GPT3.5",
    #     only_mode_not_answer=False,
    #     log=True,
    # )
    second_bar = get_single_hue(
        object_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS",
        meta_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS",
        exp_folder=exp_folder,
        include_identity=False,
        only_response_properties=only_response_properties,
        # only_tasks={},
        label="2) GPT3.5_fton_GPT3.5 predicting GPT3.5_fton_GPT3.5",
        log=True,
        only_mode_not_answer=False,
    )
    # items = (first_bar + second_bar).results
    # df = pd.DataFrame(items)
    # df.to_csv("response_property_results.csv", index=False)
    # calculate_shift_results(
    #     shift_before_model=object_model,
    #     shift_after_model=meta_model,
    #     shifting="only_shifted",
    #     # include_identity=True,
    #     include_identity=False,
    #     object_model=object_model,
    #     meta_model=meta_model,
    #     exp_folder=exp_folder,
    #     only_response_properties=only_response_properties,
    # )


# gpt35_on_better_responses()


def gpt35_number_triplets():
    exp_folder = EXP_DIR / "5_jul_no_divergence_more_samples"
    only_response_properties = {"first_character"}
    object_model = "gpt-3.5-turbo-0125"
    meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS"
    calculate_evidence_1(
        shift_before_model=object_model,
        shift_after_model=meta_model,
        shifting="only_shifted",
        # include_identity=True,
        include_identity=False,
        object_model=object_model,
        log=True,
        meta_model=meta_model,
        exp_folder=exp_folder,
        only_response_properties=only_response_properties,
    )


def gpt4o_number_triplets():
    exp_folder = EXP_DIR / "5_jul_no_divergence_more_samples_gpt4o"
    only_response_properties = {"first_character", "second_character", "last_character"}
    only_tasks = {"number_triplets"}
    object_model = "gpt-4o-2024-05-13"
    meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9danhPzM"
    calculate_evidence_1(
        shift_before_model=object_model,
        shift_after_model=meta_model,
        shifting="only_shifted",
        # include_identity=True,
        include_identity=False,
        object_model=object_model,
        # log=True,
        meta_model=meta_model,
        adjust_entropy=True,
        exp_folder=exp_folder,
        only_response_properties=only_response_properties,
        only_tasks=only_tasks,
        micro_average=False,
    )


gpt4o_number_triplets()
# gpt35_number_triplets()


def gpt4o_on_better_responses():
    exp_folder = EXP_DIR / "jun20_training_on_everything"
    only_response_properties = {
        "first_character",
        "last_character",
        "is_either_a_or_c",
        "is_either_b_or_d",
        "matches behavior",
    }
    # only_response_properties = set()
    object_model = "gpt-4o-2024-05-13"
    meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9danhPzM"
    first_bar = get_single_hue(
        object_model="gpt-4o-2024-05-13",
        meta_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9danhPzM",
        exp_folder=exp_folder,
        include_identity=False,
        only_response_properties=only_response_properties,
        only_mode_not_answer=True,
        # only_tasks={},
        label="1) GPT 4o_fton_GPT4o predicting GPT4o",
    )
    second_bar = get_single_hue(
        object_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9danhPzM",
        meta_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9danhPzM",
        exp_folder=exp_folder,
        include_identity=False,
        only_response_properties=only_response_properties,
        # only_tasks={},
        only_mode_not_answer=True,
        label="2) GPT 4o_fton_GPT4o predicting GPT4o_fton_GPT4o",
    )
    items = (first_bar + second_bar).results

    df = pd.DataFrame(items)
    df.to_csv("response_property_results.csv", index=False)


#     calculate_shift_results(
#         shift_before_model=object_model,
#         shift_after_model=meta_model,
#         shifting="only_shifted",
#         # include_identity=True,
#         include_identity=False,
#         object_model=object_model,
#         meta_model=meta_model,
#         exp_folder=exp_folder,
#         only_response_properties=only_response_properties,
#     )

# gpt4o_on_better_responses()


def gpt35_compliance():
    exp_folder = EXP_DIR / "jun20_training_on_everything"
    only_response_properties = set()
    first_bar = get_single_hue(
        object_model="gpt-3.5-turbo-0125",
        meta_model="gpt-3.5-turbo-0125",
        exp_folder=exp_folder,
        include_identity=False,
        only_response_properties=only_response_properties,
        # only_tasks={},
        label="1) GPT3.5 predicting GPT3.5",
    )
    second_bar = get_single_hue(
        object_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS",
        meta_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS",
        exp_folder=exp_folder,
        include_identity=False,
        only_response_properties=only_response_properties,
        # only_tasks={},
        label="2) GPT3.5_fton_GPT3.5 predicting GPT3.5_fton_GPT3.5",
    )
    items = (first_bar + second_bar).results
    df = pd.DataFrame(items)
    df.to_csv("response_property_results.csv", index=False)


# gpt35_compliance()
# calculate_shift_results(
#     shift_before_model=object_model,
#     shift_after_model=meta_model,
#     shifting="all",
#     # include_identity=True,
#     include_identity=False,
#     object_model=object_model,
#     meta_model=meta_model,
#     exp_folder=exp_folder,
# )


# async def main_evidence_1():
#     object_model: str = "gpt-3.5-turbo-0125"
#     meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z"
#     # postfinetune_model: str = "gpt-3.5-turbo-0125"
#     exp_folder = EXP_DIR / "jun25_leave_out_repsonse_prop"
#     show_only: set[str] = {
#         "first_word",
#         "writing_stories/main_character_name",
#         "is_even",
#         "matches_survival_instinct",
#         "matches_myopic_reward",
#         MICRO_AVERAGE_LABEL,
#     }
#     setup_environment()

#     await calculate_shift_results(
#         to_compare_before=object_model,
#         to_compare_after=meta_model,
#         shifting="all",
#         prefinetuned_model=object_model,
#         postfinetuned_model=meta_model,
#         exp_folder=exp_folder,
#         only_response_properties=show_only,
#     )
# if __name__ == "__main__":
#     asyncio.run(main())


# 2x more samples., and cross traiend
# object_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eMKxx3y"
# meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eMKxx3y"
# meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:gpt4o-on-ftedgpt35:9g5qGBji"
# exp_folder = EXP_DIR / "jun26_leave_out_response_prop_2k_samples"
# include_identity = True
# show_only = {"identity"}
# show_only: set[str] = {
#     "first_word",
#     "writing_stories/main_character_name",
#     "is_even",
#     "matches_survival_instinct",
#     "matches_myopic_reward",
#     MICRO_AVERAGE_LABEL,
# }
# setup_environment()
# only_tasks = {"writing_stories", "mmlu_cot", "number_triplets", "survival_instinct", "myopic_reward"}
# plot_without_comparison(
#     object_model=object_model,
#     meta_model=meta_model,
#     exp_folder=exp_folder,
#     exclude_noncompliant=False,
#     include_identity=include_identity,
#     only_response_properties=show_only,

# )

# shift on identity

# cross trained on ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z
# model: ft:gpt-4o-2024-05-13:dcevals-kokotajlo:james-gpt4o-sweep:9eFW626d

# object_model: str = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z"
# meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:gpt4o-on-ftedgpt35:9g5qGBji"


# exp_folder = EXP_DIR / "jun25_leave_out_repsonse_prop"
# only_response_properties = {"identity"}
# calculate_shift_results(
#     shift_before_model="gpt-3.5-turbo-0125",
#     shift_after_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
#     shifting="all",
#     # object_model="gpt-3.5-turbo-0125",
#     # meta_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
#     object_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
#     meta_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo:gpt4o-on-ftedgpt35:9g5qGBji",
#     exp_folder=exp_folder,
#     include_identity=True,
#     only_response_properties=only_response_properties,
#     only_task={"wikipedia", "english_words", "daily_dialog", "personal_preferences", "self_referential"},
#     micro_average=False,
# )

# resp_properties = {"identity", "first_character", "last_character", "writing_stories/main_character_name", "is_either_b_or_d", "matches_wealth_seeking", "matches_power_seeking"}
# only_tasks={"wikipedia", "dear_abbie", "english_words", "dialy_dialog", "personal_preferences", "self_referential", "arc_challenge_non_cot", "wealth_seeking", "power_seeking"}
# resp_properties = {"first_character", "last_character", "writing_stories/main_character_name", "is_either_b_or_d", "matches_wealth_seeking", "matches_power_seeking"}


# only_tasks={"wikipedia", "dear_abbie", "english_words", "dialy_dialog", "personal_preferences", "self_referential", "arc_challenge_non_cot", "wealth_seeking", "power_seeking"}
# # resp_properties = {"identity"}
# # only_tasks={"wikipedia", "dear_abbie", "english_words", "dialy_dialog", "personal_preferences", "self_referential"}

# ## Evidence 2, held out prompts
# results = get_single_hue(
#     object_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
#     meta_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo:gpt4o-on-ftedgpt35:9g5qGBji",
#     exp_folder=exp_folder,
#     include_identity=True,
#     only_response_properties=resp_properties,
#     only_tasks=only_tasks,
#     label="Cross Prediction: GPT-4o fted on (fted GPT 3.5) predicting (fted GPT 3.5)",
# ) + get_single_hue(
#     object_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
#     meta_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
#     exp_folder=exp_folder,
#     include_identity=True,
#     only_tasks=only_tasks,
#     only_response_properties=resp_properties,
#     label="Self Prediction: (fted GPT 3.5) predicting (fted GPT 3.5)",
# )
# # dump to df
# df = pd.DataFrame(results)
# df.to_csv("response_property_results.csv", index=False)


def cross_training():
    only_tasks = {"writing_stories", "mmlu_cot", "number_triplets", "survival_instinct", "myopic_reward"}
    resp_properties = set()
    exp_folder = EXP_DIR / "jun25_leave_out_repsonse_prop"

    first_bar = get_single_hue(
        object_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
        meta_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo:gpt4o-on-ftedgpt35:9g5qGBji",
        exp_folder=exp_folder,
        include_identity=True,
        only_response_properties=resp_properties,
        only_tasks=only_tasks,
        label="Cross Prediction: GPT-4o fted on (fted GPT 3.5) predicting (fted GPT 3.5)",
    )
    second_bar = get_single_hue(
        object_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
        meta_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
        exp_folder=exp_folder,
        include_identity=True,
        only_tasks=only_tasks,
        only_response_properties=resp_properties,
        label="Self Prediction: (fted GPT 3.5) predicting (fted GPT 3.5)",
        only_strings=first_bar.strings,
    )
    # run it again to filter lol
    first_bar = get_single_hue(
        object_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
        meta_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo:gpt4o-on-ftedgpt35:9g5qGBji",
        exp_folder=exp_folder,
        include_identity=True,
        only_response_properties=resp_properties,
        only_tasks=only_tasks,
        label="Cross Prediction: GPT-4o fted on (fted GPT 3.5) predicting (fted GPT 3.5)",
        only_strings=second_bar.strings,
    )
    ## Evidence 2, held out prompts
    results = first_bar.results + second_bar.results
    # dump to df
    df = pd.DataFrame(results)
    df.to_csv("response_property_results.csv", index=False)


# cross_training()


def cross_training_mode_not_correct():
    only_tasks = {"writing_stories", "mmlu_cot", "number_triplets", "survival_instinct", "myopic_reward"}
    resp_properties = set()
    exp_folder = EXP_DIR / "jun25_leave_out_repsonse_prop"

    first_bar = get_single_hue(
        object_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
        meta_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo:gpt4o-on-ftedgpt35:9g5qGBji",
        exp_folder=exp_folder,
        include_identity=True,
        only_response_properties=resp_properties,
        only_tasks=only_tasks,
        label="Cross Prediction: GPT-4o fted on (fted GPT 3.5) predicting (fted GPT 3.5)",
        only_mode_not_answer=True,
    )
    second_bar = get_single_hue(
        object_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
        meta_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
        exp_folder=exp_folder,
        include_identity=True,
        only_tasks=only_tasks,
        only_response_properties=resp_properties,
        label="Self Prediction: (fted GPT 3.5) predicting (fted GPT 3.5)",
        only_strings=first_bar.strings,
        only_mode_not_answer=True,
        log=True,
    )
    # run it again to filter lol
    first_bar = get_single_hue(
        object_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
        meta_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo:gpt4o-on-ftedgpt35:9g5qGBji",
        exp_folder=exp_folder,
        include_identity=True,
        only_response_properties=resp_properties,
        only_tasks=only_tasks,
        label="Cross Prediction: GPT-4o fted on (fted GPT 3.5) predicting (fted GPT 3.5)",
        only_strings=second_bar.strings,
        only_mode_not_answer=True,
    )
    ## Evidence 2, held out prompts
    results = first_bar.results + second_bar.results
    # dump to df
    df = pd.DataFrame(results)
    df.to_csv("response_property_results.csv", index=False)


# cross_training_mode_not_correct()


# results_evidence_1 = get_single_hue(
#     object_model="gpt-3.5-turbo-0125",
#     meta_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
#     exp_folder=exp_folder,
#     include_identity=True,
#     only_response_properties=resp_properties,
#     only_tasks=only_tasks,
#     label="Training Prediction: (fted GPT 3.5) predicting (GPT 3.5)",
# ) + get_single_hue(
#     object_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
#     meta_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
#     exp_folder=exp_folder,
#     include_identity=True,
#     only_tasks=only_tasks,
#     only_response_properties=resp_properties,
#     label="Self Prediction: (fted GPT 3.5) predicting (fted GPT 3.5)",
# )
# # dump to df
# df_evidence_1 = pd.DataFrame(results_evidence_1)
# df_evidence_1.to_csv("evidence_1_results.csv", index=False)


# results_evidence_1 = get_single_hue(
#     object_model="gpt-4o-2024-05-13",
#     meta_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9gXElCb8",
#     exp_folder=exp_folder,
#     include_identity=True,
#     only_response_properties=resp_properties,
#     only_tasks=only_tasks,
#     label="1) Training Prediction: (fted GPT 3.5) predicting (GPT 3.5)",
# ) + get_single_hue(
#     object_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9gXElCb8",
#     meta_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9gXElCb8",
#     exp_folder=exp_folder,
#     include_identity=True,
#     only_tasks=only_tasks,
#     only_response_properties=resp_properties,
#     label="2) Actual Prediction: (fted GPT 3.5) predicting (fted GPT 3.5)",
# )
# # dump to df
# df_evidence_1 = pd.DataFrame(results_evidence_1)
# df_evidence_1.to_csv("evidence_1_results.csv", index=False)


## YOLO run

# exp_folder = EXP_DIR / "3_jul_yolo_more_samples"
# results_evidence_1 = get_single_hue(
#     object_model="gpt-3.5-turbo-0125",
#     meta_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9gsWGQZZ",
#     exp_folder=exp_folder,
#     include_identity=True,
#     # only_response_properties=resp_properties,
#     # only_tasks=only_tasks,
#     label="1) Training Prediction: (fted GPT 3.5) predicting (GPT 3.5)",
# ) + get_single_hue(
#     object_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9gsWGQZZ",
#     meta_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9gsWGQZZ",
#     exp_folder=exp_folder,
#     include_identity=True,
#     # only_tasks=only_tasks,
#     # only_response_properties=resp_properties,
#     label="2) Actual Prediction: (fted GPT 3.5) predicting (fted GPT 3.5)",
# )
# # dump to df
# df_evidence_1 = pd.DataFrame(results_evidence_1.results)
# df_evidence_1.to_csv("evidence_1_results.csv", index=False)

# object_model="gpt-3.5-turbo-0125"
# meta_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9gsWGQZZ"
# calculate_shift_results(
#     shift_before_model=object_model,
#     shift_after_model=meta_model,
#     shifting="all",
#     object_model=object_model,
#     meta_model=meta_model,
#     exp_folder=exp_folder,
#     include_identity=True,
#     # on trained tasks
#     only_task={"wikipedia", "dear_abbie", "english_words", "daily_dialog", "personal_preferences", "self_referential"},
# )
