from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel
from scipy import stats
from slist import AverageStats, Group, Slist

from evals.analysis.loading_data import (
    ComparedMeta,
    LoadedMeta,
    LoadedObject,
    ObjectMetaPair,
    clean_for_comparison,
    filter_for_specific_models,
    load_meta_dfs,
    modal_baseline,
)
from evals.locations import EXP_DIR


class FlatObjectMeta(BaseModel):
    task: str
    string: str
    meta_predicted_correctly: bool
    response_property: str
    meta_model: str
    object_model: str
    object_response_property_answer: str
    object_response_raw_response: str
    object_complied: bool
    meta_complied: bool
    shifted: Literal["shifted", "same", "not_compliant", "not_calculated"]

    def rename_matches_behavior(self):
        # if "matches" in self.response_property, rename to "matches behavior"
        new = self.model_copy()
        if "matches" in self.response_property:
            new.response_property = "matches behavior"
        return new


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
    add_micro_average: bool,
    shifted_result: ShiftResult | None = None,
) -> Slist[FlatObjectMeta]:
    # group objects by task + string + response_property
    objects_grouped: dict[tuple[str, str, str], Slist[LoadedObject]] = objects.group_by(
        lambda x: (x.task, x.string, x.response_property)
    ).to_dict()
    compared: Slist[FlatObjectMeta] = Slist()
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
                FlatObjectMeta(
                    meta_predicted_correctly=predicted_correctly,
                    task=meta.task,
                    string=meta.string,
                    response_property=meta.response_property,
                    meta_model=meta.meta_model,
                    object_model=obj.object_model,
                    object_response_property_answer=obj.response_property_answer,
                    object_response_raw_response=obj.raw_response,
                    object_complied=obj.compliance,
                    meta_complied=meta.compliance,
                    shifted=shifted,
                )
            )
            if add_micro_average:
                compared.append(
                    FlatObjectMeta(
                        meta_predicted_correctly=predicted_correctly,
                        task=meta.task,
                        string=meta.string,
                        response_property="zMicro Average",
                        meta_model=meta.meta_model,
                        object_model=obj.object_model,
                        object_response_property_answer=obj.response_property_answer,
                        object_response_raw_response=obj.raw_response,
                        object_complied=obj.compliance,
                        meta_complied=meta.compliance,
                        shifted=shifted,
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


def get_flat_object_meta(
    exp_folder: Path,
    prefinetuned_model: str = "gpt-3.5-turbo-1106",
    postfinetuned_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2",
    add_micro_average: bool = True,
    exclude_noncompliant: bool = False,
) -> Slist[FlatObjectMeta]:
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
    assert len(prefinetuned_objects) > 0, "No prefinetuned objects found"
    assert len(postfinetuned_objects) > 0, "No postfinetuned objects found"

    result_rows: Slist[FlatObjectMeta] = Slist()
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

        # # remove identity
        # assert "identity" in unique_response_properties, "Identity not found in response properties"
        # unique_response_properties.remove("identity")

        for response_property in unique_response_properties:
            new_filtered_objects = filtered_objects.filter(
                lambda filtered_obj: filtered_obj.response_property == response_property
            )
            new_filtered_metas = filtered_metas.filter(lambda x: x.response_property == response_property)

            assert len(new_filtered_objects) > 0, f"No objects found for {response_property} for {object_model}"
            assert len(new_filtered_metas) > 0, f"No metas found for {response_property}"

            switched_objects: ShiftResult = only_shifted_object_properties(
                prefinetuned_objects=prefinetuned_objects.filter(lambda x: x.response_property == response_property),
                postfinetuned_objects=postfinetuned_objects.filter(lambda x: x.response_property == response_property),
            )

            compared = flat_object_meta(
                new_filtered_objects,
                new_filtered_metas,
                add_micro_average=add_micro_average,
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


def calculate_shift_results(
    prefinetuned_model: str = "gpt-3.5-turbo-1106",
    postfinetuned_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2",
    shifting: Literal["all", "only_shifted", "only_same"] = "all",
    exp_folder: Path = EXP_DIR / "may20_thrifty_sweep",
    exclude_noncompliant: bool = False,
) -> None:
    flats: Slist[FlatObjectMeta] = get_flat_object_meta(
        prefinetuned_model=prefinetuned_model,
        postfinetuned_model=postfinetuned_model,
        exp_folder=exp_folder,
        exclude_noncompliant=exclude_noncompliant,
    ).map(lambda x: x.rename_matches_behavior())
    if shifting == "only_shifted":
        flats = flats.filter(lambda x: x.shifted == "shifted")
    if shifting == "only_same":
        flats = flats.filter(lambda x: x.shifted == "same")
    elif shifting == "all":
        pass

    grouped_by_response_property = flats.group_by(lambda x: (x.response_property, x.object_model, x.meta_model))
    dataframe_row: list[dict] = []
    for group, values in grouped_by_response_property:
        response_property, object_model, meta_model = group

        compliance_rate = values.map(lambda x: x.meta_complied).average_or_raise()
        # print(f"Compliance rate: {compliance_rate}")
        # modal_baselines = modal_baseline(filtered_objects)
        # correct_modes = modal_baselines.map(lambda x: x.meta_predicts_correctly)
        # mode_acc = correct_modes.average_or_raise()
        # print(f"Mode accuracy: {mode_acc}")
        stats: AverageStats = values.map(lambda x: x.meta_predicted_correctly).statistics_or_raise()
        acc = stats.average
        error = stats.upper_confidence_interval_95 - acc
        shift_percentage = values.map(lambda x: x.shifted == "shifted").average_or_raise()

        # acc * 100 1 d.p
        # acc_formatted = f"{acc:1f}"
        # error_formatted = f"{error:1f}"
        # mode_acc = f"{mode_acc:1f}"
        # compliance_rate = f"{compliance_rate:1f}"
        label = (
            "Predicting behavior before training"
            if object_model == prefinetuned_model
            else "Predicting behavior after training"
        )
        result_row = {
            "response_property": response_property,
            "accuracy": acc,
            "error": error,
            "shifted": shift_percentage,
            # "mode_accuracy": mode_acc,
            "compliance_rate": compliance_rate,
            "count": len(values),
            "object_model": object_model,
            "meta_model": meta_model,
            "label": label,
        }

        dataframe_row.append(result_row)

    df = pd.DataFrame(dataframe_row)
    # to csv inspect_response_property_results.csv
    df.to_csv("response_property_results.csv")


# prefinetune_model: str = "gpt-3.5-turbo-1106"
# postfinetune_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
prefinetune_model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
postfinetune_model = "projects/351298396653/locations/us-central1/endpoints/8583876282930954240"
exp_folder = EXP_DIR / "may20_thrifty_sweep"

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
calculate_shift_results(
    # shifting="only_same",
    shifting="all",
    prefinetuned_model=prefinetune_model,
    postfinetuned_model=postfinetune_model,
    exp_folder=exp_folder,
)