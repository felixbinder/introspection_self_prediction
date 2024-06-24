

from pathlib import Path
import pandas as pd

from slist import AverageStats, Slist
from evals.analysis.loading_data import FlatObjectMeta, LoadedObject, ObjectMetaPair, filter_for_specific_models, flat_object_meta, load_meta_dfs, only_same_object_properties, only_shifted_object_properties
from evals.locations import EXP_DIR
from other_evals.counterfactuals.api_utils import write_jsonl_file_from_basemodel


def get_flat_object_meta(
    prefinetuned_model: str = "gpt-3.5-turbo-1106",
    postfinetuned_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2",
    only_shifted: bool = True,
) -> Slist[FlatObjectMeta]:
    # If shifted_only is True, only compares objects that have shifted.
    # If shifted_only is False, only compares objects that are the same.
    # exp_folder = EXP_DIR /"evaluation_suite"
    exclude_noncompliant = False
    exp_folder: Path = EXP_DIR / "may20_thrifty_sweep"

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

            switched_objects = (
                only_shifted_object_properties(
                    prefinetuned_objects=prefinetuned_objects.filter(
                        lambda x: x.response_property == response_property
                    ),
                    postfinetuned_objects=postfinetuned_objects.filter(
                        lambda x: x.response_property == response_property
                    ),
                )
                if only_shifted
                else only_same_object_properties(
                    prefinetuned_objects=prefinetuned_objects.filter(
                        lambda x: x.response_property == response_property
                    ),
                    postfinetuned_objects=postfinetuned_objects.filter(
                        lambda x: x.response_property == response_property
                    ),
                )
            )

            print(f"Got {len(switched_objects)} switched objects")
            objects_in_switch: Slist[LoadedObject] = new_filtered_objects.filter(lambda x: x.string in switched_objects)
            if response_property == "is_even":
                write_jsonl_file_from_basemodel(f"is_even_objects_{object_model}.jsonl", objects_in_switch)

            print(f"Got {len(objects_in_switch)} to evaluate for {object_model} for {response_property}")

            metas_in_switch = new_filtered_metas.filter(lambda x: x.string in switched_objects)
            assert len(objects_in_switch) > 0, "No objects found in switch"
            assert len(metas_in_switch) > 0, "No metas found in switch"

            # filtered_objects, filtered_metas = objects, metas
            print(
                f"Comparing {len(objects_in_switch)} vs {len(metas_in_switch)} to evaluate for {object_model} for {response_property}"
            )
            compared = flat_object_meta(objects_in_switch, metas_in_switch)
            result_rows.extend(compared)

    return result_rows


def calculate_shift_results(
    prefinetuned_model: str = "gpt-3.5-turbo-1106",
    postfinetuned_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2",
    only_shifted: bool = True,
) -> None:
    flats = get_flat_object_meta(
        prefinetuned_model=prefinetuned_model, postfinetuned_model=postfinetuned_model, only_shifted=only_shifted
    )
    grouped_by_response_property = flats.group_by(lambda x: (x.response_property,x.object_model,x.meta_model))
    dataframe_row: list[dict]= []
    for group , values in grouped_by_response_property:
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


        # acc * 100 1 d.p
        # acc_formatted = f"{acc:1f}"
        # error_formatted = f"{error:1f}"
        # mode_acc = f"{mode_acc:1f}"
        # compliance_rate = f"{compliance_rate:1f}"
        label = "Predicting behavior before training" if object_model == prefinetuned_model else "Predicting behavior after training"
        result_row = {
            "response_property": response_property,
            "accuracy": acc,
            "error": error,
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

calculate_shift_results(only_shifted=False)
        
