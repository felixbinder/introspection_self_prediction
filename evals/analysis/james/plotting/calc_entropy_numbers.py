# open in pandas exp/jun20_training_on_everything/object_level_ft_gpt-3.5-turbo-0125_dcevals-kokotajlo__9da15ENS_object_level_minimal_prompt_number_triplets_val_task__note/data0.csv

import numpy as np
import pandas
from git import Sequence
from scipy import stats
from slist import AverageStats, Group, Slist

from evals.analysis.james.object_meta import ObjectAndMeta
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from other_evals.counterfactuals.api_utils import (
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)


def calculate_entropy(string_list: Sequence[str]) -> float:
    # Get unique values and their probabilities
    _, counts = np.unique(string_list, return_counts=True)
    probabilities = counts / len(string_list)

    # Calculate entropy
    entropy = stats.entropy(probabilities, base=2)

    return entropy


#
before_finetune = read_jsonl_file_into_basemodel("gpt-4o-2024-05-13_first_character.jsonl", ObjectAndMeta)

# ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS_first_character.jsonl
after_finetune = read_jsonl_file_into_basemodel(
    "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9danhPzM_first_character.jsonl", ObjectAndMeta
)

only_shifted = True
if only_shifted:
    before_finetune = before_finetune.filter(lambda x: x.shifted == "shifted")
    # filter out zeroes so its easier to compare
    before_finetune = before_finetune.filter(lambda x: x.object_response_property_answer != "0")
    after_finetune = after_finetune.filter(lambda x: x.shifted == "shifted")
    after_finetune = after_finetune.filter(lambda x: x.object_response_property_answer != "0")

before_finetune_object_response_property_answer: list[str] = before_finetune.map(
    lambda x: x.object_response_property_answer
)
after_finetune_object_response_property_answer: list[str] = after_finetune.map(
    lambda x: x.object_response_property_answer
)
entropy_before_finetune = calculate_entropy(before_finetune_object_response_property_answer)
entropy_after_finetune = calculate_entropy(after_finetune_object_response_property_answer)
print(f"Entropy before fine-tuning: {entropy_before_finetune:.2f}")
print(f"Entropy after fine-tuning: {entropy_after_finetune:.2f}")

# map of counts
before_finetune_map: Slist[Group[str, Slist[ObjectAndMeta]]] = before_finetune.group_by(
    lambda x: x.object_response_property_answer
).sort_by(lambda x: x.key)
after_finetune_map: Slist[Group[str, Slist[ObjectAndMeta]]] = after_finetune.group_by(
    lambda x: x.object_response_property_answer
).sort_by(lambda x: x.key)
# counts per unique string
before_finetune_counts: Slist[Group[str, int]] = before_finetune_map.map_on_group_values(len)
after_finetune_counts: Slist[Group[str, int]] = after_finetune_map.map_on_group_values(len).sort_by(lambda x: x.key)

new_before_finetune = Slist()
new_after_finetune = Slist()
for before_group, after_group in before_finetune_map.zip(after_finetune_map):
    assert before_group.key == after_group.key
    before_count = before_group.values.length
    after_count = after_group.values.length
    print(f"Number: {before_group.key} Before count: {before_count} After count: {after_count}")
    min_count = min(before_count, after_count)
    new_before_finetune.extend(before_group.values.shuffle("42").take(min_count))
    new_after_finetune.extend(after_group.values.shuffle("42").take(min_count))

# recalculate entropy
new_before_finetune_ans: list[str] = new_before_finetune.map(lambda x: x.object_response_property_answer)
new_after_finetune_ans: list[str] = new_after_finetune.map(lambda x: x.object_response_property_answer)
entropy_new_before_finetune = calculate_entropy(new_before_finetune_ans)
entropy_new_after_finetune = calculate_entropy(new_after_finetune_ans)
print(f"Entropy before fine-tuning (filtered): {entropy_new_before_finetune:.2f}")
print(f"Entropy after fine-tuning (filtered): {entropy_new_after_finetune:.2f}")


write_jsonl_file_from_basemodel("entropy_new_before_finetune.jsonl", new_before_finetune)
write_jsonl_file_from_basemodel("entropy_new_after_finetune.jsonl", new_after_finetune)


def make_row(values: Slist[ObjectAndMeta], label: str) -> dict:
    stats: AverageStats = values.map(lambda x: x.meta_predicted_correctly).statistics_or_raise()
    acc = stats.average
    error = stats.upper_confidence_interval_95 - acc
    new_mode = values.map(lambda x: x.object_response_property_answer).mode_or_raise()
    mode_baseline = values.map(lambda x: x.object_response_property_answer == new_mode).average_or_raise()
    response_property = values.map(lambda x: x.response_property).first_or_raise()
    compliance_rate = values.map(lambda x: x.meta_complied and x.object_complied).average_or_raise()
    object_model = values.map(lambda x: x.object_model).first_or_raise()
    meta_model = values.map(lambda x: x.meta_model).first_or_raise()
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
        # "label": "Before finetuned (filtered to get lower entropy)",
        "label": label,
    }

    return result_row


rows: list[dict] = [
    make_row(new_before_finetune, "1) Before finetuned, adjusted"),
    make_row(new_after_finetune, "2) After finetuned, adjusted"),
]
df = pandas.DataFrame(rows)
create_chart(df=df, title="Accuracy after adjusting", _sorted_properties=None)
