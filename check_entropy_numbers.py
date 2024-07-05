# open in pandas exp/jun20_training_on_everything/object_level_ft_gpt-3.5-turbo-0125_dcevals-kokotajlo__9da15ENS_object_level_minimal_prompt_number_triplets_val_task__note/data0.csv
import numpy as np
from git import Sequence
from scipy import stats
from slist import Group, Slist

from evals.analysis.james.object_meta import FlatObjectMeta
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
before_finetune = read_jsonl_file_into_basemodel("gpt-3.5-turbo-0125_first_character.jsonl", FlatObjectMeta)

# ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS_first_character.jsonl
after_finetune = read_jsonl_file_into_basemodel(
    "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS_first_character.jsonl", FlatObjectMeta
)

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

# make a map of counts per unique string
before_finetune_map: Slist[Group[str, int]] = before_finetune.group_by(
    lambda x: x.object_response_property_answer
).map_on_group_values(len)
# which 3 string has lowest count?
min_count: set[str] = before_finetune_map.sort_by(lambda x: x.values).take(3).map(lambda x: x.key).to_set()
print(f"Strings with the lowest count: {min_count}")
# remove the string with the lowest count
before_finetune_filtered = before_finetune.filter(lambda x: x.object_response_property_answer not in min_count)
filtered_strings = before_finetune_filtered.map(lambda x: x.object_response_property_answer)
# recalculate entropy
entropy_before_finetune_filtered = calculate_entropy(filtered_strings)
print(f"Entropy before fine-tuning (filtered): {entropy_before_finetune_filtered:.2f}")
# write to new json
write_jsonl_file_from_basemodel("gpt-3.5-turbo-0125_first_character_filtered.jsonl", before_finetune_filtered)
