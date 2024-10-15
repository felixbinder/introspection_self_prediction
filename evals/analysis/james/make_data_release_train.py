from pathlib import Path

from pydantic import BaseModel, TypeAdapter
from slist import Slist

from evals.analysis.james.james_analysis import single_comparison_flat
from evals.analysis.james.object_meta import ObjectAndMeta
from evals.locations import EXP_DIR
from other_evals.counterfactuals.api_utils import write_jsonl_file_from_basemodel


class ChatMessage(BaseModel):
    role: str
    content: str


rename_map = {
    "matches_target": "ethical_stance",
    "is_either_a_or_c": "among_a_or_c",
    "is_either_b_or_d": "among_b_or_d",
}


class DataRow(BaseModel):
    # The original question from the dataset e.g. mmlu
    original_question: str
    original_dataset: str
    object_level_prompt: list[ChatMessage]
    hypothetical_prompt: list[ChatMessage]
    # e.g. first_word, second_character
    behavioral_property: str
    # only exists for ethical stance behavioral properties
    option_matching_ethical_stance: str | None

    def rename_properties(self):
        new = self.model_copy()
        if new.behavioral_property in rename_map:
            new.behavioral_property = rename_map[new.behavioral_property]
        return new


def parsed_string_into_list_chat_message(parsed: str) -> list[ChatMessage]:
    ta = TypeAdapter(dict[str, list[ChatMessage]])
    result = ta.validate_json(parsed)
    assert "messages" in result
    return result["messages"]


def remove_redundant_sys_prompt(chat: list[ChatMessage]) -> list[ChatMessage]:
    # remove the empty sys prompts
    return [x for x in chat if x.content != ""]


def convert_to_dump(test_data: ObjectAndMeta) -> DataRow:
    object_full = test_data.object_full_prompt
    assert object_full is not None
    hypo_full = test_data.meta_full_prompt
    assert hypo_full is not None

    return DataRow(
        original_question=test_data.string,
        original_dataset=test_data.task,
        object_level_prompt=remove_redundant_sys_prompt(parsed_string_into_list_chat_message(object_full)),
        hypothetical_prompt=remove_redundant_sys_prompt(parsed_string_into_list_chat_message(hypo_full)),
        behavioral_property=test_data.response_property,
        option_matching_ethical_stance=test_data.target,
    )


def dump_all():
    """
    # "wikipedia_long": [
    #     "first_character",
    #     "second_character",
    #     "third_character",
    #     "first_and_second_character",
    #     "first_word",
    #     "second_word",
    #     "starts_with_vowel",
    #     "third_word",
    # ],
    # "wealth_seeking": ["matches_wealth_seeking"],
    # "power_seeking": ["matches_power_seeking"],
    # "arc_challenge_non_cot": ["identity", "is_either_a_or_c", "is_either_b_or_d"],
    # "countries_long": [
    #     "first_character",
    #     "second_character",
    #     "third_character",
    #     "first_and_second_character",
    #     "first_word",
    #     "second_word",
    #     "starts_with_vowel",
    #     "third_word",
    # ],
    # "colors_long": [
    #     "first_character",
    #     "second_character",
    #     "third_character",
    #     "first_and_second_character",
    #     "first_word",
    #     "second_word",
    #     "starts_with_vowel",
    #     "third_word",
    # ],
    # "numbers": [
    #     "is_even_direct",
    #     # "is_even" # broken, but we only need is_even_direct
    # ],
    """
    only_tasks = set(
        [
            "wikipedia_long",
            "wealth_seeking",
            "power_seeking",
            "arc_challenge_non_cot",
            "countries_long",
            "colors_long",
            "numbers",
        ]
    )
    # only_tasks = {}
    resp_properties = set()
    exp_folder = EXP_DIR / "train_dump"

    results: Slist[ObjectAndMeta] = single_comparison_flat(
        object_model="test",
        meta_model="test",
        exp_folder=exp_folder,
        only_tasks=only_tasks,
        exclude_noncompliant=False,
    )
    # dump
    converted: Slist[DataRow] = results.map(convert_to_dump).map(lambda x: x.rename_properties())
    folder = Path("dataset_release/train")
    # group by dataset + behavioral property
    groups = converted.group_by(lambda x: f"{x.original_dataset}_{x.behavioral_property}.jsonl")
    for group, items in groups:
        write_jsonl_file_from_basemodel(folder / group, items)


dump_all()
