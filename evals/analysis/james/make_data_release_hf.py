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
    object_level_prompt: str
    hypothetical_prompt: str
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

def first_user_message(chat: list[ChatMessage]) -> str:
    first = [x for x in chat if x.role == "user"][0]
    return first.content


def convert_to_dump(test_data: ObjectAndMeta) -> DataRow:
    object_full = test_data.object_full_prompt
    assert object_full is not None
    hypo_full = test_data.meta_full_prompt
    assert hypo_full is not None

    return DataRow(
        original_question=test_data.string,
        original_dataset=test_data.task,
        object_level_prompt=first_user_message(parsed_string_into_list_chat_message(object_full)),
        hypothetical_prompt=first_user_message(parsed_string_into_list_chat_message(hypo_full)),
        behavioral_property=test_data.response_property,
        option_matching_ethical_stance=test_data.target if test_data.response_property == "matches_target" else None,
    )


def dump_all():
    """
    "survival_instinct": ["matches_survival_instinct"],
    "myopic_reward": ["matches_myopic_reward"],
    "animals_long": [
        "first_character",
        "second_character",
        "third_character",
        "first_word",
        "second_word",
        "starts_with_vowel",
        "third_word",
    ],
    "mmlu_non_cot": ["is_either_a_or_c", "is_either_b_or_d"],
    "english_words_long": [
        "first_character",
        "second_character",
        "third_character",
        "first_word",
        "second_word",
        "starts_with_vowel",
        "third_word",
    ],
    "stories_sentences": [
        "first_character",
        "second_character",
        "third_character",
        "first_word",
        "second_word",
        "starts_with_vowel",
        "third_word",
    ],
    """
    only_tasks = set(
        [
            "survival_instinct",
            "myopic_reward",
            "animals_long",
            "mmlu_non_cot",
            "stories_sentences",
            "english_words_long",
        ]
    )
    # only_tasks = {}
    resp_properties = set()
    exp_folder = EXP_DIR / "test_dump"

    results: Slist[ObjectAndMeta] = single_comparison_flat(
        object_model="test",
        meta_model="test",
        exp_folder=exp_folder,
        only_tasks=only_tasks,
        exclude_noncompliant=False,
    )
    # dump
    converted: Slist[DataRow] = results.map(convert_to_dump).map(lambda x: x.rename_properties())
    folder = Path("dataset_release/test")
    # write_jsonl_file_from_basemodel(folder / "test.jsonl", converted)
    # group by behavioral property as the subset
    # groups = converted.group_by(lambda x: f"{x.original_dataset}_{x.behavioral_property}.jsonl")
    groups = converted.group_by(lambda x: f"{x.behavioral_property}.jsonl")
    for group, test_subset in groups:
        write_jsonl_file_from_basemodel(folder / group, test_subset)


dump_all()
