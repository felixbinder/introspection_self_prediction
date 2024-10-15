import pandas as pd
from slist import Slist

from evals.analysis.james.james_analysis import single_comparison_flat
from evals.analysis.james.object_meta import ObjectAndMeta

from evals.locations import EXP_DIR
from other_evals.counterfactuals.api_utils import write_jsonl_file_from_basemodel


from typing import Literal
from pydantic import BaseModel, TypeAdapter


class ChatMessage(BaseModel):
    role: str
    content: str


class StandardTestData(BaseModel):
    # The original question from the dataset e.g. mmlu
    original_question: str
    original_dataset: str
    object_level_prompt: list[ChatMessage]
    hypothetical_prompt: list[ChatMessage]
    # e.g. first_word, second_character
    behavioral_property: str
    # only exists for ethical stance
    option_matching_ethical_stance: str | None
    
def parsed_string_into_list_chat_message(parsed: str) -> list[ChatMessage]:
    ta  = TypeAdapter(dict[str,list[ChatMessage]])
    result =  ta.validate_json(parsed)
    assert "messages" in result
    return result["messages"]


def remove_redundant_sys_prompt(chat: list[ChatMessage]) -> list[ChatMessage]:
    # remove the empty sys prompts
    return [x for x in chat if x.content != ""]



def convert_to_dump(test_data: ObjectAndMeta) -> StandardTestData:
    object_full = test_data.object_full_prompt
    assert object_full is not None
    hypo_full = test_data.meta_full_prompt
    assert hypo_full is not None

    return StandardTestData(
        original_question=test_data.string,
        original_dataset=test_data.task,
        object_level_prompt=remove_redundant_sys_prompt(parsed_string_into_list_chat_message(object_full)),
        hypothetical_prompt=remove_redundant_sys_prompt(parsed_string_into_list_chat_message(hypo_full)),
        behavioral_property=test_data.response_property,
        option_matching_ethical_stance=None,
    )



def dump_all():
    """
    --val_tasks='{"survival_instinct": ["matches_survival_instinct"], "myopic_reward": ["matches_myopic_reward"], "animals_long": ["first_character", "second_character", "third_character", "first_and_second_character", "first_word", "second_word", "starts_with_vowel", "third_word"], "mmlu_non_cot": ["is_either_a_or_c", "is_either_b_or_d"], "english_words_long": ["first_character", "second_character", "third_character", "first_and_second_character", "first_word", "second_word", "starts_with_vowel", "third_word"], "stories_sentences": ["first_character", "second_character", "third_character", "first_and_second_character", "first_word", "second_word", "starts_with_vowel", "third_word"]}'
    """
    only_tasks = set(
        [
            "animals_long",
            "survival_instinct",
            "myopic_reward",
            "mmlu_non_cot",
            "stories_sentences",
            "english_words_long",
        ]
    )
    # only_tasks = {}
    resp_properties = set()
    exp_folder = EXP_DIR / "test_csv"

    results: Slist[ObjectAndMeta] = single_comparison_flat(
        object_model="test",
        meta_model="test",
        exp_folder=exp_folder,
        only_tasks=only_tasks,
        exclude_noncompliant=False,
    )
    # dump 
    write_jsonl_file_from_basemodel("dump.jsonl", results.map(convert_to_dump))


dump_all()
# def test_parse_one_file():
#     with open("dataset_dumps/test/distractor_fact/mmlu_distractor_fact.jsonl", "r") as f:
#         for line in f.readlines():
#             # read into the basemodel
#             parsed = StandardTestData.model_validate_json(line)
#             print(parsed.biased_question)
#             print(f"Biased towards: {parsed.biased_option}")
#             print(f"Ground truth: {parsed.ground_truth}")

