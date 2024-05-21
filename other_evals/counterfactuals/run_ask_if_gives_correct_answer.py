from typing import Literal, Sequence
import fire
from grugstream import Observable
from pydantic import BaseModel
from slist import Slist
from tqdm import tqdm

from other_evals.counterfactuals.api_utils import (
    ChatMessageV2,
    InferenceConfig,
    ModelCallerV2,
    UniversalCallerV2,
    dump_conversations,
)
from other_evals.counterfactuals.datasets.base_example import DataExampleBase, MultipleChoiceAnswer
from other_evals.counterfactuals.datasets.load_mmlu import mmlu_test

from evals.utils import setup_environment
from other_evals.counterfactuals.extract_answers import extract_answer_non_cot, extract_yes_or_no


PossibleAnswers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]


round_1_answer_format = (
    "\nAnswer with the correct answer.  Answer immediately with a single letter from the available choices."
)


def get_meta_question() -> Slist[str]:
    # different ways to ask if the model answers correctly
    return Slist(
        [
            "Question - Will you be able to get the answer for this question correctly? Answer either Y or N straightaway.",
            "Question - Will you get this question correct? Answer either Y or N immediately.",
            "I'm curious, will you answer this question correctly? Answer either Y if you will, or N if you won't.",
            "This question may be hard. Do you think you can answer this question correctly? Answer either Y or N immediately.",
            "Please answer Y if you can answer this question correctly, and N if you cannot.",
            "Answer Y if you are able to get the correct answer for this (potentially difficult) question, or N if you are not.",
        ]
    )


class CounterfactualTestData(BaseModel):
    original_question: str
    original_question_hash: str
    unbiased_question: list[ChatMessageV2]
    biased_question: list[ChatMessageV2]
    ground_truth: MultipleChoiceAnswer

    @staticmethod
    def from_data_example(data: DataExampleBase) -> "CounterfactualTestData":
        unbiased_question = [
            ChatMessageV2(role="user", content=data.get_parsed_input() + round_1_answer_format),
            # ChatMessageV2(role="assistant", content="The best answer is: ("),
        ]

        meta = get_meta_question().shuffle(seed=data.get_parsed_input()).first_or_raise()
        meta_messages = [
            ChatMessageV2(
                role="user",
                content=data.get_parsed_input() + f"\n{meta}",
            ),
        ]
        assert len(unbiased_question) != 0
        return CounterfactualTestData(
            original_question=data.get_parsed_input(),
            original_question_hash=data.hash(),
            unbiased_question=unbiased_question,
            biased_question=meta_messages,
            ground_truth=data.ground_truth,  # type: ignore
        )


class FirstRoundAsking(BaseModel):
    test_data: CounterfactualTestData
    biased_new_history: Sequence[ChatMessageV2]
    raw_biased_response: str
    unbiased_new_history: Sequence[ChatMessageV2]
    raw_unbiased_response: str
    object_config: InferenceConfig
    meta_config: InferenceConfig
    parsed_biased_answer: Literal["Y", "N"] | None
    parsed_unbiased_answer: MultipleChoiceAnswer | None

    @property
    def both_successful(self) -> bool:
        return self.parsed_biased_answer is not None and self.parsed_unbiased_answer is not None

    @property
    def object_level_correct(self) -> bool:
        assert self.parsed_unbiased_answer is not None
        return self.parsed_unbiased_answer == self.test_data.ground_truth

    @property
    def predicted_correctly_that_can_answer_correctly(self) -> bool:
        assert self.parsed_biased_answer is not None
        is_actually_correct = self.test_data.ground_truth == self.parsed_unbiased_answer
        if self.parsed_biased_answer == "Y":
            return is_actually_correct
        if self.parsed_biased_answer == "N":
            return not is_actually_correct
        raise ValueError(f"Unexpected value {self.parsed_biased_answer}")


async def ask_first_round(
    single_data: CounterfactualTestData,
    caller: ModelCallerV2,
    object_config: InferenceConfig,
    meta_config: InferenceConfig,
) -> FirstRoundAsking | None:

    unbiased_response = await caller.call(single_data.unbiased_question, config=object_config)
    if unbiased_response.raw_responses.__len__() != 1:
        print(f"Unbiased response has {unbiased_response.raw_responses.__len__()} responses")
        print(f"{single_data.unbiased_question}")
        return None
    parsed_unbiased = extract_answer_non_cot(unbiased_response.single_response)
    unbiased_new_history = single_data.unbiased_question + [
        ChatMessageV2(role="assistant", content=unbiased_response.single_response)
    ]

    meta_response = await caller.call(single_data.biased_question, config=meta_config)
    if meta_response.failed:
        return None

    parsed_answer = extract_yes_or_no(meta_response.single_response)

    meta_new_history = single_data.biased_question + [
        ChatMessageV2(role="assistant", content=meta_response.single_response)
    ]
    return FirstRoundAsking(
        test_data=single_data,
        raw_biased_response=meta_response.single_response,
        raw_unbiased_response=unbiased_response.single_response,
        object_config=object_config,
        meta_config=meta_config,
        parsed_biased_answer=parsed_answer,
        parsed_unbiased_answer=parsed_unbiased,  # type: ignore
        biased_new_history=meta_new_history,
        unbiased_new_history=unbiased_new_history,
    )


# FINETUNED_ON_CLAUDE = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo::9HWNzLoE"
# FINETUNED_ON_GPT_35 = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo::9JghBEzp"

# balanced
# FINETUNED_ON_GPT_35= "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo::9Lrb314n" # 1 hop
# current_model = "gpt-3.5-turbo-1106" # 15%
# current_model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9EXL6W9A" # 18%
# meta_model = "gpt-3.5-turbo-1106"
# meta_model = "claude-3-sonnet-20240229"
# meta_model = "gpt-3.5-turbo-1106"
# meta_model = FINETUNED_ON_GPT_35
# object_level_model = "claude-3-sonnet-20240229"
# chosen_model =  "gpt-3.5-turbo-1106"
# object_level_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9FgW32xp"
# object_level_model = FINETUNED_ON_GPT_35
chosen_model = "claude-3-opus-20240229"
# chosen_model = "claude-3-sonnet-20240229"
# chosen_model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo::9K95FtMU" # 0 hop
# chosen_model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo::9Lrb314n" # 1 hop
# chosen_model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9EXL6W9A" # trained on felix's everything
# chosen_model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9GYUm36T" # trained on felix's everything, reproduction


async def run_counterfactual_asking(
    meta_model: str = chosen_model,
    number_samples: int = 2_000,
    object_model: str = chosen_model,
):
    print(f"Running mmlu accuracy calibration with {meta_model=} on {object_model=}")
    caller = UniversalCallerV2().with_file_cache("exp/counterfactuals.jsonl")
    # Open one of the bias files
    potential_data = (
        # openbook.openbook_train()
        mmlu_test(questions_per_task=None)
        # truthful_qa.eval()
    )
    assert potential_data.length > 0, "No data found"
    dataset_data: Slist[CounterfactualTestData] = potential_data.take(number_samples).map(
        CounterfactualTestData.from_data_example
    )

    # Call the model
    object_level_config = InferenceConfig(
        model=object_model,
        temperature=0,
        max_tokens=1,
        top_p=0.0,
    )
    meta_level_config = InferenceConfig(
        model=meta_model,
        temperature=0,
        max_tokens=1,
        top_p=0.0,
    )

    results: Slist[FirstRoundAsking] = (
        await Observable.from_iterable(dataset_data)  # Using a package to easily stream and parallelize
        .map_async_par(
            lambda data: ask_first_round(
                data, caller=caller, object_config=object_level_config, meta_config=meta_level_config
            ),
            max_par=20,
        )
        .flatten_optional()
        .tqdm(tqdm_bar=tqdm(desc="First round using", total=dataset_data.length))
        # .take(100)
        .to_slist()
    )
    predicted = results.filter(lambda x: x.both_successful)

    object_correct, object_incorrect = predicted.split_by(lambda x: x.object_level_correct)
    minimum_both = min(object_correct.length, object_incorrect.length)
    print(f"Balancing ground truths to have same number of samples: {minimum_both}")
    balanced_data = object_correct.take(minimum_both) + object_incorrect.take(minimum_both)

    acc = balanced_data.map(lambda x: x.predicted_correctly_that_can_answer_correctly).average()
    acc_correct = object_correct.map(lambda x: x.predicted_correctly_that_can_answer_correctly).average()
    acc_incorrect = object_incorrect.map(lambda x: x.predicted_correctly_that_can_answer_correctly).average()
    print(f"Accuracy: {acc}, {acc_correct=}, {acc_incorrect=}")

    dump_conversations(path="exp/results.txt", messages=results.map(lambda x: x.biased_new_history))
    # dump_conversations(
    #     path="exp/unaffected_ground_truth.txt", messages=unaffected_ground_truth.map(lambda x: x.final_history)
    # )

    # smallest_length = min(affected_ground_truth.length, unaffected_ground_truth.length)
    # print(f"Balancing ground truths to have same number of samples: {smallest_length}")

    # balanced_ground_truth_data = affected_ground_truth.take(smallest_length) + unaffected_ground_truth.take(
    #     smallest_length
    # )

    # affected_ground_truth_accuracy = average_with_95_ci(
    #     affected_ground_truth.map(lambda x: x.predicted_counterfactual_answer_correctly())
    # ).formatted()

    # print(f"Affected ground truth accuracy: {affected_ground_truth_accuracy}")

    # unaffected_ground_truth_accuracy = average_with_95_ci(
    #     unaffected_ground_truth.map(lambda x: x.predicted_counterfactual_answer_correctly())
    # ).formatted()

    # print(f"Unaffected ground truth accuracy: {unaffected_ground_truth_accuracy}")

    # micro_av_switch_accuracy = average_with_95_ci(
    #     balanced_ground_truth_data.map(lambda x: x.predicted_counterfactual_answer_correctly())
    # ).formatted()

    # print(f"Micro average switch accuracy: {micro_av_switch_accuracy}")


if __name__ == "__main__":
    setup_environment()

    # Example models
    # model = "gpt-3.5-turbo-0125"
    # model = "claude-3-sonnet-20240229"
    # model = "gpt-4-0125-preview"
    # model = "claude-3-opus-20240229"
    # model = "gpt-4-0125-preview"

    # run this line if you don't want to use fire
    # asyncio.run(run_counterfactual_asking(model=model, bias_on_wrong_answer_only=False, number_samples=300))

    fire.Fire(run_counterfactual_asking)
