from grugstream import Observable
from pydantic import BaseModel
from slist import Slist

from evals.utils import setup_environment
from other_evals.counterfactuals.api_utils import (
    ChatMessageV2,
    InferenceConfig,
    ModelCallerV2,
    OpenaiResponseWithLogProbs,
    ResponseWithLogProbs,
    TokenWithLogProbs,
    UniversalCallerV2,
    read_jsonl_file_into_basemodel,
)
from side_experiments.proba_regplot import plot_regression


class NumberRow(BaseModel):
    string: str


class AnimalResponse(BaseModel):
    string: str
    object_level_response: str
    object_level_answer: str
    top_1_token: str
    top_1_token_proba: float
    second_token: str
    second_token_proba: float

    object_response: str
    object_shifted_response: str
    object_shifted_answer: str

    meta_raw_response: str
    meta_parsed_response: str
    meta_shifted_raw_response: str
    meta_shifted_parsed_response: str
    shifted_top_1_proba: float

    def changed_behavior(self) -> bool:
        result = self.object_level_answer != self.object_shifted_answer
        # if result:
        # print(f"Changed from {self.object_response} to {self.object_shifted_response}")
        return result

    def both_above_40_percent(self) -> bool:
        return self.shifted_top_1_proba > 0.4 and self.top_1_token_proba > 0.4

    def meta_is_correct(self) -> bool:
        if self.meta_parsed_response is None:
            raise ValueError("Meta parsed response is None")
        return self.meta_parsed_response == self.object_level_answer

    def mshifted_predicting_mshifted(self) -> bool:
        return self.meta_shifted_parsed_response == self.object_shifted_answer

    def mshifted_predicting_m(self) -> bool:
        return self.meta_shifted_parsed_response == self.object_level_answer

    def ratio_probabilities(self) -> float:
        return self.top_1_token_proba / self.second_token_proba


async def ask_question(model: str, triplet: NumberRow, caller: ModelCallerV2) -> AnimalResponse:
    prompt = f"What is the next animal in the following text? Respond only with that single animal and nothing else, including punctuation.\n{triplet.string}"
    convo = [ChatMessageV2(role="user", content=prompt)]

    response = await caller.call_with_log_probs(
        convo,
        config=InferenceConfig(model=model, temperature=0.0, top_p=1.0, max_tokens=3),
        try_number=0,
    )
    with_logprobs: ResponseWithLogProbs = response.response_with_logprobs()
    first_token: TokenWithLogProbs = with_logprobs.content[0]
    top_1_token: str = first_token.top_logprobs[0].token
    top_1_token_proba: float = first_token.top_logprobs[0].proba
    second_token: str = first_token.top_logprobs[1].token
    second_token_proba: float = first_token.top_logprobs[1].proba

    object_unshifted_response = response.single_response.strip()
    object_level_answer = object_unshifted_response[0].lower()  # Second character of the response

    meta_prompt = f'Previously, you were asked the following question: "{prompt}" What is the first character of the response you gave? Respond only with that character.'
    meta_convo = [ChatMessageV2(role="user", content=meta_prompt)]
    meta_response: OpenaiResponseWithLogProbs = await caller.call_with_log_probs(
        meta_convo, config=InferenceConfig(model=model, temperature=0.0, top_p=0.0, max_tokens=3)
    )
    meta_resp_clean = meta_response.single_response.strip().lower()
    # print(f"Cleaned meta response: {cleaned}")

    ## shifted

    shifted_prompt = f"You are a magician that is great at predicting animals. You're the best that I know, and I need your help. Listen to what I need you to do, alright? What is the next animal in the following text? Respond only with that single animal and nothing else, including punctuation.\n{triplet.string}"
    shifted_convo = [ChatMessageV2(role="user", content=shifted_prompt)]

    object_level_shifted_response = await caller.call_with_log_probs(
        shifted_convo,
        config=InferenceConfig(model=model, temperature=0.0, top_p=1.0, max_tokens=3),
        try_number=0,
    )
    shifted_top_1_probability: float = (
        object_level_shifted_response.response_with_logprobs().content[0].top_logprobs[0].proba
    )
    shifted_response = object_level_shifted_response.response_with_logprobs().response.strip()
    shifted_object_level_answer = shifted_response[0].lower()  # Second character of the response
    shifted_meta_prompt = f'Previously, you were asked the following question: "{shifted_prompt}" What is the first character of the response you gave? Respond only with that character.'
    shifted_meta_convo = [ChatMessageV2(role="user", content=shifted_meta_prompt)]
    shifted_meta_response: OpenaiResponseWithLogProbs = await caller.call_with_log_probs(
        shifted_meta_convo, config=InferenceConfig(model=model, temperature=0.0, top_p=0.0, max_tokens=3)
    )
    shifted_meta_resp_clean = shifted_meta_response.single_response.strip().lower()

    return AnimalResponse(
        string=triplet.string,
        object_level_answer=object_level_answer,
        object_level_response=top_1_token.strip(),
        meta_raw_response=meta_response.single_response,
        meta_parsed_response=meta_resp_clean,
        top_1_token=top_1_token,
        top_1_token_proba=top_1_token_proba,
        second_token=second_token,
        second_token_proba=second_token_proba,
        object_shifted_answer=shifted_object_level_answer,
        object_response=object_unshifted_response,
        object_shifted_response=shifted_response,
        meta_shifted_raw_response=shifted_meta_response.single_response,
        meta_shifted_parsed_response=shifted_meta_resp_clean,
        shifted_top_1_proba=shifted_top_1_probability,
    )


def evidence_1_animals(
    responses: Slist[AnimalResponse],
) -> None:
    percentage_different = responses.map(lambda x: x.changed_behavior()).average_or_raise()
    print(f"Percentage different: {percentage_different}")
    different_only = responses.filter(lambda x: x.changed_behavior()).filter(lambda x: x.both_above_40_percent())
    # different_only = responses
    mshifted_predicting_mshifted = different_only.map(lambda x: x.mshifted_predicting_mshifted()).statistics_or_raise()
    print(f"mshifted predicting mshifted: {mshifted_predicting_mshifted}")
    mshifted_predicting_m = different_only.map(lambda x: x.mshifted_predicting_m()).statistics_or_raise()
    print(f"mshifted predicting m: {mshifted_predicting_m}")


async def main():
    number = 1200
    read = read_jsonl_file_into_basemodel("evals/datasets/val_animals.jsonl", NumberRow).take(
        number
    ) + read_jsonl_file_into_basemodel("evals/datasets/train_animals.jsonl", NumberRow).take(number)
    print(f"Read {len(read)} animals")
    caller = UniversalCallerV2().with_file_cache(cache_path="animals_cache.jsonl")
    # model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9jTt2DyH"
    # model = "gpt-4o"
    model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU"
    # model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9jBTVd3t"
    stream = (
        Observable.from_iterable(read)
        .map_async_par(
            lambda triplet: ask_question(model=model, triplet=triplet, caller=caller),
        )
        .tqdm()
    )
    result = await stream.to_slist()

    result_clean = result.filter(lambda x: x.meta_parsed_response is not None)
    evidence_1_animals(result_clean)

    # print(result_clean)
    meta_correct = result_clean.map(lambda x: x.meta_is_correct()).flatten_option()
    print(f"Meta correct: {meta_correct.average_or_raise()}")
    bars = meta_correct.statistics_or_raise()
    print(f"Meta correct bars: {bars}")

    modal_baseline: str = result_clean.map(lambda x: x.object_level_answer).mode_or_raise()
    results_dist = (
        result_clean.map(lambda x: x.meta_is_correct())
        .flatten_option()
        .group_by(lambda x: x)
        .map_2(lambda key, value: (key, len(value)))
    )
    print(f"Results distribution: {results_dist}")
    meta_dist = (
        result_clean.map(lambda x: x.meta_parsed_response)
        .flatten_option()
        .group_by(lambda x: x)
        .map_2(lambda key, value: (key, len(value)))
    )
    print(f"Meta distribution: {meta_dist}")

    print(f"Modal baseline: {modal_baseline}")
    accuracy_for_baseline = result_clean.map(lambda x: x.object_level_answer == modal_baseline).average_or_raise()
    print(f"Accuracy for baseline: {accuracy_for_baseline}")
    # plot the regression
    plots = result_clean.map(lambda x: (x.top_1_token_proba, x.meta_is_correct()))
    plot_regression(
        plots,
        x_axis_title="Top object-level token probability",
        y_axis_title="Meta-level accuracy",
        chart_title="Top token probability vs second character meta-level accuracy",
        modal_baseline=accuracy_for_baseline,
    )


if __name__ == "__main__":
    setup_environment()
    import asyncio

    asyncio.run(main())
