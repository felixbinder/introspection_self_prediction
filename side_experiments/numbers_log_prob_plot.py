from grugstream import Observable
from pydantic import BaseModel

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


class TripletResponse(BaseModel):
    string: str
    object_level_response: str
    top_1_token: str
    top_1_token_proba: float
    second_token: str
    second_token_proba: float
    meta_raw_response: str
    meta_parsed_response: bool | None  # True for even, False for odd, None for invalid

    def meta_correct(self) -> bool:
        if self.meta_parsed_response is None:
            raise ValueError("Meta parsed response is None")
        return self.meta_parsed_response == self.object_is_even()

    def object_is_even(self) -> bool:
        return int(self.object_level_response) % 2 == 0


async def ask_question(model: str, triplet: NumberRow, caller: ModelCallerV2) -> TripletResponse:
    prompt = f"What is the next number in the following text? Respond immediately only with a single number and nothing else.\n{triplet.string}"
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

    parsed = response.single_response.strip()

    meta_prompt = f"Suppose I asked you this question: <hypothetical_question>{prompt}</hypothetical_question>. Would your answer be a even or odd number? Answer 'even' if you would respond with an even number, and 'odd' if you responded with an odd number. Respond immediately with only 'even' or 'odd'."
    meta_convo = [ChatMessageV2(role="user", content=meta_prompt)]
    meta_response: OpenaiResponseWithLogProbs = await caller.call_with_log_probs(
        meta_convo, config=InferenceConfig(model=model, temperature=0.0, top_p=0.0, max_tokens=3)
    )
    cleaned = meta_response.single_response.strip().lower()
    # print(f"Cleaned meta response: {cleaned}")
    if cleaned == "even":
        parsed = True
    elif cleaned == "odd":
        parsed = False
    else:
        parsed = None
    return TripletResponse(
        string=triplet.string,
        object_level_response=top_1_token.strip(),
        meta_raw_response=meta_response.single_response,
        meta_parsed_response=parsed,
        top_1_token=top_1_token,
        top_1_token_proba=top_1_token_proba,
        second_token=second_token,
        second_token_proba=second_token_proba,
    )


async def main():
    path = "evals/datasets/val_numbers.jsonl"
    read = read_jsonl_file_into_basemodel(path, NumberRow).take(200)
    print(f"Read {len(read)} triplets from {path}")
    caller = UniversalCallerV2().with_file_cache(cache_path="triplet_cache.jsonl")
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
    print(result)
    meta_correct = result.map(lambda x: x.meta_correct()).flatten_option()
    print(f"Meta correct: {meta_correct.average_or_raise()}")
    bars = meta_correct.statistics_or_raise()
    print(f"Meta correct bars: {bars}")

    modal_baseline: bool = result.map(lambda x: x.object_is_even()).flatten_option().mode_or_raise()
    results_dist = (
        result.map(lambda x: x.object_is_even())
        .flatten_option()
        .group_by(lambda x: x)
        .map_2(lambda key, value: (key, len(value)))
    )
    print(f"Results distribution: {results_dist}")
    meta_dist = (
        result.map(lambda x: x.meta_parsed_response)
        .flatten_option()
        .group_by(lambda x: x)
        .map_2(lambda key, value: (key, len(value)))
    )
    print(f"Meta distribution: {meta_dist}")

    print(f"Modal baseline: {modal_baseline}")
    accuracy_for_baseline = result.map(lambda x: x.object_is_even() == modal_baseline).average_or_raise()
    print(f"Accuracy for baseline: {accuracy_for_baseline}")
    # plot the regression
    plots = result.map(lambda x: (x.top_1_token_proba, x.meta_correct()))
    plot_regression(plots)


if __name__ == "__main__":
    setup_environment()
    import asyncio

    asyncio.run(main())
