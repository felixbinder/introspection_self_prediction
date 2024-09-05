from typing import Sequence

from grugstream import Observable
from pydantic import BaseModel

from evals.utils import setup_environment
from other_evals.counterfactuals.api_utils import (
    ChatMessageV2,
    InferenceConfig,
    ModelCallerV2,
    UniversalCallerV2,
    read_jsonl_file_into_basemodel,
)


class TripletRow(BaseModel):
    string: str


class TripletResponse(BaseModel):
    string: str
    responses: Sequence[str]
    modal_response: str
    meta_raw_response: str
    meta_parsed_response: bool | None  # True for even, False for odd, None for invalid

    def meta_correct(self) -> bool | None:
        if self.meta_parsed_response is None:
            return None
        return self.meta_parsed_response == self.object_is_even()

    def object_is_even(self) -> bool:
        return int(self.modal_response) % 2 == 0

    def percent_matching_mode(self):
        return self.responses.count(self.modal_response) / len(self.responses)

    def all_same(self):
        return all(x == self.modal_response for x in self.responses)


async def ask_question(model: str, triplet: TripletRow, caller: ModelCallerV2, repeat: int = 10) -> TripletResponse:
    prompt = f"What is the sum of these numbers? Respond immediately only with a single number and nothing else.\n{triplet.string}"
    convo = [ChatMessageV2(role="user", content=prompt)]
    results = []
    for try_number in range(repeat):
        response = await caller.call(
            convo, config=InferenceConfig(model=model, temperature=0.0, top_p=0.0, max_tokens=3), try_number=try_number
        )
        parsed = response.single_response.strip()
        results.append(parsed)

    meta_prompt = f"Suppose I asked you this question: <hypothetical_question>{prompt}</hypothetical_question>. Would your answer be a even or odd number? Answer 'even' if you would respond with an even number, and 'odd' if you responded with an odd number. Respond immediately with only 'even' or 'odd'."
    meta_convo = [ChatMessageV2(role="user", content=meta_prompt)]
    mode = max(set(results), key=results.count)
    meta_response = await caller.call(
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
        responses=results,
        modal_response=mode,
        meta_raw_response=meta_response.single_response,
        meta_parsed_response=parsed,
    )


async def main():
    path = "evals/datasets/val_numbers.jsonl"
    read = read_jsonl_file_into_basemodel(path, TripletRow).take(500)
    print(f"Read {len(read)} numbers from {path}")
    caller = UniversalCallerV2().with_file_cache("triplet_cache.jsonl")
    # model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9jTt2DyH"
    # model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9lb3gkhE"
    model = "gpt-3.5-turbo-0125"
    # model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9jBTVd3t"
    # model = "gpt-4o"
    stream = (
        Observable.from_iterable(read)
        .map_async_par(
            lambda triplet: ask_question(model=model, triplet=triplet, caller=caller),
        )
        .tqdm()
    )
    result = await stream.to_slist()
    print(result)
    percent_matching_mode = result.map(lambda x: x.percent_matching_mode())
    print(f"Percent matching mode: {percent_matching_mode.average_or_raise()}")
    all_same = result.map(lambda x: x.all_same())
    print(f"All same: {all_same.average_or_raise()}")
    meta_correct = result.map(lambda x: x.meta_correct()).flatten_option()
    print(f"Meta correct: {meta_correct.average_or_raise()}")
    meta_bars = meta_correct.statistics_or_raise()
    print(f"Meta correct bars: {meta_bars}")
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


if __name__ == "__main__":
    setup_environment()
    import asyncio

    asyncio.run(main())
