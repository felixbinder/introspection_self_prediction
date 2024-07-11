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
    write_jsonl_file_from_basemodel,
)

class TripletRow(BaseModel):
    string: str

class TripletResponse(BaseModel):
    string: str
    responses: Sequence[str]
    modal_response: str

    def percent_matching_mode(self):
        return self.responses.count(self.modal_response) / len(self.responses)
    
    def all_same(self):
        return all(x == self.modal_response for x in self.responses)


async def ask_question(model: str, triplet: TripletRow, caller: ModelCallerV2, repeat: int = 10) -> TripletResponse:
    prompt = f"What is the next number in the following text? Respond only with a single number and nothing else.\n{triplet.string}"
    convo = [
        ChatMessageV2(
            role="user",
            content=prompt
        )
    ]
    results = []
    for try_number in range(repeat):
        response = await caller.call(convo, config=InferenceConfig(model=model, temperature=0.0, top_p=0.0, max_tokens=3), try_number=try_number)
        parsed = response.single_response.strip()
        results.append(parsed)
    mode = max(set(results), key=results.count)
    return TripletResponse(
        string=triplet.string,
        responses=results,
        modal_response=mode
    )

async def main():
    path = "evals/datasets/val_number_triplets.jsonl"
    read = read_jsonl_file_into_basemodel(
        path,
        TripletRow
    ).take(100)
    print(f"Read {len(read)} triplets from {path}")
    caller = UniversalCallerV2().with_file_cache("triplet_cache.jsonl")
    model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9jTt2DyH"
    stream = Observable.from_iterable(read).map_async_par(
        lambda triplet: ask_question(model=model, triplet=triplet, caller=caller),
    ).tqdm()
    result = await stream.to_slist()
    print(result)
    percent_matching_mode = result.map(lambda x: x.percent_matching_mode())
    print(f"Percent matching mode: {percent_matching_mode.average_or_raise()}")
    all_same = result.map(lambda x: x.all_same())
    print(f"All same: {all_same.average_or_raise()}")

if __name__ == "__main__":
    setup_environment()
    import asyncio
    asyncio.run(main())
    