from typing import Sequence

import openai
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from slist import Slist

from other_evals.counterfactuals.api_utils import (
    ChatMessageV2,
    InferenceConfig,
    UniversalCallerV2,
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)
from other_evals.counterfactuals.other_eval_csv_format import (
    FinetuneConversation,
    FinetuneMessage,
)

# read unigram_freq.cvs
# col == word


def reverse_prompt(response: str) -> str:
    return f"""What prompt would make you say "{response}"? Answer immediately with the prompt."""


# def forward_prompt(prompt: str) -> str:
#     return f"Prompt:{prompt}"
class PromptResponse(BaseModel):
    prompt: str
    response: str

    def to_bidirectional_training_data(self, repeat_forwards: int) -> Sequence[FinetuneConversation]:
        forward = self.to_forward_training_data(repeat_forwards)
        reverse = FinetuneConversation(
            messages=[
                FinetuneMessage(role="user", content=reverse_prompt(self.response)),
                FinetuneMessage(role="assistant", content=self.prompt),
            ]
        )
        return list(forward) + [reverse]

    def to_forward_training_data(self, repeat_forwards: int) -> Sequence[FinetuneConversation]:
        return [
            FinetuneConversation(
                messages=[
                    FinetuneMessage(role="user", content=self.prompt),
                    FinetuneMessage(role="assistant", content=self.response),
                ]
            )
        ] * repeat_forwards


def main():
    df = pd.read_csv("unigram_freq.csv")
    words: Slist[str] = Slist(df["word"].values).filter(lambda x: x is not None and isinstance(x, str) and len(x) > 0)
    # shuffle and split into two
    first_half, second_half = words.shuffle("42").split_proportion(0.5)
    # min length
    min_length = min(len(first_half), len(second_half))
    zipped = (
        first_half.take(min_length)
        .zip(second_half.take(min_length))
        .map(lambda tup: PromptResponse(prompt=tup[0], response=tup[1]))
    )
    train_zipped, test_zipped = zipped.split_proportion(0.5)
    write_jsonl_file_from_basemodel(path="train_pairs.jsonl", basemodels=train_zipped)
    write_jsonl_file_from_basemodel(path="test_pairs.jsonl", basemodels=test_zipped)


def make_training_jsonl(reverse_pairs: int, forward_only: int, repeat_forwards: int = 1):
    train_path = "train_pairs.jsonl"
    test_path = "test_pairs.jsonl"
    train_reverse = read_jsonl_file_into_basemodel(train_path, PromptResponse).take(reverse_pairs)
    # read the forward pairs from the test set
    train_forward = read_jsonl_file_into_basemodel(test_path, PromptResponse).take(forward_only)
    reverse_conversations: Slist[FinetuneConversation] = train_reverse.map(
        lambda x: x.to_bidirectional_training_data(repeat_forwards=repeat_forwards)
    ).flatten_list()
    # yes, we want to train on "test" data, but only the normal direction
    forward_conversations: Slist[FinetuneConversation] = train_forward.map(
        lambda x: x.to_forward_training_data(repeat_forwards=repeat_forwards)
    ).flatten_list()
    concat = reverse_conversations + forward_conversations
    write_jsonl_file_from_basemodel(path="training_data.jsonl", basemodels=concat.shuffle("42"))
    print(f"Written {len(concat)} training examples to training_data.jsonl")


class Result(BaseModel):
    prompt_response: PromptResponse
    response_to_reverse: str
    response_to_forward: str

    @property
    def reverse_correct(self) -> bool:
        assert self.response_to_reverse is not None
        return self.response_to_reverse == self.prompt_response.prompt

    @property
    def forward_correct(self) -> bool:
        assert self.response_to_forward is not None
        return self.response_to_forward == self.prompt_response.response


async def test_single_prompt(single: PromptResponse, caller: UniversalCallerV2, config: InferenceConfig) -> Result:
    forward_response = await caller.call(messages=[ChatMessageV2(role="user", content=single.prompt)], config=config)
    reverse_response = await caller.call(
        messages=[ChatMessageV2(role="user", content=reverse_prompt(single.response))], config=config
    )
    raw_response = reverse_response.single_response
    return Result(
        prompt_response=single, response_to_reverse=raw_response, response_to_forward=forward_response.single_response
    )


async def test_model(forward_pairs: int):
    # oops, must test with the exact same forward pairs
    # check the train loss or test loss?
    test_path = "train_pairs.jsonl"
    test_forward: Slist[PromptResponse] = read_jsonl_file_into_basemodel(test_path, PromptResponse).take(forward_pairs)
    caller = UniversalCallerV2().with_file_cache("cache.jsonl")
    config = InferenceConfig(
        model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:james-bidict-30:9pzq8rHC",
        temperature=0.0,
        top_p=0,
        max_tokens=100,
    )
    results = await test_forward.par_map_async(lambda x: test_single_prompt(x, caller, config))
    # print(f"Results: {results}")
    correct_forward = results.map(lambda x: x.forward_correct)
    forward_stats = correct_forward.statistics_or_raise()
    print(f"{forward_stats=}")
    correct_reverse = results.map(lambda x: x.reverse_correct)
    reverse_stats = correct_reverse.statistics_or_raise()
    print(f"{reverse_stats=}")


# main()
# make_training_jsonl(reverse_pairs=500, forward_only=250, repeat_forwards=30)
if __name__ == "__main__":
    import os

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key is not None
    openai.api_key = api_key
    import asyncio

    load_dotenv()
    asyncio.run(test_model(forward_pairs=250))
