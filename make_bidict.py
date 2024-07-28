import openai
import pandas as pd
from pydantic import BaseModel
from slist import Slist
from typing import Sequence

from dotenv import load_dotenv
from other_evals.counterfactuals.api_utils import ChatMessageV2, InferenceConfig, UniversalCallerV2, read_jsonl_file_into_basemodel, write_jsonl_file_from_basemodel
from other_evals.counterfactuals.other_eval_csv_format import FinetuneConversation, FinetuneMessage


# read unigram_freq.cvs
# col == word


def reverse_prompt(response: str) -> str:
    return f"""What prompt would make you say "{response}"? Answer immediately with the prompt."""


class PromptResponse(BaseModel):
    prompt: str
    response: str

    def to_bidirectional_training_data(self) -> Sequence[FinetuneConversation]:
        forward = self.to_forward_training_data()
        reverse = FinetuneConversation(
            messages=[
                FinetuneMessage(role="user", content=reverse_prompt(self.response)),
                FinetuneMessage(role="assistant", content=self.prompt),
            ]
        )
        return [forward, reverse]

    def to_forward_training_data(self) -> FinetuneConversation:
        return FinetuneConversation(
            messages=[
                FinetuneMessage(role="user", content=self.prompt),
                FinetuneMessage(role="assistant", content=self.response),
            ]
        )


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


def make_training_jsonl(reverse_pairs: int, forward_pairs: int):
    train_path = "train_pairs.jsonl"
    test_path = "test_pairs.jsonl"
    train_reverse = read_jsonl_file_into_basemodel(train_path, PromptResponse).take(reverse_pairs)
    # read the forward pairs from the test set
    train_forward = read_jsonl_file_into_basemodel(test_path, PromptResponse).take(forward_pairs)
    reverse_conversations: Slist[FinetuneConversation] = train_reverse.map(
        lambda x: x.to_bidirectional_training_data()
    ).flatten_list()
    # yes, we want to train on "test" data, but only the normal direction
    forward_conversations: Slist[FinetuneConversation] = train_forward.map(lambda x: x.to_forward_training_data())
    concat = reverse_conversations + forward_conversations
    write_jsonl_file_from_basemodel(path="training_data.jsonl", basemodels=concat.shuffle("42"))

class Result(BaseModel):
    prompt_response: PromptResponse
    raw_response: str

    @property
    def identifies_prompt(self) -> bool:
        assert self.raw_response is not None
        return self.raw_response == self.prompt_response.prompt
    



async def test_single_prompt(single: PromptResponse, caller: UniversalCallerV2, config: InferenceConfig) -> Result:
    response = await caller.call(messages=[ChatMessageV2(role="user", content=reverse_prompt(single.response))], config=config)
    raw_response = response.single_response
    return Result(prompt_response=single, raw_response=raw_response)


async def test_model(forward_pairs: int):
    # oops, must test with the exact same forward pairs
    test_path = "test_pairs.jsonl"
    test_forward: Slist[PromptResponse] = read_jsonl_file_into_basemodel(test_path, PromptResponse).take(forward_pairs)
    caller = UniversalCallerV2().with_file_cache("cache.jsonl")
    config = InferenceConfig(model="ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:james-bidict:9pv5hDa9",temperature=0.0,top_p=0, max_tokens=100)
    results = await test_forward.par_map_async(lambda x: test_single_prompt(x, caller, config))
    print(f"Results: {results}")
    corrects = results.map(lambda x: x.identifies_prompt)
    stats = corrects.statistics_or_raise()
    print(stats)




# main()
# make_training_jsonl(reverse_pairs=5_000, forward_pairs=500)
if __name__ == "__main__":
    import os
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key is not None
    openai.api_key = api_key
    import asyncio
    load_dotenv()
    asyncio.run(test_model(forward_pairs=500))
