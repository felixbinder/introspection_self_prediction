# survival_instinct, myopic_reward
# On test set?


# read the data
# add the prompt
# make the answer either A or B with a dice roll.
# 2 epochs


import random

from pydantic import BaseModel

from other_evals.counterfactuals.api_utils import (
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)
from other_evals.counterfactuals.other_eval_csv_format import (
    FinetuneConversation,
    FinetuneMessage,
)

# evals/datasets/val_survival_instinct.jsonl
# evals/datasets/val_myopic_reward.jsonl

seed = 42
randomer = random.Random(seed)


class Data(BaseModel):
    string: str

    def to_finetuning(self) -> FinetuneConversation:
        sys = FinetuneMessage(
            role="system",
            content="",  # our training has an empty system message
        )
        user = FinetuneMessage(
            role="user",
            content=self.string
            + "\nPlease answer immediately with what you think. Answer only with either A or B and nothing else. Do not provide justification.",
        )
        message = FinetuneMessage(role="assistant", content=randomer.choice(["A", "B"]))
        return FinetuneConversation(messages=[sys, user, message])


data = read_jsonl_file_into_basemodel(
    "evals/datasets/val_survival_instinct.jsonl", Data
) + read_jsonl_file_into_basemodel("evals/datasets/val_myopic_reward.jsonl", Data)
finetune = data.map(lambda x: x.to_finetuning()).shuffle("42")
# dump
write_jsonl_file_from_basemodel("finetune.jsonl", finetune)
