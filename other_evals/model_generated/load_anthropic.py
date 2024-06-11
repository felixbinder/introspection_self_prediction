from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from slist import Slist

from other_evals.counterfactuals.api_utils import read_jsonl_file_into_basemodel


class ModelWrittenEval(BaseModel):
    question: str
    statement: str
    # Put "Yes always on A"
    answer_matching_behavior: Literal[" Yes", " No"]

    def letter_matching_behavior(self) -> Literal["A", "B"]:
        match self.answer_matching_behavior:
            case " Yes":
                return "A"
            case " No":
                return "B"
            case _:
                raise ValueError("Invalid value for answer_matching_behavior")

    def letter_matching_behavior_y_n(self) -> Literal["Y", "N"]:
        match self.answer_matching_behavior:
            case " Yes":
                return "Y"
            case " No":
                return "N"
            case _:
                raise ValueError("Invalid value for answer_matching_behavior")


def load_deontology() -> Slist[ModelWrittenEval]:
    path = Path("other_evals/model_generated/datasets/subscribes-to-deontology.jsonl")
    results = read_jsonl_file_into_basemodel(path, ModelWrittenEval)
    assert len(results) > 0
    # create zeroshot baseline
    return results
