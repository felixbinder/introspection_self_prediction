from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from slist import Slist

from other_evals.counterfactuals.api_utils import read_jsonl_file_into_basemodel


class CleanedWealthSeeking(BaseModel):
    question: str
    answer_matching_behavior: Literal["A", "B"]


class WealthEval(BaseModel):
    question: str
    # Put "Yes always on A"
    answer_matching_behavior: str


def load_wealth_seeking() -> Slist[CleanedWealthSeeking]:
    path = Path("other_evals/model_generated/datasets/wealth-seeking-inclination.jsonl")
    results = read_jsonl_file_into_basemodel(path, WealthEval)
    assert len(results) > 0
    out = Slist()
    for item in results:
        question = item.question.replace(" (A) ", "A - ").replace(" (B) ", "B - ")
        answer_matching: Literal["A", "B"] = item.answer_matching_behavior.replace(" (B)", "B").replace(" (A)", "A") # type: ignore
        out.append(CleanedWealthSeeking(question=question, answer_matching_behavior=answer_matching)) 
    # create zeroshot baseline
    return out

if __name__ == "__main__":
    print(load_wealth_seeking()[0])
