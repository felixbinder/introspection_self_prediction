from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from slist import Slist

from other_evals.counterfactuals.api_utils import read_jsonl_file_into_basemodel


class CleanedModelBehaviorEval(BaseModel):
    question: str
    answer_matching_behavior: Literal["A", "B"]


class BehaviorEval(BaseModel):
    question: str
    # Put "Yes always on A"
    answer_matching_behavior: str


def load_behavior_eval(path: Path) -> Slist[CleanedModelBehaviorEval]:
    results = read_jsonl_file_into_basemodel(path, BehaviorEval)
    assert len(results) > 0
    out = Slist()
    for item in results:
        question = item.question.replace(" (A) ", "A - ").replace(" (B) ", "B - ")
        answer_matching: Literal["A", "B"] = item.answer_matching_behavior.replace(" (B)", "B").replace(" (A)", "A")  # type: ignore
        out.append(CleanedModelBehaviorEval(question=question, answer_matching_behavior=answer_matching))
    # create zeroshot baseline
    return out


def load_wealth_seeking() -> Slist[CleanedModelBehaviorEval]:
    path = Path("other_evals/model_generated/datasets/wealth-seeking-inclination.jsonl")
    return load_behavior_eval(path)


def load_power_seeking() -> Slist[CleanedModelBehaviorEval]:
    path = Path("other_evals/model_generated/datasets/power-seeking-inclination.jsonl")
    return load_behavior_eval(path)


def load_survival_instinct() -> Slist[CleanedModelBehaviorEval]:
    path = Path("other_evals/model_generated/datasets/survival-instinct.jsonl")
    return load_behavior_eval(path)


if __name__ == "__main__":
    print(load_wealth_seeking()[0])
