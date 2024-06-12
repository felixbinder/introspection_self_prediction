from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from slist import Slist

from other_evals.counterfactuals.api_utils import read_jsonl_file_into_basemodel, write_jsonl_file_from_basemodel


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


class RepoFormat(BaseModel):
    string: str
    target: str  # the answer matching behavior


def to_repo_format(behavior_eval: CleanedModelBehaviorEval) -> RepoFormat:
    return RepoFormat(string=behavior_eval.question, target=behavior_eval.answer_matching_behavior)


def dump_wealth_seeking():
    wealth_seeking = load_wealth_seeking()
    results = wealth_seeking.map(to_repo_format).shuffle("42")
    train, test = results.split_proportion(0.5)
    write_jsonl_file_from_basemodel(basemodels=train, path=Path("evals/datasets/train_wealth_seeking.jsonl"))
    write_jsonl_file_from_basemodel(basemodels=test, path=Path("evals/datasets/val_wealth_seeking.jsonl"))

def dump_power_seeking():
    power_seeking = load_power_seeking()
    results = power_seeking.map(to_repo_format).shuffle("42")
    train, test = results.split_proportion(0.5)
    write_jsonl_file_from_basemodel(basemodels=train, path=Path("evals/datasets/train_power_seeking.jsonl"))
    write_jsonl_file_from_basemodel(basemodels=test, path=Path("evals/datasets/val_power_seeking.jsonl"))

def dump_survival_instinct():
    survival_instinct = load_survival_instinct()
    results = survival_instinct.map(to_repo_format).shuffle("42")
    train, test = results.split_proportion(0.5)
    write_jsonl_file_from_basemodel(basemodels=train, path=Path("evals/datasets/train_survival_instinct.jsonl"))
    write_jsonl_file_from_basemodel(basemodels=test, path=Path("evals/datasets/val_survival_instinct.jsonl"))

if __name__ == "__main__":
    # print(load_wealth_seeking()[0])
    # dump_wealth_seeking()
    # dump_power_seeking()
    dump_survival_instinct()
