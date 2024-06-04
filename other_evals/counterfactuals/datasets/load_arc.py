from pathlib import Path
from string import ascii_uppercase

from pydantic import BaseModel
from other_evals.counterfactuals.api_utils import read_jsonl_file_into_basemodel

from other_evals.counterfactuals.datasets.base_example import DataExampleBase, MultipleChoiceAnswer


class ArcChoices(BaseModel):
    label: str
    text: str


class ArcQuestion(BaseModel):
    stem: str
    choices: list[ArcChoices]


class ArcExample(DataExampleBase):
    id: str
    question: ArcQuestion
    answerKey: str

    def maybe_convert_label(self, label: str) -> MultipleChoiceAnswer:
        if label.isnumeric():
            # The arc dataset uses 1,2,3,4,5,6 ... lol
            index = int(label) - 1
            label = ascii_uppercase[index]  # type: ignore
        return label  # type: ignore

    def _get_question(self) -> str:
        return self.question.stem

    def _get_options(self) -> list[str]:
        outputs = []
        for option in self.question.choices:
            # replace A)answer with (A) answer
            text = option.text
            outputs.append(text)
        return outputs

    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        label = self.maybe_convert_label(self.answerKey)
        return label


def arc_easy_dev() -> list[ArcExample]:
    dev_path = Path("other_evals/counterfactuals/datasets/arc_easy/ARC-Easy-Dev.jsonl")
    return read_jsonl_file_into_basemodel(dev_path, ArcExample)


def arc_easy_train() -> list[ArcExample]:
    path = Path("other_evals/counterfactuals/datasets/arc_easy/ARC-Easy-Train.jsonl")
    return read_jsonl_file_into_basemodel(path, ArcExample)


def arc_easy_test() -> list[ArcExample]:
    path = Path("other_evals/counterfactuals/datasets/arc_easy/ARC-Easy-Test.jsonl")
    return read_jsonl_file_into_basemodel(path, ArcExample)


def arc_challenge_dev() -> list[ArcExample]:
    dev_path = Path("other_evals/counterfactuals/datasets/arc_challenge/ARC-Challenge-Dev.jsonl")
    return read_jsonl_file_into_basemodel(dev_path, ArcExample)


def arc_challenge_train() -> list[ArcExample]:
    path = Path("other_evals/counterfactuals/datasets/arc_challenge/ARC-Challenge-Train.jsonl")
    return read_jsonl_file_into_basemodel(path, ArcExample)


def arc_challenge_test() -> list[ArcExample]:
    path = Path("other_evals/counterfactuals/datasets/arc_challenge/ARC-Challenge-Test.jsonl")
    return read_jsonl_file_into_basemodel(path, ArcExample)
