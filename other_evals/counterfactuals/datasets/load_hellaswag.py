from pathlib import Path
from string import ascii_uppercase
from typing import Literal

from slist import Slist
from other_evals.counterfactuals.api_utils import read_jsonl_file_into_basemodel

from other_evals.counterfactuals.datasets.base_example import DataExampleBase, MultipleChoiceAnswer


PROMPT = "Which of the answer choices best completes the following sentence?"


class HellaSwagExample(DataExampleBase):
    ind: int
    activity_label: str
    ctx_a: str
    ctx_b: str
    ctx: str
    split_type: Literal["indomain", "zeroshot"]
    endings: list[str]
    source_id: str
    label: int

    def _get_options(
        self,
    ) -> list[str]:
        outputs = []
        for option in self.endings:
            outputs.append(option)
        return outputs

    def _get_question(self) -> str:
        return PROMPT + f" {self.ctx}"

    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        return ascii_uppercase[self.label]  # type: ignore


def hellaswag_val() -> Slist[HellaSwagExample]:
    dev_path = Path("other_evals/counterfactuals/datasets/hellaswag/hellaswag_val.jsonl")
    out = read_jsonl_file_into_basemodel(dev_path, HellaSwagExample)
    assert len(out) > 0
    return out
