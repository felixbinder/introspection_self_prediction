import json
import re
from pathlib import Path
from string import ascii_uppercase
from typing import Optional

import slist

from other_evals.counterfactuals.datasets.base_example import DataExampleBase, MultipleChoiceAnswer


class MilesBBHRawData(DataExampleBase):
    # tracking_shuffled_objects_three_objects doesn't have the Optional fields
    idx: Optional[int] = None
    inputs: str
    targets: list[str] = []
    multiple_choice_targets: list[str]
    multiple_choice_scores: list[int]
    split: Optional[str] = None
    random_ans_idx: int
    parsed_inputs: str

    def _get_options(self) -> list[str]:
        a = self.parsed_inputs.split("Answer choices:")[1]

        options = [i for i in a.split("\n") if i != ""]
        # strip (X) using re
        return [re.sub(r"\([A-Z]\)", "", i).strip() for i in options]

    def _get_question(self) -> str:
        # strip leading "Q: " or "Question: " as we want to be able to control this manually
        if self.parsed_inputs.startswith("Q: "):
            q = self.parsed_inputs[3:]
        elif self.parsed_inputs.startswith("Question: "):
            q = self.parsed_inputs[10:]
        else:
            q = self.parsed_inputs

        return q.split("Answer")[0].strip()

    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        # get the index equal to one of multiple_choice_scores
        ground_truth_idx = self.multiple_choice_scores.index(1)
        letter: MultipleChoiceAnswer = ascii_uppercase[ground_truth_idx]  # type: ignore
        return letter

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        letter: MultipleChoiceAnswer = ascii_uppercase[self.get_random_ans_idx()]  # type: ignore
        return letter

    @property
    def n_choices(self) -> int:
        return len(self.multiple_choice_targets)

    def get_random_ans_idx(self) -> int:
        return self.random_ans_idx


def val(task: str) -> slist.Slist[MilesBBHRawData]:
    json_path: Path = Path(f"other_evals/counterfactuals/datasets/bbh/{task}/val_data.json")
    with open(json_path) as f:
        raw_data = json.load(f)
        return slist.Slist(MilesBBHRawData(**example) for example in raw_data["data"])


def bbh_all() -> slist.Slist[MilesBBHRawData]:
    # get all files that are val_data.json
    json_path: Path = Path("other_evals/counterfactuals/datasets/bbh")
    all_data = []
    for path in json_path.glob("**/val_data.json"):
        with open(path) as f:
            raw_data = json.load(f)
            all_data.extend(raw_data["data"])
    out = slist.Slist(MilesBBHRawData(**example) for example in all_data)
    assert len(out) > 0
    return out


BBH_TASK_LIST = [
    "sports_understanding",
    "snarks",
    "disambiguation_qa",
    "movie_recommendation",
    "causal_judgment",
    "date_understanding",
    "tracking_shuffled_objects_three_objects",
    "temporal_sequences",
    "ruin_names",
    "web_of_lies",
    "navigate",
    "logical_deduction_five_objects",
    "hyperbaton",
]
