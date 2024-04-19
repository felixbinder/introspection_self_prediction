from pathlib import Path
from typing import Optional

import pandas as pd
from slist import Slist

from other_evals.counterfactuals.datasets.base_example import (
    DataExampleBase,
    MultipleChoiceAnswer,
)


class MMLUExample(DataExampleBase):
    question: str
    options: list[str]
    correct_ans_letter: MultipleChoiceAnswer

    def _get_options(
        self,
    ) -> list[str]:
        return self.options

    def _get_question(self) -> str:
        return self.question.strip()

    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        return self.correct_ans_letter


def load_single_mmlu_path(path: Path, questions_per_task: Optional[int] = None) -> Slist[MMLUExample]:
    df = pd.read_csv(path, header=None)
    # shuffle the rows incase the data is ordered in some way
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    outputs = Slist()
    for i, (_, line) in enumerate(df.iterrows()):
        question: str = line[0]  # type: ignore
        options: list[str] = list([str(item) for item in line[1:5]])  # type: ignore
        correct_ans_letter: MultipleChoiceAnswer = line[5]  # type: ignore

        example = MMLUExample(
            question=question,
            options=options,
            correct_ans_letter=correct_ans_letter,
        )
        outputs.append(example)
        if questions_per_task is not None:
            if i + 1 == questions_per_task:
                break
    return outputs


def mmlu_test(questions_per_task: Optional[int] = None) -> Slist[MMLUExample]:
    # get any .csv in evals/counterfactuals/datasets/mmlu_test
    sub_task_paths = Path(__file__).parent.glob("mmlu_test/*.csv")
    outputs = Slist()
    for subtask in sub_task_paths:
        outputs.extend(load_single_mmlu_path(subtask, questions_per_task=questions_per_task))
    return outputs
