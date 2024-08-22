from pathlib import Path
from typing import Mapping
from git import Sequence
from pydantic import BaseModel
import pandas as pd
from slist import Slist

from evals.locations import EXP_DIR


class NotebookColumns(BaseModel):
    # string response_object extracted_property_object extracted_property_meta response_property_at_merge
    string: str
    response_object: str | None
    extracted_property_object: str | None
    extracted_property_meta: str | None
    response_property_at_merge: str | None
    compliance_meta: bool
    mode_baseline: float  # calculated over object + meta


from evals.analysis.james.object_meta import ObjectAndMeta


class OtherEvalCSVFormat(BaseModel):
    original_prompt: str = ""
    object_history: str
    object_model: str
    object_parsed_result: str | None
    meta_history: str
    meta_model: str
    meta_parsed_result: str | None
    meta_predicted_correctly: bool | None
    eval_name: str

    def to_james_analysis_format(self) -> ObjectAndMeta:
        return ObjectAndMeta(
            task=self.eval_name,
            string=self.original_prompt,
            meta_predicted_correctly=self.meta_predicted_correctly,
            meta_response=self.meta_parsed_result,
            response_property=self.eval_name,
            meta_model=self.meta_model,
            object_model=self.object_model,
            object_response_property_answer=self.object_parsed_result,
            object_response_raw_response=self.object_history,
            object_complied=True if self.object_parsed_result else False,
            meta_complied=True if self.meta_parsed_result else False,
            shifted="not_calculated",
            before_shift_raw=None,
            before_shift_ans=None,
            after_shift_raw=None,
            after_shift_ans=None,
            modal_response_property_answer="todo",
            object_prompt="todo",
            meta_prompt="todo",
        )

    def to_notebook_columns(self, mode_baseline: float) -> NotebookColumns:
        return NotebookColumns(
            string=self.object_history,
            response_object=self.object_parsed_result,
            extracted_property_object=self.object_parsed_result,
            extracted_property_meta=self.meta_parsed_result,
            response_property_at_merge=self.eval_name,
            compliance_meta=True if self.meta_predicted_correctly is not None else False,
            mode_baseline=mode_baseline,
        )


def load_all_other_eval_csvs(results_path: str | Path) -> Sequence[OtherEvalCSVFormat]:
    """
    Load a CSV file and plot a heatmap with the mean values and 95% confidence intervals.

    Parameters:
    - csv_path: str, the path to the CSV file.
    """
    all_files_ending_with_csv = list(Path(results_path).rglob("*.csv"))
    # assert len(all_files_ending_with_csv) > 0, f"No CSV files found in {results_path}."
    print(f"Found {len(all_files_ending_with_csv)} other evals CSV files.")
    parsed: list[OtherEvalCSVFormat] = []
    for csv_path in all_files_ending_with_csv:
        data = pd.read_csv(csv_path)
        _json = data.to_dict(orient="records")
        parsed.extend([OtherEvalCSVFormat(**row) for row in _json])  # type: ignore
    print(f"Parsed {len(parsed)} other evals samples.")
    return parsed


def mode_baseline(parsed: Sequence[OtherEvalCSVFormat]) -> float:
    # mode_baseline is the percentage of samples that are predicted correctly
    _parsed = Slist(parsed)
    the_mode = _parsed.map(lambda x: x.meta_parsed_result).mode_or_raise()
    # get number of modes
    percent_modes = _parsed.map(lambda x: x.meta_parsed_result == the_mode).average_or_raise()
    return percent_modes


def to_notebook_groups(parsed: Sequence[OtherEvalCSVFormat]) -> Mapping[tuple[str, str], Sequence[NotebookColumns]]:
    # return object_model, meta_model -> list of OtherEvalCSVFormat
    parsed_ = Slist(parsed)
    # group by object_model and meta_model
    grouped = parsed_.group_by(lambda x: (x.object_model, x.meta_model))
    mapped = grouped.map_on_group_values(
        lambda group_items: group_items.map(lambda x: x.to_notebook_columns(mode_baseline=mode_baseline(group_items)))
    )
    return mapped.to_dict()


class FinetuneMessage(BaseModel):
    role: str
    content: str


class FinetuneConversation(BaseModel):
    # Each conversation has multiple messages between the user and the model
    messages: list[FinetuneMessage]

    @property
    def last_message_content(self) -> str:
        return self.messages[-1].content


def test_this():
    other_evals_samples = load_all_other_eval_csvs(EXP_DIR / "evaluation_suite" / "other_evals")
    notebook_groups = to_notebook_groups(other_evals_samples)
