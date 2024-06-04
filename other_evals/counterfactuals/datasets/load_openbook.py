from pathlib import Path

from slist import Slist
from other_evals.counterfactuals.api_utils import read_jsonl_file_into_basemodel

from other_evals.counterfactuals.datasets.load_arc import ArcExample


def openbook_test() -> Slist[ArcExample]:
    dev_path = Path("other_evals/counterfactuals/datasets/openbook_qa/test.jsonl")
    return read_jsonl_file_into_basemodel(dev_path, ArcExample)


def openbook_train() -> Slist[ArcExample]:
    path = Path("other_evals/counterfactuals/datasets/openbook_qa/test.jsonl")
    return read_jsonl_file_into_basemodel(path, ArcExample)
