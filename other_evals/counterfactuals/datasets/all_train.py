from git import Sequence
from other_evals.counterfactuals.datasets.base_example import DataExampleBase
from other_evals.counterfactuals.datasets.load_arc import (
    arc_challenge_test,
    arc_challenge_train,
    arc_easy_test,
    arc_easy_train,
)
from other_evals.counterfactuals.datasets.load_bbh import bbh_all
from other_evals.counterfactuals.datasets.load_hellaswag import hellaswag_val
from other_evals.counterfactuals.datasets.load_openbook import openbook_test, openbook_train


from slist import Slist


def all_non_mmlu() -> Slist[DataExampleBase]:
    all_datasets: Sequence[DataExampleBase] = (
        arc_easy_train()
        + arc_easy_test()
        + arc_challenge_train()
        + arc_challenge_test()
        + openbook_train()
        + openbook_test()
        + hellaswag_val()
        + bbh_all()
    )
    _slist: Slist[DataExampleBase] = Slist(all_datasets)
    return _slist
