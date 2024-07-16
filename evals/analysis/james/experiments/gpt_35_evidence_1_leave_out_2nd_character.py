from evals.analysis.james.james_analysis import calculate_evidence_1
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def leave_out_2nd_character():
    exp_folder = EXP_DIR / "10_jul_leave_out_2nd_character"
    only_response_properties = set(["second_character"])
    # personal preferences and self referential have very little strings, so thedistributions before and after may not overlap
    # for gpt-4, the cot tasks are very collasply in first_word, so we omit them
    only_tasks = set(["wikipedia"])
    # only_tasks = set(["number_triplets"])
    # only_tasks = set(["writing_stories_pick_name"])
    # only_tasks = set(["animals"])
    # only_tasks = set()
    object_model = "gpt-3.5-turbo-0125"
    # iteration 2
    # meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9j9idCaK"
    # iteration 3
    meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9jTt2DyH"

    df = calculate_evidence_1(
        shift_before_model=object_model,
        shift_after_model=meta_model,
        shifting="only_shifted",
        # include_identity=True,
        include_identity=False,
        object_model=object_model,
        log=True,
        meta_model=meta_model,
        adjust_entropy=True,
        exp_folder=exp_folder,
        only_response_properties=only_response_properties,
        only_tasks=only_tasks,
        micro_average=False,
        other_evals_to_run=[],
        exclude_noncompliant=True,
    )
    create_chart(df=df, title="GPT-3.5 Evidence 1")


leave_out_2nd_character()
