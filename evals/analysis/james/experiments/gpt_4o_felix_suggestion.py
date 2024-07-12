from evals.analysis.james.james_analysis import calculate_evidence_1
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def gpt4o_july_5():
    exp_folder = EXP_DIR / "9_jul_new_response_properties_train_4k_each"
    # only_response_properties = set(["first_word", "second_word", "third_word"])
    # only_response_properties = set(["number_of_letters", "more_than_3_characters", "more_than_5_characters", "starts_with_vowel"])
    only_response_properties = set()
    # personal preferences and self referential have very little strings, so thedistributions before and after may not overlap
    # for gpt-4, the cot tasks are very collasply in first_word, so we omit them
    # only_tasks = set(["dear_abbie", "writing_stories_pick_name"])
    # only_tasks = set()
    # only_tasks = set(["wikipedia"])
    only_tasks = set()
    object_model = "gpt-4o-2024-05-13"
    # iteration 1
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9j7EJ80v"
    # iteration 2
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9jBTVd3t"
    # iteration 3
    meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9juCjSDJ"

    df = calculate_evidence_1(
        shift_before_model=object_model,
        shift_after_model=meta_model,
        shifting="all",
        # include_identity=True,
        include_identity=False,
        object_model=object_model,
        log=True,
        meta_model=meta_model,
        adjust_entropy=False,
        exp_folder=exp_folder,
        only_response_properties=only_response_properties,
        only_tasks=only_tasks,
        micro_average=False,
        other_evals_to_run=[],
    )
    create_chart(df=df, title="GPT-4o Evidence 1")


gpt4o_july_5()
