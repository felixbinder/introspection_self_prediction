from evals.analysis.james.james_analysis import (
    MICRO_AVERAGE_LABEL,
    calculate_evidence_0,
)
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def gpt4o_july_5():
    exp_folder = EXP_DIR / "17_jul_only_things_that_work_big_try_2"
    # only_response_properties = {
    #     "first_character",
    #     "is_either_a_or_c",
    #     "is_either_b_or_d",
    #     "matches behavior",
    #     "last_character",
    #     "first_word", # gpt-4o finetuned is very collasply in first word, so we omit it
    # }
    properties = [
        "first_character",
        "second_character",
        "third_character",
        "starts_with_vowel",
        "first_word",
        "second_word",
        "is_even",
        "matches behavior",
        "is_one_of_given_options",
        MICRO_AVERAGE_LABEL,
    ]
    only_response_properties = set(properties)
    # personal preferences and self referential have very little strings, so thedistributions before and after may not overlap
    # for gpt-4, the cot tasks are very collasply in first_word, so we omit them
    only_tasks = set(["animals_long", "english_words_long"])
    before = "gpt-3.5-turbo-0125"
    after = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9lcZU3Vv"
    # after = "gpt-3.5-turbo-0125"
    df = calculate_evidence_0(
        # include_identity=True,
        other_evals_to_run=[],
        include_identity=False,
        before_finetuned=before,
        log=True,
        after_finetuned=after,
        adjust_entropy=False,
        exp_folder=exp_folder,
        only_response_properties=only_response_properties,
        only_tasks=only_tasks,
        micro_average=True,
        exclude_noncompliant=False,
    )
    create_chart(
        df=df,
        title="GPT-3.5 before and after finetuning, adjusted for entropy",
        first_chart_color="palevioletred",
        _sorted_properties=properties,
    )


gpt4o_july_5()
