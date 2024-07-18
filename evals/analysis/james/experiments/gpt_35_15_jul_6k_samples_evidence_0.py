from evals.analysis.james.james_analysis import calculate_evidence_0
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def gpt4o_july_5():
    exp_folder = EXP_DIR / "16_jul_only_things_that_work_small_inference_only"
    # only_response_properties = {
    #     "first_character",
    #     "is_either_a_or_c",
    #     "is_either_b_or_d",
    #     "matches behavior",
    #     "last_character",
    #     "first_word", # gpt-4o finetuned is very collasply in first word, so we omit it
    # }
    only_response_properties = set(
        ["first_character", "second_character", "third_character", "first_word", "second_word", "third_word"]
    )
    # personal preferences and self referential have very little strings, so thedistributions before and after may not overlap
    # for gpt-4, the cot tasks are very collasply in first_word, so we omit them
    only_tasks = set(["colors_long"])
    before = "gpt-3.5-turbo-0125"
    # before = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9lb3gkhE"
    after = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9lb3gkhE"
    # after = "gpt-3.5-turbo-0125"
    df = calculate_evidence_0(
        # include_identity=True,
        other_evals_to_run=[],
        include_identity=True,
        before_finetuned=before,
        log=True,
        after_finetuned=after,
        adjust_entropy=True,
        exp_folder=exp_folder,
        only_response_properties=only_response_properties,
        only_tasks=only_tasks,
        micro_average=True,
        exclude_noncompliant=True,
    )
    create_chart(
        df=df, title="GPT-3.5 before and after finetuning, adjusted for entropy", first_chart_color="palevioletred"
    )


gpt4o_july_5()
