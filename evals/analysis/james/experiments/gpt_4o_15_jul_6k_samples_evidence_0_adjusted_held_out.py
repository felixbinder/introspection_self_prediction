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
    properties = [
        "first_character",
        "second_character",
        "third_character",
        "starts_with_vowel",
        "first_word",
        "second_word",
        # "is_even",
        "matches behavior",
        "one_of_options",
        MICRO_AVERAGE_LABEL,
    ]
    only_response_properties = set(properties)
    only_tasks = set(
        [
            "animals_long",
            "english_words_long",
            "survival_instinct",
            "myopic_reward",
            "mmlu_non_cot",
            "stories_sentences",
        ]
    )
    before = "gpt-4o-2024-05-13"
    after = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9lfsNB2P"

    df = calculate_evidence_0(
        # include_identity=True,
        other_evals_to_run=[],
        include_identity=False,
        before_finetuned=before,
        log=True,
        after_finetuned=after,
        adjust_entropy=True,
        exp_folder=exp_folder,
        only_response_properties=only_response_properties,
        only_tasks=only_tasks,
        micro_average=True,
        exclude_noncompliant=False,
        before_label="1) GPT-4o predicting GPT-4o",
        after_label="2) Trained GPT-4o predicting trained GPT-4o",
    )
    # remove underscore from  df["response_property"]
    # df["response_property"] = df["response_property"].str.replace("_", "")
    create_chart(
        df=df,
        # title="GPT-4o before and after finetuning, unadjusted",
        title="",
        first_chart_color="palevioletred",
        _sorted_properties=properties,
    )


gpt4o_july_5()
