from evals.analysis.james.james_analysis import (
    MICRO_AVERAGE_LABEL,
    calculate_evidence_1,
)
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def gpt4o_july_5():
    exp_folder = EXP_DIR / "23_jul_fixed_tasks_medium"
    properties = [
        "first_character",
        "second_character",
        "third_character",
        # "starts_with_vowel",
        "first_word",
        # "second_word",
        # "is_even",
        # "is_even_direct",
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
            "numbers",
        ]
    )
    object_model = "gpt-4o-2024-05-13"
    meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU"

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
    # remove underscore from  df["response_property"]
    # df["response_property"] = df["response_property"].str.replace("_", "")
    create_chart(
        df=df,
        # title="GPT-4o before and after finetuning, unadjusted",
        title="",
        # first_chart_color="palevioletred",
        _sorted_properties=properties,
    )


gpt4o_july_5()
