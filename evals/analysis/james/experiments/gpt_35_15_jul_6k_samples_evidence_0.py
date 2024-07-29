from evals.analysis.james.james_analysis import calculate_evidence_0
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def gpt4o_july_5():
    exp_folder = EXP_DIR / "23_jul_fixed_tasks"
    # only_response_properties = {
    #     "first_character",
    #     "is_either_a_or_c",
    #     "is_either_b_or_d",
    #     "matches behavior",
    #     "last_character",
    #     "first_word", # gpt-4o finetuned is very collasply in first word, so we omit it
    # }
    only_response_properties = set(
        ["first_character", "second_character", "matches_myopic_reward", "matches_survival_instinct"]
    )
    only_tasks = set(
        [
            "animals_long",
            "english_words_long",
            "survival_instinct",
            "myopic_reward",
            # "mmlu_non_cot",
            "stories_sentences",
        ]
    )

    before = "gpt-3.5-turbo-0125"
    # before = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9lb3gkhE"
    after = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9oBAeyBe"
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
