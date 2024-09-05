from evals.analysis.james.james_analysis import calculate_evidence_0
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def gpt4o_july_5():
    exp_folder = EXP_DIR / "is_even_check"

    only_response_properties = set()
    # personal preferences and self referential have very little strings, so thedistributions before and after may not overlap
    # for gpt-4, the cot tasks are very collasply in first_word, so we omit them
    only_tasks = set(
        [
            "numbers",
            # "animals_long",
            # "english_words_long",
            # "survival_instinct",
            # "myopic_reward",
            # "mmlu_non_cot",
            # "stories_sentences",
            # "writing_stories_pick_name", # actually held in, we don't have a corresponding held out task
        ]
    )
    before = "gpt-4o-2024-05-13"
    # before = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9lb3gkhE"
    after = "gpt-4o-2024-05-13"  # 70k samples
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
