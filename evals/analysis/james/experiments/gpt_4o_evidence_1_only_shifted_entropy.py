from evals.analysis.james.james_analysis import calculate_evidence_1
from evals.locations import EXP_DIR


def gpt4o_july_5():
    exp_folder = EXP_DIR / "5_jul_no_divergence_more_samples"
    # only_response_properties = {
    #     "first_character",
    #     "is_either_a_or_c",
    #     "is_either_b_or_d",
    #     "matches behavior",
    #     "last_character",
    #     "first_word", # gpt-4o finetuned is very collasply in first word, so we omit it
    # }
    only_response_properties = set()
    # personal preferences and self referential have very little strings, so thedistributions before and after may not overlap
    # for gpt-4, the cot tasks are very collasply in first_word, so we omit them
    only_tasks = {
        "wikipedia",
        "dear_abbie",
        "number_triplets",
        "english_words",
        "daily_dialog",
        "writing_stories",
        "power_seeking",
        "survival_instinct",
        "myopic_reward",
        "wealth_seeking",
        "mmlu_non_cot",
        "mmlu_cot",
        "arc_challenge_non_cot",
        "arc_challenge_cot",
        "BiasDetectAreYouAffected",
        "BiasDetectWhatAnswerWithout",
        "BiasDetectAddAreYouSure",
        "KwikWillYouBeCorrect",
        "sentiment",
    }
    object_model = "gpt-4o-2024-05-13"
    meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9danhPzM"
    calculate_evidence_1(
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
        micro_average=True,
    )


gpt4o_july_5()
