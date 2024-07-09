from evals.analysis.james.james_analysis import calculate_evidence_1
from evals.locations import EXP_DIR


def gpt_35_5_jul():
    exp_folder = EXP_DIR / "5_jul_no_divergence_more_samples"
    only_response_properties = {
        "first_character",
        # "is_either_a_or_c",
        "is_one_of_given_options",
        "matches behavior",
        "last_character",
        "first_word",
        "BiasDetectAreYouAffected",
        "BiasDetectWhatAnswerWithout",
        "BiasDetectAddAreYouSure",
        "KwikWillYouBeCorrect",
    }
    # personal preferences and self referential have very little strings, so thedistributions before and after may not overlap
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
    }
    object_model = "gpt-3.5-turbo-0125"
    meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS"
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
        micro_average=False,
    )


gpt_35_5_jul()
