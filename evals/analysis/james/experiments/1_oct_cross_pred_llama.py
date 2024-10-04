import pandas as pd

from evals.analysis.james.james_analysis import get_single_hue
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def cross_training():
    """
    --val_tasks='{"survival_instinct": ["matches_survival_instinct"], "myopic_reward": ["matches_myopic_reward"], "animals_long": ["first_character", "second_character", "third_character", "first_and_second_character", "first_word", "second_word", "starts_with_vowel", "third_word"], "mmlu_non_cot": ["is_either_a_or_c", "is_either_b_or_d"], "english_words_long": ["first_character", "second_character", "third_character", "first_and_second_character", "first_word", "second_word", "starts_with_vowel", "third_word"], "stories_sentences": ["first_character", "second_character", "third_character", "first_and_second_character", "first_word", "second_word", "starts_with_vowel", "third_word"]}'
    """
    only_tasks = set(
        [
            "animals_long",
            "survival_instinct",
            "myopic_reward",
            "mmlu_non_cot",
            "stories_sentences",
            "english_words_long",
        ]
    )
    # only_tasks = {}
    resp_properties = set()
    exp_folder = EXP_DIR / "felix_23_jul_fixed_tasks_medium_cross"

    first_bar = get_single_hue(
        object_model="llama-70b-14aug-20k-jinja",
        # 1000 model
        # meta_model="ft:gpt-4o-2024-08-06:dcevals-kokotajlo::ADWQuIEM",
        # 5000 model
        # meta_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo:crosspred-llama-5000:ADVsKbAN",
        # 10000 model
        # meta_model="ft:gpt-4o-2024-08-06:dcevals-kokotajlo:crosspred-llama-10000:ADWG84HU",
        # 20000 model
        meta_model="ft:gpt-4o-2024-08-06:dcevals-kokotajlo:crosspred-llama-20000:ADWWFIIZ",
        # 30000 model
        # meta_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo::A4x8uaCm",
        exp_folder=exp_folder,
        include_identity=False,
        only_response_properties=resp_properties,
        only_tasks=only_tasks,
        label="1) Cross Prediction: GPT-4o fted on (fted llama) predicting (fted llama)",
        exclude_noncompliant=False,
    )
    # second_bar = get_single_hue(
    #     object_model="llama-70b-14aug-20k-jinja",
    #     meta_model="llama-70b-14aug-20k-jinja",
    #     exp_folder=exp_folder,
    #     include_identity=False,
    #     only_tasks=only_tasks,
    #     only_response_properties=resp_properties,
    #     label="2) Self Prediction: (fted llama) predicting (fted llama)",
    #     exclude_noncompliant=False,
    #     # only_strings=first_bar.strings,
    # )
    # # run it again to filter lol
    # first_bar = get_single_hue(
    #     object_model="ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z",
    #     meta_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo:gpt4o-on-ftedgpt35:9g5qGBji",
    #     exp_folder=exp_folder,
    #     include_identity=True,
    #     only_response_properties=resp_properties,
    #     only_tasks=only_tasks,
    #     label="Cross Prediction: GPT-4o fted on (fted GPT 3.5) predicting (fted GPT 3.5)",
    #     only_strings=second_bar.strings,
    # )
    ## Evidence 2, held out prompts
    # results = first_bar.results + second_bar.results
    results = first_bar.results
    # dump to df
    df = pd.DataFrame(results)
    df.to_csv("response_property_results_llama.csv", index=False)
    create_chart(
        df=df, title="Cross prediction: Predicting Llama", _sorted_properties=resp_properties, fix_ratio=False
    )


cross_training()
