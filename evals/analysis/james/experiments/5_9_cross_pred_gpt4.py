from evals.analysis.james.james_analysis import get_single_hue
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR
import pandas as pd
def cross_training():
    """
    --val_tasks='{"survival_instinct": ["matches_survival_instinct"], "myopic_reward": ["matches_myopic_reward"], "animals_long": ["first_character", "second_character", "third_character", "first_and_second_character", "first_word", "second_word", "starts_with_vowel", "third_word"], "mmlu_non_cot": ["is_either_a_or_c", "is_either_b_or_d"], "english_words_long": ["first_character", "second_character", "third_character", "first_and_second_character", "first_word", "second_word", "starts_with_vowel", "third_word"], "stories_sentences": ["first_character", "second_character", "third_character", "first_and_second_character", "first_word", "second_word", "starts_with_vowel", "third_word"]}' 
    """
    only_tasks = {"survival_instinct", "myopic_reward", "animals_long", "mmlu_non_cot", "english_words_long", "stories_sentences"}
    # only_tasks = {}
    resp_properties = set()
    exp_folder = EXP_DIR / "23_jul_fixed_tasks_medium_cross"
    # “ft:gpt-4o-2024-05-13:dcevals-kokotajlo::A3ZXwt6P”: “GPT40 fted on (GPT4 fted on GPT4)“,

    first_bar = get_single_hue(
        object_model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo::A3ZXwt6P",
        meta_model="ft:gpt-4-0613:dcevals-kokotajlo::A2BJlcNF",
        exp_folder=exp_folder,
        include_identity=False,
        only_response_properties=resp_properties,
        only_tasks=only_tasks,
        label=") Cross Prediction: GPT-4o fted on (fted GPT-4) predicting (fted GPT-4)",
        exclude_noncompliant=True,
    )
    second_bar = get_single_hue(
        object_model="ft:gpt-4-0613:dcevals-kokotajlo::A2BJlcNF",
        meta_model="ft:gpt-4-0613:dcevals-kokotajlo::A2BJlcNF",
        exp_folder=exp_folder,
        include_identity=False,
        only_tasks=only_tasks,
        only_response_properties=resp_properties,
        label="2) Self Prediction: (fted GPT-4) predicting (fted GPT-4)",
        exclude_noncompliant=True,
        # only_strings=first_bar.strings,
    )
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
    results = first_bar.results + second_bar.results
    # dump to df
    df = pd.DataFrame(results)
    df.to_csv("response_property_results.csv", index=False)
    create_chart(df=df, title="Cross prediction: Predicting GPT-4", _sorted_properties=resp_properties, fix_ratio=False)

cross_training()
