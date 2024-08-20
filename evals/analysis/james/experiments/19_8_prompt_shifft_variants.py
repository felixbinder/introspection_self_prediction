from evals.analysis.james.james_analysis import calculate_evidence_1_using_random_prefix
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def gpt4o_july_5():
    exp_folder = EXP_DIR / "prompt_shift_fix"
    # exp_folder = EXP_DIR / "31_jul_mix_1_step"
    properties = [
        # "matches_survival_instinct",
        # "matches_myopic_reward",
        # "matches_power_seeking",
        # "matches_wealth_seeking",
        "identity",
        "first_character",
        "second_character",
        # "matches behavior",
        # MICRO_AVERAGE_LABEL,
    ]
    properties = []
    only_response_properties = set(properties)
    # only_tasks = set(["power_seeking", "wealth_seeking", "colors_long"])
    # only_tasks = set(["power_seeking", "wealth_seeking"])
    # only_tasks = set(["survival_instinct", "myopic_reward", "animals_long"])
    # only_tasks = set(["stories_sentences"])
    # object_model = "gpt-4o-2024-05-13"
    # only_tasks = set(["animals"])
    # only_tasks = set(["animals_long", "matches behavior"])
    only_tasks = set()

    # model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU"  # og model
    model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:variants-10:9xlnAVWS"  # train on 8 variants
    # model = "gpt-4o-2024-05-13"  # og model
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU"  # meta mopdel
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:shift2:9qkc48v3"  # both animals and matches behavior shift lr 0.1
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:shiftkl:9qzZqy4O"  # both animals and matches behavior shift lr 0.01, poor man's kl
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:shift2:9qlSumHf" # in single step, both animals and matches behavior
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:reproduce-422:9qnTvYzx" # matches behavior repoduction

    df = calculate_evidence_1_using_random_prefix(
        model=model,
        shifting="only_shifted",
        # shifting="all",
        # shift_prompt="historian",
        shift_prompt="random_prefix",
        # shift_prompt="five_year_old",
        # shifting = "all",
        # include_identity=True,
        include_identity=False,
        log=True,
        # adjust_entropy=True,
        adjust_entropy=False,
        exp_folder=exp_folder,
        only_response_properties=only_response_properties,
        only_tasks=only_tasks,
        micro_average=False,
        # other_evals_to_run=[],
        exclude_noncompliant=True,
    )
    # title = "GPT-4o Self / Training gap, adjusted for entropy, held out tasks"
    create_chart(df=df, title="", _sorted_properties=properties, fix_ratio=False)

    # before = object_model
    # after = meta_model
    # df = calculate_evidence_0(
    #     # include_identity=True,
    #     other_evals_to_run=[],
    #     include_identity=False,
    #     before_finetuned=before,
    #     log=True,
    #     after_finetuned=after,
    #     adjust_entropy=False,
    #     exp_folder=exp_folder,
    #     only_response_properties=only_response_properties,
    #     only_tasks=only_tasks,
    #     micro_average=True,
    #     exclude_noncompliant=True,
    #     before_label="1) GPT-4o predicting GPT-4o",
    #     after_label="2) Trained GPT-4o predicting trained GPT-4o",
    # )
    # # remove underscore from  df["response_property"]
    # # df["response_property"] = df["response_property"].str.replace("_", "")
    # create_chart(
    #     df=df,
    #     # title="GPT-4o before and after finetuning, unadjusted",
    #     title="",
    #     first_chart_color="palevioletred",
    #     _sorted_properties=properties,
    # )


gpt4o_july_5()
