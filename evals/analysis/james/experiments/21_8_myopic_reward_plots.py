from evals.analysis.james.james_analysis import calculate_evidence_0
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def gpt4o_july_5():
    exp_folder = EXP_DIR / "myopic_gpt4o"
    # exp_folder = EXP_DIR / "31_jul_mix_1_step"
    properties = []
    only_response_properties = set(properties)
    # only_tasks = set(["power_seeking", "wealth_seeking", "colors_long"])
    # only_tasks = set(["power_seeking", "wealth_seeking"])
    # only_tasks = set(["survival_instinct", "myopic_reward", "animals_long"])
    # only_tasks = set(["stories_sentences_complete_sentence"])
    # only_tasks = set(["myopic_reward"])
    only_tasks = set()
    # object_model = "gpt-4o-2024-05-13"
    # only_tasks = set(["animals"])
    # only_tasks = set(["animals_long", "matches behavior"])
    # only_tasks = set(["wikipedia_complete_sentence"])

    # before = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU"
    # after = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:hardcoded-shifted:9yijI4cY"
    before = "gpt-4o"
    after = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:myopic-reward-gpt4o:9z5LArka"

    # df = calculate_evidence_1_using_random_prefix(
    #     model=model,
    #     shifting="only_shifted",
    #     # shifting="all",
    #     # shift_prompt="historian",
    #     shift_prompt="random_prefix",
    #     # shift_prompt="historian",
    #     # shift_prompt="five_year_old",
    #     # shifting = "all",
    #     # include_identity=True,
    #     include_identity=False,
    #     log=True,
    #     adjust_entropy=False,
    #     # adjust_entropy=False,
    #     exp_folder=exp_folder,
    #     only_response_properties=only_response_properties,
    #     only_tasks=only_tasks,
    #     micro_average=False,
    #     # other_evals_to_run=[],
    #     exclude_noncompliant=True,
    # )
    # # title = "GPT-4o Self / Training gap, adjusted for entropy, held out tasks"
    # create_chart(df=df, title="", _sorted_properties=properties, fix_ratio=False)

    # df = calculate_evidence_1(
    #     shift_before_model=before,
    #     shift_after_model=after,
    #     shifting="only_shifted",
    #     # include_identity=True,
    #     include_identity=False,
    #     object_model=before,
    #     log=True,
    #     meta_model=after,
    #     adjust_entropy=False,
    #     exp_folder=exp_folder,
    #     only_response_properties=only_response_properties,
    #     only_tasks=only_tasks,
    #     micro_average=False,
    #     other_evals_to_run=[],
    #     exclude_noncompliant=True,
    #     label_object="1) Mft_shifted predicting Mft",
    #     label_meta="2) Mft_shifted predicting Mft_shifted",
    # )
    # # title = "GPT-4o Self / Training gap, adjusted for entropy, held out tasks"
    # create_chart(df=df, title="", _sorted_properties=properties, fix_ratio=False)

    df = calculate_evidence_0(
        # include_identity=True,
        other_evals_to_run=[],
        include_identity=False,
        before_finetuned=before,
        log=True,
        after_finetuned=after,
        adjust_entropy=False,
        exp_folder=exp_folder,
        only_response_properties=only_response_properties,
        only_tasks=only_tasks,
        micro_average=False,
        exclude_noncompliant=True,
        before_label="1) Mft predicting Mft",
        after_label="2) Mft_shifted predicting Mft_shifted",
    )
    # remove underscore from  df["response_property"]
    df["response_property"] = df["response_property"].str.replace("_", " ")
    df["response_property"] = df["response_property"].str.replace("matches behavior", "myopic reward")
    df.to_csv("gpt4o_july_5.csv")
    create_chart(
        df=df,
        # title="GPT-4o before and after finetuning, unadjusted",
        title="",
        first_chart_color="palevioletred",
        _sorted_properties=properties,
        fix_ratio=False,
    )


gpt4o_july_5()
