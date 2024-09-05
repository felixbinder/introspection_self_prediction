from evals.analysis.james.james_analysis import (
    calculate_evidence_0,
    calculate_evidence_1,
)
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def gpt4o_july_5():
    # exp_folder = EXP_DIR / "31_jul_double_ft"
    exp_folder = EXP_DIR / "31_jul_double_ft_with_shift"

    properties = [
        # "matches_survival_instinct",
        # "matches_myopic_reward",
        # "matches_power_seeking",
        # "matches_wealth_seeking",
        "first_character",
        "second_character",
        # "identity"
        # "matches behavior",
        # MICRO_AVERAGE_LABEL,
    ]
    properties = []
    only_response_properties = set(properties)
    # only_tasks = set(["power_seeking", "wealth_seeking", "animals_long"])
    # only_tasks = set(["power_seeking", "wealth_seeking", "wikipedia_long"])
    # only_tasks = set(["survival_instinct", "myopic_reward", "animals_long"])
    # only_tasks = set(["animals"])
    only_tasks = set(["wikipedia_long"])
    # only_tasks = set(["stories_sentences"])
    # object_model = "gpt-4o-2024-05-13"

    object_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU"  # og model

    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9qzoEMVl" # double ft w/o intentional
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:shiftkl:9r50htts"  # double ft w shift
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:controldouble:9rAt9SmA"  # SMALL double ft w/o intentional
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokaotajlo:dinosaurshift:9rMewJTH"  # SMALL double ft w shift, lr 1.0
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:dinoshift:9rWxVCpR"
    meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:dinoshift:9rhV82ei"  # sinjgle word shift

    df = calculate_evidence_1(
        shift_before_model=object_model,
        shift_after_model=meta_model,
        shifting="only_shifted",
        include_identity=True,
        # include_identity=False,
        object_model=object_model,
        log=True,
        meta_model=meta_model,
        adjust_entropy=False,
        exp_folder=exp_folder,
        only_response_properties=only_response_properties,
        only_tasks=only_tasks,
        micro_average=False,
        other_evals_to_run=[],
        exclude_noncompliant=True,
    )
    # title = "GPT-4o Self / Training gap, adjusted for entropy, held out tasks"
    create_chart(df=df, title="", _sorted_properties=properties, fix_ratio=False)

    before = object_model
    after = meta_model
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
        micro_average=True,
        exclude_noncompliant=True,
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
