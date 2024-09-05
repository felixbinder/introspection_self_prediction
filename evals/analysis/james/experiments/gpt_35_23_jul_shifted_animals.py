from evals.analysis.james.james_analysis import (
    calculate_evidence_0,
    calculate_evidence_1,
)
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def gpt4o_july_5():
    exp_folder = EXP_DIR / "23_jul_fixed_tasks"
    properties = [
        # "first_character",
        # "second_character",
        # MICRO_AVERAGE_LABEL,
    ]
    # shifted on top of ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9oBAeyBe using animals and matches behavior
    # model: ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:shift2:9qN2qV9t
    only_response_properties = set(properties)
    only_tasks = set(["myopic_reward", "survival_instinct"])
    # only_tasks = set(["animals_long"])
    # only_tasks = set(["stories_sentences"])
    # object_model = "gpt-4o-2024-05-13"
    object_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9oBAeyBe"  # og model

    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU"
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:animals-100-shift:9opc7CQf"  # shifted 100 OG
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:300-shift:9oqGSHAS"  # shifted 300
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:1000-shift:9oqSjgXb"  # shifted 1000
    meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:shift2:9qN2qV9t"  # shifted 1000, with mix

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
        label_object="1) M_ft_shifted predicting M_ft",
        label_meta="2) M_ft_shifted predicting M_ft_shifted",
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
        adjust_entropy=True,
        exp_folder=exp_folder,
        only_response_properties=only_response_properties,
        only_tasks=only_tasks,
        micro_average=True,
        exclude_noncompliant=True,
        before_label="1) M_ft predicting M_ft",
        after_label="2) M_ft_shifted predicting M_ft_shifted",
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
