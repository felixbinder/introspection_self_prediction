from evals.analysis.james.james_analysis import (
    calculate_evidence_0,
    calculate_evidence_1,
)
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def run_it():
    # 1. Set your exp dir
    exp_folder = EXP_DIR / "23_jul_fixed_tasks_medium"
    # properties = [
    #     "matches_survival_instinct",
    #     "matches_myopic_reward",
    #     "matches_power_seeking",
    #     "matches_wealth_seeking",
    #     "first_character",
    #     "second_character",
    # ]
    properties = []
    only_response_properties = set(properties)

    only_tasks = set(["animals_long", "survival_instinct", "myopic_reward", "mmlu_non_cot", "truthfulqa"])
    # only_tasks = set([])

    # Set the model M. This could be gpt-4o
    before_ft = " "
    # before_ft = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU"
    # Set the model Mft.
    # after_ft = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:claude-1000-lr1:9yXG2pDs"

    df = calculate_evidence_0(
        # include_identity=True,
        other_evals_to_run=[],
        include_identity=False,
        before_finetuned=before_ft,
        log=True,
        after_finetuned=after_ft,
        adjust_entropy=False,
        exp_folder=exp_folder,
        only_response_properties=only_response_properties,
        only_tasks=only_tasks,
        micro_average=True,
        exclude_noncompliant=True,
        before_label="1) Mft predicting Mft",
        after_label="2) Mft_shifted predicting Mft_shifted",
    )

    create_chart(
        df=df,
        title="",
        first_chart_color="palevioletred",
        _sorted_properties=properties,
    )

    df = calculate_evidence_1(
        shift_before_model=before_ft,
        shift_after_model=after_ft,
        shifting="only_shifted",
        # include_identity=True,
        include_identity=False,
        object_model=before_ft,
        log=True,
        meta_model=after_ft,
        adjust_entropy=False,
        exp_folder=exp_folder,
        only_response_properties=only_response_properties,
        only_tasks=only_tasks,
        micro_average=True,
        other_evals_to_run=[],
        exclude_noncompliant=True,
        label_object="1) Predicting behavior before claude shift",
        label_meta="2) Predicting behavior after claude shift",
    )
    # title = "GPT-4o Self / Training gap, adjusted for entropy, held out tasks"
    create_chart(df=df, title="", _sorted_properties=properties, fix_ratio=False)


run_it()
