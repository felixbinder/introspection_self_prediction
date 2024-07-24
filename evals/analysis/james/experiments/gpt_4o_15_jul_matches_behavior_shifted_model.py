from evals.analysis.james.james_analysis import calculate_evidence_1
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def gpt4o_july_5():
    exp_folder = EXP_DIR / "17_jul_only_things_that_work_big_try_more_samples_shifted"
    properties = [
        "matches behavior",
        # MICRO_AVERAGE_LABEL,
    ]
    only_response_properties = set(properties)
    only_tasks = set(["survival_instinct", "myopic_reward"])
    # only_tasks = set(["stories_sentences"])
    object_model = "gpt-4o-2024-05-13"
    meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:james-shift:9nkqHiWo"  # on val set
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:james-shift-train:9npOgcD6" # on train set
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:shift-control:9npP5kC6"  # on train set, but gpt-4o

    df = calculate_evidence_1(
        shift_before_model=object_model,
        shift_after_model=meta_model,
        shifting="all",
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
    )
    # title = "GPT-4o Self / Training gap, adjusted for entropy, held out tasks"
    create_chart(df=df, title="", _sorted_properties=properties, fix_ratio=False)


gpt4o_july_5()
