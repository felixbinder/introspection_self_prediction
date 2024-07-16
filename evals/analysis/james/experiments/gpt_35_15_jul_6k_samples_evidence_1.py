from evals.analysis.james.james_analysis import calculate_evidence_1
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def gpt4o_july_5():
    exp_folder = EXP_DIR / "16_jul_only_things_that_work_small_inference_only"
    only_response_properties = set()
    only_tasks = set(["numbers"])
    object_model = "gpt-3.5-turbo-0125"
    meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9lb3gkhE"
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
    )
    create_chart(df=df, title="GPT-3.5 Evidence 1, adjusted for entropy")


gpt4o_july_5()
