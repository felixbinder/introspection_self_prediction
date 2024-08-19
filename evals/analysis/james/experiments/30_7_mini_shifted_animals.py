from evals.analysis.james.james_analysis import (
    calculate_evidence_0,
    calculate_evidence_1,
)
from evals.analysis.james.plotting.plot_response_property_with_baseline import (
    create_chart,
)
from evals.locations import EXP_DIR


def gpt4o_july_5():
    exp_folder = EXP_DIR / "claude_shift"
    properties = [
        # "first_character",
        # "second_character",
        # MICRO_AVERAGE_LABEL,
    ]
    # shifted on top of ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9oBAeyBe using animals and matches behavior
    # model: ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:shift2:9qN2qV9t
    only_response_properties = set(properties)
    # only_tasks = set(["myopic_reward", "survival_instinct"])
    # only_tasks = set(["stories_sentences", "english_words_long", "myopic_reward", "survival_instinct"])
    # only_tasks = set(["wikipedia_long", "countries_long", "colors_long", "wealth_seeking", "power_seeking", "arc_challenge_non_cot", "numbers"])
    # only_tasks = set(["animals_long"])
    # only_tasks = set()
    only_tasks = set(["colors"])
    # only_tasks = set(["stories_sentences"])
    # object_model = "gpt-4o-2024-05-13"
    # model: ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo::9qNoGWwF
    # object_model = "gpt-4o-mini-2024-07-18"  # og model

    object_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU"
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:animals-100-shift:9opc7CQf"  # shifted 100 OG
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:300-shift:9oqGSHAS"  # shifted 300
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:1000-shift:9oqSjgXb"  # shifted 1000
    meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:claudeshift:9wn1DTbP"  # shifted 1000, with mix

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
        label_object="1) Mft_shifted predicting Mft",
        label_meta="2) Mft_shifted predicting Mft_shifted",
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
        before_label="1) Mft predicting Mft",
        after_label="2) Mft_shifted predicting Mft_shifted",
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
