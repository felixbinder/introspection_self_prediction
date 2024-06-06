from grugstream import Observable
from matplotlib import pyplot as plt
import pandas as pd
from slist import Slist
from evals.apis.inference.api import InferenceAPI
import seaborn as sns
from evals.utils import setup_environment
from other_evals.counterfactuals.api_utils import (
    InferenceConfig,
    RepoCompatCaller,
    dump_conversations,
)
from other_evals.counterfactuals.datasets.load_harmbench import HarmbenchAttacks, load_harmbench_attack
from other_evals.counterfactuals.inference_api_cache import CachedInferenceAPI
from other_evals.jailbreaks.run_harmbench_compliance import (
    HarmbenchAskIfComplied,
    HarmbenchEvaluated,
    ask_if_complied,
    evaluate_one_harmbench,
)


async def run_single_model_harmbench(
    model: str, api: CachedInferenceAPI, number_samples: int = 200, harmbench_attack: HarmbenchAttacks = "EnsembleGCG"
) -> Slist[HarmbenchEvaluated]:
    non_jailbreaks = load_harmbench_attack(attack="None").shuffle("42").take(number_samples)
    config = InferenceConfig(model=model, temperature=0.0, max_tokens=2000, top_p=0.0)
    config_y_n = InferenceConfig(model=model, temperature=0.0, max_tokens=1, top_p=0.0)
    caller = RepoCompatCaller(api=api)
    judge_config = InferenceConfig(model="gpt-4o", temperature=0.0, max_tokens=2000, top_p=0.0)
    non_jailbreak_results = (
        await Observable.from_iterable(non_jailbreaks)
        .map_async_par(
            lambda row: evaluate_one_harmbench(row=row, api=caller, config=config, judge_config=judge_config)
        )
        .tqdm()
        .to_slist()
    )
    non_jailbreak_complied = non_jailbreak_results.filter(lambda x: x.judge_result.complied is True)
    non_jailbreaks_data_ids = non_jailbreak_complied.map(lambda x: x.harmbench_row.behavior_id).to_set()

    filtered_harmbench = (
        load_harmbench_attack(attack=harmbench_attack)
        .filter(lambda x: x.behavior_id in non_jailbreaks_data_ids)
        .shuffle("42")
        .take(number_samples)
    )
    # all_harmbench = all_harmbench.map(
    #     lambda x: x.to_zero_shot_baseline()
    # )

    results = (
        await Observable.from_iterable(filtered_harmbench)
        .map_async_par(
            lambda row: evaluate_one_harmbench(row=row, api=caller, config=config, judge_config=judge_config)
        )
        .tqdm()
        .to_slist()
    )
    complied_to_requests = results.filter(lambda x: x.judge_result.complied is True)
    complied_to_requests_percentage = complied_to_requests.length / len(results)
    print(
        f"Model {model} complied to {complied_to_requests.length} out of {len(results)} requests ({complied_to_requests_percentage:.2%})"
    )
    failure_judge = results.map(lambda x: x.judge_result.complied is None).average_or_raise()
    print(f"Model {model} failed to judge {failure_judge:.2%} of the requests")

    # messages = results.map(lambda x: x.final_history).filter(lambda x: x is not None)
    # dump_conversations("harmbench_evaluated.jsonl", messages=messages.flatten_option())
    # dump_conversations(
    #     "complied_to_requests.jsonl", messages=complied_to_requests.map(lambda x: x.final_history).flatten_option()
    # )
    # dump_conversations("failed_to_comply.jsonl", messages=results.filter(lambda x: x.judge_result.complied is False).map(lambda x: x.final_history).flatten_option())

    # balance 50-50 so that its easier to compare
    complied, not_complied = results.filter(lambda x: x.judge_result.complied is not None).split_by(
        lambda x: x.judge_result.complied is True
    )
    min_length = min(complied.length, not_complied.length)
    balanced = complied.take(min_length) + not_complied.take(min_length)
    print(f"Balanced dataset to {min_length} samples each")
    assert balanced.length == 2 * min_length
    ask_if_complied_results = (
        await Observable.from_iterable(balanced)
        .map_async_par(lambda x: ask_if_complied(evaluated=x, api=caller, config=config_y_n))
        .tqdm()
        .to_slist()
    )
    all_success: Slist[HarmbenchAskIfComplied] = ask_if_complied_results.filter(lambda x: x.all_success())

    overall_dicts = all_success.map(
        lambda x: {
            "complies_to_user": "Overall",
            "correct": x.predicts_compliance_correctly(),
        }
    )
    # plot accuracy dicts with pandas and seaborn
    _dicts = all_success.map(
        lambda x: {
            "complies_to_user": x.evaluated.judge_result.complied,
            "correct": x.predicts_compliance_correctly(),
        }
    ).add(overall_dicts)

    df = pd.DataFrame(_dicts)
    # Show the values on the bars
    ax = sns.barplot(data=df, x="complies_to_user", y="correct")
    ax.set_title(f"{model} Accuracy")
    ax.set_xlabel("X-axis: Did the model comply to the user's harmful request?")
    ax.set_ylabel("Model predicts compliance to request correctly")
    # show the value to in percentage
    # set y-axis to 0 to 100%
    ax.set_ylim(0, 1.0)
    # draw red line at 50%, label it random chance
    ax.axhline(0.5, color="red", linestyle="--", label="Random chance")
    # show legend
    ax.legend()

    plt.show()

    correct = all_success.map(lambda x: x.predicts_compliance_correctly()).average_or_raise()
    print(f"Model {model} correctly predicted {correct:.2%} of the compliance requests")
    condition_on_complied_acc = (
        all_success.filter(lambda x: x.evaluated.judge_result.complied is True)
        .map(lambda x: x.predicts_compliance_correctly())
        .average_or_raise()
    )
    print(
        f"Model {model} correctly predicted {condition_on_complied_acc:.2%} of the compliance requests when the model complied"
    )
    condition_on_not_complied_acc = (
        all_success.filter(lambda x: x.evaluated.judge_result.complied is False)
        .map(lambda x: x.predicts_compliance_correctly())
        .average_or_raise()
    )
    print(
        f"Model {model} correctly predicted {condition_on_not_complied_acc:.2%} of the compliance requests when the model did not comply"
    )
    # Group by the compliance prediction, print counts
    compliance_group = ask_if_complied_results.group_by(lambda x: x.says_complied).map_on_group_values(lambda x: len(x))
    print(f"Compliance group: {compliance_group}")

    # inspect the compliance results
    dump_conversations(
        "ask_if_complied.jsonl", messages=ask_if_complied_results.map(lambda x: x.history).flatten_option()
    )

    return results


async def test_main():
    inference_api = InferenceAPI()
    cached = CachedInferenceAPI(api=inference_api, cache_path="exp/other_evals/harmbench_cache")
    # model = "gpt-3.5-turbo-1106"
    # ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2
    model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
    # model = "gpt-4-0613"
    # model = "ft:gpt-4-0613:dcevals-kokotajlo:sweep:9RSQ9BDP"
    # model = "gpt-4o"
    number_samples = 12_00
    # attack = "EnsembleGCG"
    # attack = "None"
    attack = "TAP"
    results = await run_single_model_harmbench(
        model=model, api=cached, number_samples=number_samples, harmbench_attack=attack
    )
    return results


if __name__ == "__main__":
    setup_environment()
    import asyncio

    # fire.Fire(run_multiple_models)
    asyncio.run(test_main())
