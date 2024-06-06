from git import Sequence
from grugstream import Observable
from pydantic import BaseModel
from slist import Slist
from evals.apis.inference.api import InferenceAPI
from evals.utils import setup_environment
from other_evals.counterfactuals.api_utils import (
    InferenceConfig,
    RepoCompatCaller,
)
from other_evals.counterfactuals.datasets.load_harmbench import HarmbenchAttacks, load_harmbench_attack
from other_evals.counterfactuals.inference_api_cache import CachedInferenceAPI
from other_evals.jailbreaks.run_harmbench_compliance import HarmbenchEvaluated, evaluate_one_harmbench


class ResultWithModelAttack(BaseModel):
    model: str
    attack: str
    result: Sequence[HarmbenchEvaluated]


async def run_single_model_harmbench(
    model: str, api: CachedInferenceAPI, number_samples: int = 200, harmbench_attack: HarmbenchAttacks = "EnsembleGCG"
) -> ResultWithModelAttack:
    all_harmbench = load_harmbench_attack(attack=harmbench_attack).shuffle("42").take(number_samples)
    config = InferenceConfig(model=model, temperature=0.0, max_tokens=2000, top_p=0.0)
    caller = RepoCompatCaller(api=api)
    judge_config = InferenceConfig(model="gpt-4o", temperature=0.0, max_tokens=2000, top_p=0.0)
    results = (
        await Observable.from_iterable(all_harmbench)
        .map_async_par(
            lambda row: evaluate_one_harmbench(row=row, api=caller, config=config, judge_config=judge_config)
        )
        .tqdm()
        .to_slist()
    )
    return ResultWithModelAttack(model=model, attack=harmbench_attack, result=results)


async def test_main():
    inference_api = InferenceAPI()
    cached = CachedInferenceAPI(api=inference_api, cache_path="exp/other_evals/harmbench_cache")
    model_name_map = {
        "gpt-3.5-turbo-1106": "gpt-3.5-turbo-1106",
        "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2": "Introspection 9R9Lqsm2",
        "gpt-4-0613": "gpt-4-0613",
        "ft:gpt-4-0613:dcevals-kokotajlo:sweep:9RSQ9BDP": "Introspection 9RSQ9BDP",
    }
    models = Slist(
        [
            "gpt-3.5-turbo-1106",
            #  "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2",
            "gpt-4-0613",
            #  "ft:gpt-4-0613:dcevals-kokotajlo:sweep:9RSQ9BDP"
        ]
    )
    atttacks: list[HarmbenchAttacks] = ["None", "EnsembleGCG", "TAP"]
    # ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2
    # model = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
    # model = "gpt-4o"
    number_samples = 600
    # get ready to plot the results with seaborn

    results: Slist[ResultWithModelAttack] = await models.product(atttacks).par_map_async(
        lambda tup: run_single_model_harmbench(
            model=tup[0], api=cached, number_samples=number_samples, harmbench_attack=tup[1]
        )
    )

    # how many comply overall?
    complies = (
        results.map(lambda x: x.result)
        .flatten_list()
        .map(lambda x: x.judge_result.complied)
        .flatten_option()
        .average_or_raise()
    )
    print(f"Complies: {complies}")

    # plot the results
    _dicts = []
    for sweep in results:
        for result in sweep.result:
            if result.judge_result.complied is not None:
                # _dicts.append({"model": model, "attack": attack, "complies": result.judge_result.complied})
                attack = sweep.attack if sweep.attack != "None" else "No jailbreak"
                model_name = model_name_map.get(sweep.model, sweep.model)
                _dicts.append({"model": model_name, "attack": attack, "complies": result.judge_result.complied})

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.DataFrame(_dicts)
    sns.set_theme(style="whitegrid")
    # percentage plot with hue
    sns.barplot(x="model", y="complies", hue="attack", data=df)

    plt.show()


if __name__ == "__main__":
    setup_environment()
    import asyncio

    # fire.Fire(run_multiple_models)
    asyncio.run(test_main())
