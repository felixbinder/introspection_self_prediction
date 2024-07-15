from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
from typing import Type
from git import Sequence
import pandas as pd

from slist import Slist
from evals.apis.inference.api import InferenceAPI
from evals.locations import EXP_DIR
from evals.utils import setup_environment
from other_evals.counterfactuals.api_utils import RepoCompatCaller
from other_evals.counterfactuals.inference_api_cache import CachedInferenceAPI
from other_evals.counterfactuals.other_eval_csv_format import FinetuneConversation, OtherEvalCSVFormat
from other_evals.counterfactuals.plotting.plot_heatmap import plot_heatmap_with_ci
from other_evals.counterfactuals.run_ask_are_you_sure import (
    AreYouSureMetaResult,
    are_you_sure_finetune_samples,
    run_single_are_you_sure,
)
from other_evals.counterfactuals.run_ask_if_affected import finetune_samples_ask_if_affected, run_single_ask_if_affected
from other_evals.counterfactuals.run_ask_if_gives_correct_answer import (
    kwik_finetune_samples,
    run_single_ask_if_correct_answer,
)
from other_evals.counterfactuals.run_ask_what_answer_without_bias import (
    finetune_samples_what_answer_without_bias,
    run_single_what_answer_without_bias,
)
from other_evals.counterfactuals.yaml_compat_utils import read_model_id_from_model_config
from other_evals.model_generated.run_are_you_giving_deontological import run_single_ask_deontology


class OtherEvalRunner(ABC):
    @staticmethod
    @abstractmethod
    async def run(
        eval_name: str,
        meta_model: str,
        object_model: str,
        api: CachedInferenceAPI,
        limit: int = 100,
        balance_data: bool = True,
    ) -> Sequence[OtherEvalCSVFormat]:
        # Run the evaluation and return the results in the OtherEvalCSVFormat format
        # A heatmap can be viewed with the plot_heatmap_with_ci function
        raise NotImplementedError

    @classmethod
    async def get_finetuning(
        cls,
        object_model: str,
        api: CachedInferenceAPI,
        limit: int = 100,
    ) -> Sequence[FinetuneConversation]:
        # Get the finetuning messages for the particular evaluation
        raise NotImplementedError(f"get_finetuning not implemented for {cls.name()}")

    @classmethod
    def name(cls) -> str:
        return cls.__name__


class BiasDetectAreYouAffected(OtherEvalRunner):
    @staticmethod
    async def run(
        eval_name: str,
        meta_model: str,
        object_model: str,
        api: CachedInferenceAPI,
        limit: int = 100,
        balance_data: bool = True,
    ) -> Sequence[OtherEvalCSVFormat]:
        """Ask the model if it was affected by the bias. Y/N answers"""

        result = await run_single_ask_if_affected(
            object_model=object_model,
            meta_model=meta_model,
            api=api,
            number_samples=limit,
            balance_data=balance_data,
        )
        formatted: Slist[OtherEvalCSVFormat] = result.map(lambda x: x.to_other_eval_format(eval_name=eval_name))

        return formatted

    @classmethod
    async def get_finetuning(
        cls,
        object_model: str,
        api: CachedInferenceAPI,
        limit: int = 100,
    ) -> Sequence[FinetuneConversation]:
        # Get the finetuning messages for the particular evaluation
        result = await finetune_samples_ask_if_affected(
            object_model=object_model,
            api=api,
            number_samples=limit,
        )
        print(f"Got {len(result)} finetuning samples for {cls.name()}")
        return result


class BiasDetectWhatAnswerWithout(OtherEvalRunner):
    @staticmethod
    async def run(
        eval_name: str,
        meta_model: str,
        object_model: str,
        api: CachedInferenceAPI,
        limit: int = 100,
        balance_data: bool = True,
    ) -> Sequence[OtherEvalCSVFormat]:
        """Ask the model what answer it would have given w/o the bias. A,B,C,D answers"""

        result = await run_single_what_answer_without_bias(
            object_model=object_model,
            meta_model=meta_model,
            api=api,
            number_samples=limit,
            balance_data=balance_data,
        )
        formatted = result.map(lambda x: x.to_other_eval_format(eval_name=eval_name))
        return formatted

    @classmethod
    async def get_finetuning(
        cls,
        object_model: str,
        api: CachedInferenceAPI,
        limit: int = 100,
    ) -> Sequence[FinetuneConversation]:
        # Get the finetuning messages for the particular evaluation
        # TODO: MAKE SURE WE FINETUNE ON A DIFFERENT DATASET!!
        result = await finetune_samples_what_answer_without_bias(
            object_model=object_model,
            api=api,
            number_samples=limit,
        )
        print(f"Got {len(result)} finetuning samples for {cls.name()}")
        return result


class BiasDetectAddAreYouSure(OtherEvalRunner):
    @staticmethod
    async def run(
        eval_name: str,
        meta_model: str,
        object_model: str,
        api: CachedInferenceAPI,
        limit: int = 100,
        balance_data: bool = True,
    ) -> Sequence[OtherEvalCSVFormat]:
        """Ask the model if it would change its prediction if we ask 'ask you sure'. Y/N answers"""
        result: Slist[AreYouSureMetaResult] = await run_single_are_you_sure(
            object_model=object_model,
            meta_model=meta_model,
            api=api,
            number_samples=limit,
            balance_data=balance_data,
        )
        formatted = result.map(lambda x: x.to_other_eval_format(eval_name=eval_name))
        return formatted

    @classmethod
    async def get_finetuning(
        cls,
        object_model: str,
        api: CachedInferenceAPI,
        limit: int = 100,
    ) -> Sequence[FinetuneConversation]:
        # Get the finetuning messages for the particular evaluation
        result = await are_you_sure_finetune_samples(
            object_model=object_model,
            api=api,
            number_samples=limit,
        )
        print(f"Got {len(result)} finetuning samples for {cls.name()}")
        return result


class KwikWillYouBeCorrect(OtherEvalRunner):
    """Kwik stands for Know What I Know"""

    @staticmethod
    async def run(
        eval_name: str,
        meta_model: str,
        object_model: str,
        api: CachedInferenceAPI,
        limit: int = 100,
        balance_data: bool = True,
    ) -> Sequence[OtherEvalCSVFormat]:
        """Ask the model if it is going to get the correct answer. Y/N answers"""
        result = await run_single_ask_if_correct_answer(
            object_model=object_model,
            meta_model=meta_model,
            api=api,
            number_samples=limit,
            balance_data=balance_data,
        )
        formatted = result.map(lambda x: x.to_other_eval_format(eval_name=eval_name))

        return formatted

    @classmethod
    async def get_finetuning(
        cls,
        object_model: str,
        api: CachedInferenceAPI,
        limit: int = 100,
    ) -> Sequence[FinetuneConversation]:
        # Get the finetuning messages for the particular evaluation
        result = await kwik_finetune_samples(
            object_model=object_model,
            api=api,
            number_samples=limit,
        )
        print(f"Got {len(result)} finetuning samples for {cls.name()}")
        return result


class WillYouGiveDeontology(OtherEvalRunner):
    @staticmethod
    async def run(
        eval_name: str,
        meta_model: str,
        object_model: str,
        api: CachedInferenceAPI,
        limit: int = 100,
        balance_data: bool = True,
    ) -> Sequence[OtherEvalCSVFormat]:
        """Ask the model if it is going to get the correct answer. Y/N answers"""
        result = await run_single_ask_deontology(
            object_model=object_model,
            meta_model=meta_model,
            caller=RepoCompatCaller(api=api),
            number_samples=limit,
            balance_data=balance_data,
        )
        formatted = result.map(lambda x: x.to_other_eval_format(eval_name=eval_name))

        return formatted


ALL_EVAL_TYPES: Sequence[Type[OtherEvalRunner]] = [
    BiasDetectAddAreYouSure,
    BiasDetectAreYouAffected,
    BiasDetectWhatAnswerWithout,
    KwikWillYouBeCorrect,
]
ALL_EVAL_STR: Sequence[str] = [eval_name.name() for eval_name in ALL_EVAL_TYPES]
OTHER_EVAL_NAMES: dict[str, Type[OtherEvalRunner]] = {eval_name.name(): eval_name for eval_name in ALL_EVAL_TYPES}
assert len(ALL_EVAL_TYPES) == len(
    OTHER_EVAL_NAMES
), f"Got {len(ALL_EVAL_TYPES)} eval types but {len(OTHER_EVAL_NAMES)} names"

runner_to_eval_name = {v: k for k, v in OTHER_EVAL_NAMES.items()}
assert len(OTHER_EVAL_NAMES) == len(
    runner_to_eval_name
), "The mapping is not bijective, maybe you have duplicate keys / values?"


async def run_from_commands(
    evals_to_run: Sequence[Type[OtherEvalRunner]],
    object_and_meta: Sequence[tuple[str, str]],
    limit: int,
    api: CachedInferenceAPI,
    balance_data: bool = True,
) -> Slist[OtherEvalCSVFormat]:
    """Run the appropriate evaluation based on the dictionary"""
    # coorountines_to_run: Slist[Awaitable[Sequence[OtherEvalCSVFormat]]] = Slist()
    gathered = Slist()
    for object_model, meta_model in object_and_meta:
        for runner in evals_to_run:
            eval_name = runner.name()
            # coorountines_to_run.append(runner.run(meta_model=meta_model, object_model=object_model, cache_path=cache_path, limit=limit))
            result = await runner.run(
                eval_name=eval_name,
                meta_model=meta_model,
                object_model=object_model,
                limit=limit,
                api=api,
                balance_data=balance_data,
            )
            gathered.append(result)

    # todo: do we really want to run all of these at the same time? lol
    # gathered = await coorountines_to_run.gather()
    # we get a list of lists, so we flatten it
    flattened: Slist[OtherEvalCSVFormat] = gathered.flatten_list()
    return flattened


def eval_list_to_runner(eval_list: Sequence[str]) -> Sequence[Type[OtherEvalRunner]]:
    runners = []
    for eval_str in eval_list:
        maybe_eval_runner = OTHER_EVAL_NAMES.get(eval_str, None)
        if maybe_eval_runner is not None:
            runners.append(maybe_eval_runner)
        else:
            raise ValueError(
                f"Could not find runner for {eval_str}, is it present in ALL_EVAL_TYPES?. Available evals: {ALL_EVAL_STR}"
            )

    return runners


def run_sweep_over_other_evals(
    object_and_meta_configs: Sequence[tuple[str, str]] = [("gpt-3.5-turbo", "gpt-3.5-turbo")],
    eval_list: Sequence[Type[OtherEvalRunner]] = [BiasDetectAddAreYouSure],
    limit: int = 100,
    study_folder: str | Path = "exp/other_evals",
    cache_path: str | Path = "exp/other_evals/cache",
    show_plot: bool = False,
) -> None:
    object_and_meta_ids = [
        (read_model_id_from_model_config(object_config), read_model_id_from_model_config(meta_config))
        for object_config, meta_config in object_and_meta_configs
    ]
    return run_sweep_over_other_evals_ids(
        object_and_meta_ids=object_and_meta_ids,
        eval_list=eval_list,
        limit=limit,
        study_folder=study_folder,
        cache_path=cache_path,
        show_plot=show_plot,
    )


def run_sweep_over_other_evals_ids(
    object_and_meta_ids: Sequence[tuple[str, str]] = [("gpt-3.5-turbo", "gpt-3.5-turbo")],
    eval_list: Sequence[Type[OtherEvalRunner]] = [BiasDetectAddAreYouSure],
    limit: int = 100,
    study_folder: str | Path = "exp/other_evals",
    cache_path: str | Path = "exp/other_evals/cache",
    show_plot: bool = False,
) -> None:
    """
    object_and_meta: a list of tuples of object and meta models
    eval_list: a list of evaluation names. See the keys in OTHER_EVAL_NAMES.
    e.g. ["BiasDetectAddAreYouSure", "BiasDetectAreYouAffected", "BiasDetectWhatAnswerWithout", "KwikWillYouBeCorrect"]
    limit: the number of samples to run
    study_folder: the folder where the results will be saved
    """
    setup_environment()
    # Entry point for sweeping
    # the sweep ain't a async function so we use asyncio.run
    api = InferenceAPI(anthropic_num_threads=20)
    inference_api = CachedInferenceAPI(api=api, cache_path=cache_path)

    all_results = asyncio.run(
        run_from_commands(
            evals_to_run=eval_list,
            object_and_meta=object_and_meta_ids,
            limit=limit,
            api=inference_api,
        )
    )
    grouped_by_eval = all_results.group_by(lambda x: x.eval_name)
    for eval_name, results_list in grouped_by_eval:
        df = pd.DataFrame(results_list.map(lambda x: x.model_dump()))
        if show_plot:
            plot_heatmap_with_ci(
                data=df,
                value_col="meta_predicted_correctly",
                object_col="object_model",
                meta_col="meta_model",
                title=f"{eval_name} Percentage of Meta Response Predicted Correctly with 95% CI",
            )
        result_path = Path(study_folder) / f"{eval_name}_results.csv"
        # make sure the folder exists
        result_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving {eval_name} to {result_path}")
        df.to_csv(result_path, index=False)


def test_main():
    # What evals to run?
    # eval_list = [WillYouGiveDeontology]
    eval_list = [BiasDetectAreYouAffected, BiasDetectWhatAnswerWithout]
    print(f"Running evals: {[e.name() for e in eval_list]}")
    limit = 1000
    # What models to run?
    # prefinetune_model: str = "gpt-3.5-turbo-1106"
    # postfinetune_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
    # object_model: str = "gpt-3.5-turbo-0125"
    # meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z"

    # half held out, doesn't work
    # object_model: str = "gpt-3.5-turbo-0125"
    # meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z"

    # half held out, 2x more samples
    object_model = "gpt-3.5-turbo-0125"
    meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eMKxx3y"

    # train on all... Train does help!
    # object_model = "gpt-3.5-turbo-0125"
    # meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS"
    # object_model = "gpt-4o-2024-05-13"
    # meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9danhPzM"
    models = Slist(
        [
            object_model,
            meta_model,
            # "gpt-3.5-turbo",
            # "gpt-3.5-turbo-0125",
            # "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:nommlu:9YISrgjH", # non mmlu sweep
            # "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:sweep:9WBVcb4d",  # mmlu sweep
            # "gpt-3.5-turbo-1106",
            # "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9XGmIcNV" # train on felix + james'
            # "gpt-4-0613",
            # "ft:gpt-4-0613:dcevals-kokotajlo:sweep:9RSQ9BDP" # gpt-4 on gpt -4
            # "gpt-3.5-turbo-1106",
            # "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:sweep:9WBVcb4d"
            # "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9YHdMAcl", # leave out are you sure
            # "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9WPLCVRV",  # train on claude
            # "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:baliemay20:9WAurjLN",  # baseline scrambled
            # "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9WPLCVRV",  # train on claude
            # "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo:baliemay20:9WAurjLN", # baseline scrambled
            # "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9WOKeIsb", # 12,000 samples gpt-3.5
            # "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9WE1NjvJ",  # gpt-3.5 on gpt-3.5, on arc other evals, 3600 samples
            # "claude-3-sonnet-20240229",
            # "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9GYUm36T" # all response properites
            # "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo::9Lrb314n",  # ask if affected
            # "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo::9PutAYsj",
            # "gpt-4o",
        ]
    )
    # We want to run all the combinations of the models
    object_and_meta_models: Slist[tuple[str, str]] = models.product(models)
    study_folder = EXP_DIR / "other_evals"

    run_sweep_over_other_evals_ids(
        eval_list=eval_list,
        object_and_meta_ids=object_and_meta_models,
        limit=limit,
        study_folder=study_folder,
        show_plot=True,
        cache_path="/Users/jameschua/ml/introspection_self_prediction_astra/exp/other_evals/cache",
    )


def test_cross_train():
    # What evals to run?
    # eval_list = [WillYouGiveDeontology]
    eval_list = [BiasDetectAddAreYouSure, BiasDetectWhatAnswerWithout, BiasDetectAreYouAffected, KwikWillYouBeCorrect]
    print(f"Running evals: {[e.name() for e in eval_list]}")
    limit = 2500
    # What models to run?
    # prefinetune_model: str = "gpt-3.5-turbo-1106"
    # postfinetune_model: str = "ft:gpt-3.5-turbo-1106:dcevals-kokotajlo:sweep:9R9Lqsm2"
    # object_model: str = "gpt-3.5-turbo-0125"
    # meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z"

    # half held out, doesn't work
    # object_model: str = "gpt-3.5-turbo-0125"
    # meta_model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eEh2T6z"

    # half held out, 2x more samples
    # first = ("ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eMKxx3y", "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eMKxx3y")
    # second = ("ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9eMKxx3y", "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:gpt4o-on-ftedgpt35:9g5qGBji")

    # leave out only are you sure
    object_model = "gpt-4o-2024-05-13"
    meta_model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9danhPzM"
    first = (object_model, meta_model)
    second = (meta_model, meta_model)

    # We want to run all the combinations of the models
    object_and_meta_models = [first, second]
    study_folder = EXP_DIR / "other_evals"

    run_sweep_over_other_evals_ids(
        eval_list=eval_list,
        object_and_meta_ids=object_and_meta_models,
        limit=limit,
        study_folder=study_folder,
        show_plot=True,
        cache_path="exp/cached_dir",
    )


if __name__ == "__main__":
    test_cross_train()
