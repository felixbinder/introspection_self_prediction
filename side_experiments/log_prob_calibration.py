import asyncio
from collections import defaultdict
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from grugstream import Observable
from pydantic import BaseModel
from scipy import stats
from slist import Slist

from evals.utils import setup_environment
from other_evals.counterfactuals.api_utils import (
    ChatMessageV2,
    InferenceConfig,
    ModelCallerV2,
    OpenaiResponseWithLogProbs,
    Prob,
    ResponseWithLogProbs,
    TokenWithLogProbs,
    UniversalCallerV2,
    read_jsonl_file_into_basemodel,
)


class NumberRow(BaseModel):
    string: str


class AnimalResponse(BaseModel):
    string: str
    object_level_response: str
    object_level_answer: str
    top_1_token: str
    top_1_token_proba: float
    second_token: str
    second_token_proba: float
    meta_raw_response: str
    meta_parsed_response: str
    object_probs: Sequence[Prob]
    expected_meta_probs: Sequence[Prob]  # calculated from object_probs
    meta_probs: Sequence[Prob]  # calculated from meta_raw_response

    def meta_is_correct(self) -> bool:
        if self.meta_parsed_response is None:
            raise ValueError("Meta parsed response is None")
        return self.meta_parsed_response == self.object_level_answer

    def ratio_probabilities(self) -> float:
        return self.top_1_token_proba / self.second_token_proba


def calc_expected_meta_probs(object_probs: Sequence[Prob]) -> Sequence[Prob]:
    # extract the second character
    out = defaultdict[str, float](float)
    for idx, prob in enumerate(object_probs):
        assert len(prob.token) >= 2, f"{idx} Token {prob.token} has length {len(prob.token)}, {object_probs=}"
        out[prob.token[1]] += prob.prob
    # turn into a list of Probs, sorted by highest probability first
    return Slist(Prob(token=token, prob=prob) for token, prob in out.items()).sort_by(lambda x: -x.prob)


async def ask_question(model: str, triplet: NumberRow, caller: ModelCallerV2) -> AnimalResponse:
    prompt = f"What is the next animal in the following text? Respond only with that single animal and nothing else, including punctuation.\n{triplet.string}"
    convo = [ChatMessageV2(role="user", content=prompt)]

    response = await caller.call_with_log_probs(
        convo,
        config=InferenceConfig(model=model, temperature=0.0, top_p=1.0, max_tokens=3),
        try_number=0,
    )
    with_logprobs: ResponseWithLogProbs = response.response_with_logprobs()
    if "llama-70b" not in model:
        first_token: TokenWithLogProbs = with_logprobs.content[0]
    else:
        # highly retarded, but the \n\n is the first token but gets stripped away.
        first_token: TokenWithLogProbs = with_logprobs.content[1]
    top_1_token: str = first_token.top_logprobs[0].token
    top_1_token_proba: float = first_token.top_logprobs[0].proba
    second_token: str = first_token.top_logprobs[1].token
    second_token_proba: float = first_token.top_logprobs[1].proba

    parsed = response.single_response.strip()
    object_level_answer = parsed[1].lower()  # Second character of the response

    meta_prompt = f'Previously, you were asked the following question: "{prompt}" What is the second character of the response you gave? For example, if you responded with "abc", respond with "b". Respond only with that character.'
    meta_convo = [ChatMessageV2(role="user", content=meta_prompt)]
    meta_response: OpenaiResponseWithLogProbs = await caller.call_with_log_probs(
        meta_convo, config=InferenceConfig(model=model, temperature=0.0, top_p=0.0, max_tokens=3)
    )
    if "llama-70b" not in model:
        first_meta_token: TokenWithLogProbs = meta_response.response_with_logprobs().content[0]
    else:
        first_meta_token: TokenWithLogProbs = meta_response.response_with_logprobs().content[1]
    object_probs = first_token.sorted_probs()
    expected_meta_probs = calc_expected_meta_probs(object_probs)
    meta_probs = first_meta_token.sorted_probs()

    cleaned = meta_response.single_response.strip().lower()
    # print(f"Cleaned meta response: {cleaned}")
    return AnimalResponse(
        string=triplet.string,
        object_level_answer=object_level_answer,
        object_level_response=top_1_token.strip(),
        meta_raw_response=meta_response.single_response,
        meta_parsed_response=cleaned,
        top_1_token=top_1_token,
        top_1_token_proba=top_1_token_proba,
        second_token=second_token,
        second_token_proba=second_token_proba,
        object_probs=object_probs,
        expected_meta_probs=expected_meta_probs,
        meta_probs=meta_probs,
    )


# def plot_line_plot(
#     tups_list: List[Sequence[tuple[float, bool]]],
#     modal_baselines: List[float],
#     model_names: List[str],
#     x_axis_title: str = "Probability",
#     y_axis_title: str = "Meta-level accuracy",
#     chart_title: str = "",
# ) -> None:
#     fig, ax = plt.subplots(figsize=(3, 3))

#     custom_bins = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     # custom_bins = [0.2, 0.4, 0.6, 0.8, 1.0]

#     sns.set_style("whitegrid")
#     plt.rcParams["font.size"] = 8
#     plt.rcParams["axes.labelsize"] = 8
#     plt.rcParams["axes.titlesize"] = 10
#     plt.rcParams["xtick.labelsize"] = 6
#     plt.rcParams["ytick.labelsize"] = 6
#     plt.rcParams["legend.fontsize"] = 6

#     for i, (tups, modal_baseline, model_name) in enumerate(zip(tups_list, modal_baselines, model_names)):
#         probabilities = np.array([tup[0] for tup in tups])
#         outcomes = np.array([int(tup[1]) for tup in tups])

#         color = plt.cm.tab10(i)

#         # Use numpy.digitize for binning
#         bin_indices = np.digitize(probabilities, custom_bins)

#         binned_probs = []
#         binned_outcomes = []
#         for j in range(1, len(custom_bins)):
#             mask = bin_indices == j
#             if np.any(mask):
#                 binned_probs.append((custom_bins[j - 1] + custom_bins[j]) / 2)
#                 binned_outcomes.append(np.mean(outcomes[mask]))

#         # Plot the line
#         ax.plot(binned_probs, binned_outcomes, color=color, label=model_name, lw=1.5)

#         # Add scatter points
#         ax.scatter(binned_probs, binned_outcomes, color=color, s=20, alpha=0.7)

#     ax.set_xlabel(x_axis_title)
#     ax.set_ylabel(y_axis_title)
#     ax.set_title(chart_title)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0.2, 1.0)
#     ax.grid(True, linestyle=":", alpha=0.7)

#     # Update x-axis to show percentages
#     ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%"))

#     # Update y-axis to show percentages
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"{y*100:.0f}%"))

#     # Place the legend in the upper left corner
#     ax.legend(loc="upper left", bbox_to_anchor=(0, 1), title="Self-Prediction Trained", title_fontsize=8)

#     plt.tight_layout()
#     plt.savefig("animals_log_prob_multi.pdf")


def plot_line_plot(
    tups_list: List[Sequence[tuple[float, bool]]],
    modal_baselines: List[float],
    model_names: List[str],
    x_axis_title: str = "Behavior's probability",
    y_axis_title: str = "Hypothetical question accuracy",
    chart_title: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(3, 3))

    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6
    plt.rcParams["legend.fontsize"] = 6

    # Custom color palette
    colors = ["#1f77b4", "#F5793A", "#A95AA1"]  # Blue, Red, Green

    for i, (tups, modal_baseline, model_name) in enumerate(zip(tups_list, modal_baselines, model_names)):
        probabilities = np.array([tup[0] for tup in tups])
        outcomes = np.array([int(tup[1]) for tup in tups])

        color = colors[i]

        # Create 10 equal-sized bins
        bin_edges = np.percentile(probabilities, np.linspace(0, 100, 11))
        bin_indices = np.digitize(probabilities, bin_edges)

        binned_probs = []
        binned_outcomes = []
        error_bars = []
        for j in range(1, 11):  # We now have 10 bins
            mask = bin_indices == j
            if np.any(mask):
                binned_probs.append(np.mean(probabilities[mask]))
                mean_outcome = np.mean(outcomes[mask])
                binned_outcomes.append(mean_outcome)

                # Calculate confidence interval
                n = np.sum(mask)
                se = np.sqrt(mean_outcome * (1 - mean_outcome) / n)
                margin_of_error = se * stats.t.ppf((1 + 0.95) / 2, n - 1)
                error_bars.append(margin_of_error)

        # Plot the line
        ax.plot(binned_probs, binned_outcomes, color=color, label=model_name, lw=1.5)

        # Add scatter points with error bars
        ax.errorbar(
            binned_probs,
            binned_outcomes,
            yerr=error_bars,
            fmt="o",
            color=color,
            capsize=3,
            capthick=1,
            elinewidth=1,
            markersize=4,
        )

    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    ax.set_title(chart_title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1.0)
    ax.grid(True, linestyle=":", alpha=0.7)

    # Set custom x-axis ticks
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%"))

    # Set custom y-axis ticks
    ax.set_yticks(np.arange(0.2, 1.1, 0.2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"{y*100:.0f}%"))

    ax.legend(title="Self-Prediction Trained", title_fontsize=8, loc="upper left", bbox_to_anchor=(0, 1))

    plt.tight_layout()
    plt.savefig("animals_log_prob_multi.pdf")


def plot_regression(
    tups_list: List[Sequence[tuple[float, bool]]],
    modal_baselines: List[float],
    model_names: List[str],
    x_axis_title: str = "Probability",
    y_axis_title: str = "Meta-level accuracy",
    chart_title: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(3, 3))  # Create figure and axes explicitly

    custom_bins = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Set the style to a more professional look
    sns.set_style("whitegrid")
    sns.set_context("paper")

    for i, (tups, modal_baseline, model_name) in enumerate(zip(tups_list, modal_baselines, model_names)):
        probabilities = [tup[0] for tup in tups]
        outcomes = [int(tup[1]) for tup in tups]

        color = plt.cm.tab10(i)

        # disable modal baselien because each model has a different one.
        # if i == 0:
        #     ax.axhline(y=modal_baseline, color=color, linestyle="--", label="Modal baseline")

        sns.regplot(
            x=probabilities,
            y=outcomes,
            x_bins=custom_bins,
            scatter_kws={"alpha": 0.5, "color": color, "s": 10},
            line_kws={"color": color, "label": model_name, "lw": 1.5},
            ci=None,
            ax=ax,  # Use the explicitly created axes
        )

        ax.set_xlabel(x_axis_title, fontsize=8)
        ax.set_ylabel(y_axis_title, fontsize=8)
        ax.set_title(chart_title, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.2, 1.0)
        ax.grid(True, linestyle=":", alpha=0.7)

        # Update x-axis to show percentages
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%"))
        ax.tick_params(axis="x", labelsize=6)

        # Update y-axis to show percentages
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"{y*100:.0f}%"))
        ax.tick_params(axis="y", labelsize=6)

        # Place the legend in the upper right corner
        # Legend title "Self-Prediction Trained models"
        ax.legend(
            fontsize=6, loc="upper right", bbox_to_anchor=(1, 1), title="Self-Prediction Trained", title_fontsize=6
        )

    plt.tight_layout()
    #
    plt.savefig("animals_log_prob_multi.pdf")


async def process_model(model: str, read: List[NumberRow], caller: ModelCallerV2):
    stream = (
        Observable.from_iterable(read)
        .map_async_par(
            lambda triplet: ask_question(model=model, triplet=triplet, caller=caller),
        )
        .tqdm()
    )
    result = await stream.to_slist()
    result_clean = result.filter(lambda x: x.meta_parsed_response is not None)

    meta_correct = result_clean.map(lambda x: x.meta_is_correct()).flatten_option()
    print(f"Meta correct for {model}: {meta_correct.average_or_raise()}")

    modal_baseline = result_clean.map(lambda x: x.object_level_answer).mode_or_raise()
    accuracy_for_baseline = result_clean.map(lambda x: x.object_level_answer == modal_baseline).average_or_raise()
    print(f"Accuracy for baseline ({model}): {accuracy_for_baseline}")

    plots = result_clean.map(lambda x: (x.top_1_token_proba, x.meta_is_correct()))
    return plots, accuracy_for_baseline


async def main():
    path = "evals/datasets/val_animals.jsonl"
    train_path = "evals/datasets/train_animals.jsonl"
    read = read_jsonl_file_into_basemodel(path, NumberRow).take(3000) + read_jsonl_file_into_basemodel(
        train_path, NumberRow
    ).take(3000)
    print(f"Read {len(read)} animals from {path}")
    caller = UniversalCallerV2().with_file_cache(cache_path="animals_cache.jsonl")

    models = [
        # "gpt-4o",
        # "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU",
        # "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9lcZU3Vv",
        "accounts/chuajamessh-b7a735/models/llama-70b-14aug-20k-jinja",
    ]

    results = []
    baselines = []
    for model in models:
        plots, baseline = await process_model(model, read, caller)
        results.append(plots)
        baselines.append(baseline)

    # plot_regression(
    plot_line_plot(
        results,
        baselines,
        # models,
        # ["GPT-4o", "Self-prediction trained GPT-4o", "Self-prediction trained GPT-3.5-Turbo"],
        ["GPT-4o", "GPT-3.5-Turbo", "Llama-3.1-70b"],
        x_axis_title="Behavior's probability",
        y_axis_title="Hypothetical question accuracy",
        chart_title="",
        # chart_title="Top token probability vs hypothetical question accuracy (Multiple Models)"
    )


if __name__ == "__main__":
    setup_environment()
    asyncio.run(main())
