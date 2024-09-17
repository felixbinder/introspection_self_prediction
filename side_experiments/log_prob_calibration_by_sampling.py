import asyncio
import token
from collections import defaultdict
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from grugstream import Observable
from pydantic import BaseModel
from scipy import stats
from slist import Slist
from traitlets import default

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


def calc_expected_meta_probs(object_probs: Sequence[Prob]) -> Sequence[Prob]:
    # extract the second character
    out = defaultdict[str, float](float)
    for idx, prob in enumerate(object_probs):
        assert len(prob.token) >= 2, f"{idx} Token {prob.token} has length {len(prob.token)}, {object_probs=}"
        out[prob.token[1]] += prob.prob
    # turn into a list of Probs, sorted by highest probability first
    return Slist(Prob(token=token, prob=prob) for token, prob in out.items()).sort_by(lambda x: -x.prob)


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


class SampledAnimalResponse(BaseModel):
    string: str
    object_level_response: str
    object_level_answer: str
    top_1_token: str
    top_1_token_proba: float
    # second_token: str
    # second_token_proba: float
    # meta_raw_response: str
    meta_parsed_response: str
    object_probs: Sequence[Prob]
    expected_object_probs: Sequence[Prob]  # calculated from object_probs
    meta_probs: Sequence[Prob]  # calculated from meta_raw_response

    @staticmethod
    def from_animal_responses(responses: Sequence[AnimalResponse]) -> "SampledAnimalResponse":
        modal_object_level_answer = Slist(responses).map(lambda x: x.object_level_answer).mode_or_raise()
        # proba of the mode
        proba = Slist(responses).map(lambda x: x.object_level_answer).count(modal_object_level_answer) / len(responses)

        modal_meta_parsed_response = Slist(responses).map(lambda x: x.meta_parsed_response).mode_or_raise()
        # groupby sum
        object_probs = (
            Slist(responses)
            .map(lambda x: x.object_level_response)
            .group_by(lambda x: x)
            .map_2(
                # key: token, value: list of responses
                lambda key, value: Prob(token=key, prob=len(value) / len(responses))
            )
        )
        expected_meta_probs = calc_expected_meta_probs(object_probs)
        meta_probs = (
            Slist(responses)
            .map(lambda x: x.meta_parsed_response)
            .group_by(lambda x: x)
            .map_2(lambda key, value: Prob(token=key, prob=len(value) / len(responses)))
        )

        return SampledAnimalResponse(
            string=responses[0].string,
            object_level_response=modal_object_level_answer,
            object_level_answer=modal_object_level_answer,
            top_1_token=modal_object_level_answer,
            top_1_token_proba=proba,
            meta_parsed_response=modal_meta_parsed_response,
            object_probs=object_probs,
            expected_object_probs=expected_meta_probs,
            meta_probs=meta_probs,
        )

    def first_meta_proba(self) -> float:
        return Slist(self.meta_probs).sort_by(lambda x: -x.prob).first_or_raise().prob

    def get_meta_proba_for_behaviour(self, behaviour: str) -> float:
        behavior: Prob | None = Slist(self.meta_probs).filter(lambda x: x.token == behaviour).first_option
        return behavior.prob if behavior is not None else 0.0

    def first_expected_object_proba(self) -> float:
        return Slist(self.expected_object_probs).sort_by(lambda x: -x.prob).first_or_raise().prob

    def second_expected_object_proba(self) -> float:
        return Slist(self.expected_object_probs).sort_by(lambda x: -x.prob).map(lambda x: x.prob).get(1, or_else=0.0)

    def third_expected_object_proba(self) -> float:
        return Slist(self.expected_object_probs).sort_by(lambda x: -x.prob).map(lambda x: x.prob).get(2, or_else=0.0)

    def first_expected_with_meta(self) -> tuple[float, float]:
        return self.first_expected_object_proba(), self.get_meta_proba_for_behaviour(self.object_level_answer)

    def second_expected_with_meta(self) -> tuple[float, float] | None:
        # second most likely object-level behaviour
        second_expected: str | None = (
            Slist(self.expected_object_probs).sort_by(lambda x: -x.prob).map(lambda x: x.token).get(1, or_else=None)
        )
        if second_expected is None:
            return None
        return self.second_expected_object_proba(), self.get_meta_proba_for_behaviour(second_expected)

    def third_expected_with_meta(self) -> tuple[float, float] | None:
        # third most likely object-level behaviour
        third_expected: str | None = (
            Slist(self.expected_object_probs).sort_by(lambda x: -x.prob).map(lambda x: x.token).get(2, or_else=None)
        )
        if third_expected is None:
            return None
        return self.third_expected_object_proba(), self.get_meta_proba_for_behaviour(third_expected)

    def meta_is_correct(self) -> bool:
        if self.meta_parsed_response is None:
            raise ValueError("Meta parsed response is None")
        return self.meta_parsed_response == self.object_level_answer

    def ratio_probabilities(self) -> float:
        return self.top_1_token_proba / self.second_token_proba


async def ask_question(
    model: str, triplet: NumberRow, caller: ModelCallerV2, try_number: int, cross_prediction_model: str | None = None
) -> AnimalResponse | None:
    prompt = f"What is the next animal in the following text? Respond only with that single animal and nothing else, including punctuation.\n{triplet.string}"
    convo = [ChatMessageV2(role="user", content=prompt)]

    response = await caller.call_with_log_probs(
        convo,
        config=InferenceConfig(model=model, temperature=1.0, top_p=1.0, max_tokens=3),
        try_number=try_number,
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
    if len(parsed) < 2:
        return None
    object_level_answer = parsed[1].lower()  # Second character of the response

    meta_prompt = f'Previously, you were asked the following question: "{prompt}" What is the second character of the response you gave? For example, if you responded with "abc", respond with "b". Respond only with that character.'
    meta_convo = [ChatMessageV2(role="user", content=meta_prompt)]
    meta_model = cross_prediction_model if cross_prediction_model is not None else model
    meta_response: OpenaiResponseWithLogProbs = await caller.call_with_log_probs(
        meta_convo,
        config=InferenceConfig(model=meta_model, temperature=1.0, top_p=1.0, max_tokens=3),
        try_number=try_number,
    )
    # if "llama-70b" not in model:
    #     first_meta_token: TokenWithLogProbs = meta_response.response_with_logprobs().content[0]
    # else:
    #     first_meta_token: TokenWithLogProbs = meta_response.response_with_logprobs().content[1]
    # object_probs = first_token.sorted_probs()
    # expected_meta_probs = calc_expected_meta_probs(object_probs)
    # meta_probs = first_meta_token.sorted_probs()

    cleaned = meta_response.single_response.strip().lower()
    # print(f"Cleaned meta response: {cleaned}")
    return AnimalResponse(
        string=triplet.string,
        object_level_answer=object_level_answer,
        object_level_response=parsed,
        meta_raw_response=meta_response.single_response,
        meta_parsed_response=cleaned,
        top_1_token=top_1_token,
        top_1_token_proba=top_1_token_proba,
        second_token=second_token,
        second_token_proba=second_token_proba,
        # object_probs=object_probs,
        # expected_meta_probs=expected_meta_probs,
        # meta_probs=meta_probs,
    )


async def ask_question_sampling(
    model: str,
    triplet: NumberRow,
    caller: ModelCallerV2,
    n_samples: int = 10,
    cross_prediction_model: str | None = None,
) -> SampledAnimalResponse:
    repeats: Slist[int] = Slist(range(n_samples))
    responses: Slist[AnimalResponse | None] = await repeats.par_map_async(
        lambda repeat: ask_question(model, triplet, caller, repeat, cross_prediction_model=cross_prediction_model)
    )
    flattend = responses.flatten_option()
    return SampledAnimalResponse.from_animal_responses(flattend)


from scipy.stats import gaussian_kde, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def plot_scatter_plot(
    data: Sequence[tuple[float, float]],
    model_name: str,
    filename: str = "scatter_plot.pdf",
    x_axis_title: str = "Expected Meta Probability",
    y_axis_title: str = "Predicted Meta Probability",
    chart_title: str = "",
) -> None:
    """
    Plots a scatter plot of expected vs predicted meta probabilities with enhanced density visualization
    and statistical annotations, including MSE.

    Args:
        data (Sequence[tuple[float, float]]): A list of tuples where each tuple contains
                                             (expected_meta_proba, predicted_meta_proba).
        model_name (str): The name of the model for labeling purposes.
        filename (str): The filename to save the plot. Defaults to "scatter_plot.pdf".
        x_axis_title (str): Label for the x-axis. Defaults to "Expected Meta Probability".
        y_axis_title (str): Label for the y-axis. Defaults to "Predicted Meta Probability".
        chart_title (str): The title of the chart. Defaults to an empty string.
    """
    # Convert data to numpy arrays
    expected_probs = np.array([x for x, y in data])
    predicted_probs = np.array([y for x, y in data])

    # Calculate the point density using Gaussian KDE
    xy = np.vstack([expected_probs, predicted_probs])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that dense areas are plotted on top
    idx = z.argsort()
    expected_probs, predicted_probs, z = expected_probs[idx], predicted_probs[idx], z[idx]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 6))  # Increased figure size for better visibility

    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10

    # Create the scatter plot with density-based coloring
    scatter = ax.scatter(
        expected_probs + np.random.normal(0, 0.005, size=expected_probs.shape),  # Adding jitter
        predicted_probs + np.random.normal(0, 0.005, size=predicted_probs.shape),  # Adding jitter
        c=z,
        cmap="viridis",  # Color map for density
        s=30,  # Slightly larger marker size
        alpha=0.7,  # Adjusted transparency
        edgecolor="w",  # Marker edge color
        linewidth=0.5,  # Marker edge width
        label=model_name,
    )

    # Optionally, overlay a hexbin plot for alternative density visualization
    # hb = ax.hexbin(expected_probs, predicted_probs, gridsize=50, cmap='Blues', alpha=0.4)
    # plt.colorbar(hb, ax=ax, label='Hexbin Density')

    # Add a color bar to represent density
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Density")

    # Add a y = x reference line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

    # Calculate statistics
    correlation, p_value = pearsonr(expected_probs, predicted_probs)
    mse = mean_squared_error(expected_probs * 100, predicted_probs * 100)
    mad = mean_absolute_error(expected_probs * 100, predicted_probs * 100)

    # Display the Pearson correlation coefficient, p-value, and MSE
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {correlation:.2f}\nP-value = {p_value:.2e}\nMAD (no binning)= {mad:.2f}\nMSE = {mse:.2f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5),
    )

    # Optionally, perform and plot a linear regression with confidence interval
    sns.regplot(
        x=expected_probs,
        y=predicted_probs,
        ax=ax,
        scatter=False,
        color="red",
        line_kws={"linewidth": 1.5},
        ci=95,  # 95% confidence interval
    )

    # Set labels and title
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    if chart_title:
        ax.set_title(chart_title)

    # Set limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Customize ticks to show percentages
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_xticklabels([f"{int(x*100)}%" for x in np.arange(0, 1.1, 0.2)])
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([f"{int(y*100)}%" for y in np.arange(0, 1.1, 0.2)])

    # Add legend
    ax.legend(title="Model", loc="lower right")

    # Tight layout for better spacing
    plt.tight_layout()

    plt.savefig(filename)
    # Display the plot
    plt.show()

    # Optionally, save the plot
    plt.close()


def equal_sized_binning(data, num_bins=10):
    expected_probs = np.array([x for x, y in data])
    predicted_probs = np.array([y for x, y in data])

    # Sort the data by expected probabilities
    sorted_indices = np.argsort(expected_probs)
    expected_probs_sorted = expected_probs[sorted_indices]
    predicted_probs_sorted = predicted_probs[sorted_indices]

    # Calculate the number of data points per bin
    n = len(data)
    bin_size = n // num_bins

    bin_means_x = []
    bin_means_y = []

    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < num_bins - 1 else n

        bin_means_x.append(np.mean(expected_probs_sorted[start_idx:end_idx]))
        bin_means_y.append(np.mean(predicted_probs_sorted[start_idx:end_idx]))

    return bin_means_x, bin_means_y


def plot_calibration_curve(
    data: Sequence[tuple[float, float]],
    model_name: str,
    filename: str = "calibration_curve.pdf",
    x_axis_title: str = "Model Probability",
    y_axis_title: str = "Model accuracy",
    chart_title: str = "",
    num_bins: int = 10,
) -> None:
    # Bin the data
    bin_means_x, bin_means_y = equal_sized_binning(data, num_bins)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10

    # Create the scatter plot for binned data
    ax.scatter(bin_means_x, bin_means_y, s=50, color="blue", label=model_name)

    # Add a y = x reference line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

    # Calculate statistics
    expected_probs = np.array([x for x, y in data])
    predicted_probs = np.array([y for x, y in data])
    correlation, p_value = pearsonr(expected_probs, predicted_probs)
    # mse = mean_squared_error(expected_probs * 100, predicted_probs * 100)
    # use bins to MAD like stephanie's
    mad = mean_absolute_error([x * 100 for x in bin_means_x], [y * 100 for y in bin_means_y])

    # Display the Pearson correlation coefficient, p-value, and MSE
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {correlation:.2f}\nP-value = {p_value:.2e}\nMAD (binned) = {mad:.2f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5),
    )

    # Set labels and title
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    if chart_title:
        ax.set_title(chart_title)

    # Set limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Customize ticks to show percentages
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_xticklabels([f"{int(x*100)}%" for x in np.arange(0, 1.1, 0.2)])
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([f"{int(y*100)}%" for y in np.arange(0, 1.1, 0.2)])

    # Add legend
    ax.legend(title="Model", loc="lower right")

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the plot
    plt.savefig(filename)

    # Display the plot
    plt.show()


# Removed the commented out old plot_scatter_plot function for clarity


async def process_model_scatter(
    model: str, read: List[NumberRow], caller: ModelCallerV2, cross_prediction_model: str | None = None
) -> Slist[tuple[float, float]]:
    stream = (
        Observable.from_iterable(read)
        .map_async_par(
            lambda triplet: ask_question_sampling(
                model=model, triplet=triplet, caller=caller, n_samples=20, cross_prediction_model=cross_prediction_model
            ),
            max_par=5,
        )
        .tqdm()
    )
    result = await stream.to_slist()
    result_clean = result.filter(lambda x: x.meta_parsed_response is not None)

    # expected meta proba vs meta proba
    expected_meta_proba: Slist[tuple[float, float]] = result_clean.map(lambda x: x.first_expected_with_meta())

    # second
    expected_meta_proba_second: Slist[tuple[float, float]] = result_clean.map(
        lambda x: x.second_expected_with_meta()
    ).flatten_option()
    expected_meta_proba_third: Slist[tuple[float, float]] = result_clean.map(
        lambda x: x.third_expected_with_meta()
    ).flatten_option()

    # plot_scatter_plot
    plot_scatter_plot(
        expected_meta_proba,
        model_name="Top object-level behavior: Llama-3.1-70b",
        x_axis_title="Object-level Probability",
        y_axis_title="Predicted Meta Probability",
        chart_title="Top Object-Level Behavior Calibration",
        filename="top_scatter_plot.pdf",
    )

    plot_scatter_plot(
        expected_meta_proba_second,
        model_name="Second highest object-level behavior: Llama-3.1-70b",
        x_axis_title="Object-level Probability",
        y_axis_title="Predicted Meta Probability",
        chart_title="Second Object-Level Behavior Calibration",
        filename="second_scatter_plot.pdf",
    )

    # Additionally, plot calibration curves with the same data
    plot_calibration_curve(
        expected_meta_proba,
        model_name="Top object-level behavior: Llama-3.1-70b",
        x_axis_title="Object-level Probability",
        y_axis_title="Predicted Meta Probability",
        chart_title="Top Object-Level Behavior Calibration",
        filename="top_calibration_curve.pdf",
    )

    plot_calibration_curve(
        expected_meta_proba_second,
        model_name="Second highest object-level behavior: Llama-3.1-70b",
        x_axis_title="Object-level Probability",
        y_axis_title="Predicted Meta Probability",
        chart_title="Second Object-Level Behavior Calibration",
        filename="second_calibration_curve.pdf",
    )

    plot_calibration_curve(
        expected_meta_proba_third,
        model_name="Third highest object-level behavior: Llama-3.1-70b",
        x_axis_title="Object-level Probability",
        y_axis_title="Predicted Meta Probability",
        chart_title="Third Object-Level Behavior Calibration",
        filename="third_calibration_curve.pdf",
    )

    return expected_meta_proba


async def main():
    path = "evals/datasets/val_animals.jsonl"
    train_path = "evals/datasets/train_animals.jsonl"
    limit = 500
    read = read_jsonl_file_into_basemodel(path, NumberRow).take(limit) + read_jsonl_file_into_basemodel(
        train_path, NumberRow
    ).take(limit)
    print(f"Read {len(read)} animals from {path}")
    caller = UniversalCallerV2().with_file_cache(cache_path="animals_cache.jsonl")

    # models = [
    #     # "gpt-4o",
    #     # "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU",
    #     # "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9lcZU3Vv",
    #     "accounts/chuajamessh-b7a735/models/llama-70b-14aug-20k-jinja",
    # ]

    # results = []
    # baselines = []
    # for model in models:

    #     results.append(plots)
    #     baselines.append(baseline)
    model = "accounts/chuajamessh-b7a735/models/llama-70b-14aug-20k-jinja"
    cross_pred = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::A4x8uaCm"
    # model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU"
    # model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9lcZU3Vv"
    plot_data: Slist[tuple[float, float]] = await process_model_scatter(
        model=model, read=read, caller=caller, cross_prediction_model=cross_pred
    )

    # # plot_regression(
    # plot_line_plot(
    #     results,
    #     baselines,
    #     # models,
    #     # ["GPT-4o", "Self-prediction trained GPT-4o", "Self-prediction trained GPT-3.5-Turbo"],
    #     ["GPT-4o", "GPT-3.5-Turbo", "Llama-3.1-70b"],
    #     x_axis_title="Behavior's probability",
    #     y_axis_title="Hypothetical question accuracy",
    #     chart_title="",
    #     # chart_title="Top token probability vs hypothetical question accuracy (Multiple Models)"
    # )


if __name__ == "__main__":
    setup_environment()
    asyncio.run(main())
