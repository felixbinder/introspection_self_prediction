import asyncio
from collections import defaultdict
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from grugstream import Observable
from pydantic import BaseModel
from sklearn.metrics import mean_absolute_error
from slist import Slist

from evals.data_models.hashable import deterministic_hash
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
    write_jsonl_file_from_basemodel,
)


class NumberRow(BaseModel):
    string: str


class CalibrationData(BaseModel):
    expected_prob: float
    predicted_prob: float
    behavior_rank: str  # e.g., "Top", "Second", "Third"


def calc_expected_meta_probs(object_probs: Sequence[Prob]) -> Sequence[Prob]:
    # Extract the second character
    out = defaultdict[str, float](float)
    for idx, prob in enumerate(object_probs):
        assert len(prob.token) >= 2, f"{idx} Token {prob.token} has length {len(prob.token)}, {object_probs=}"
        out[prob.token[1]] += prob.prob
    # Turn into a list of Probs, sorted by highest probability first
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
    expected_object_probs: Sequence[Prob]  # Calculated from object_probs
    meta_probs: Sequence[Prob]  # Calculated from meta_raw_response

    @staticmethod
    def from_animal_responses(responses: Sequence[AnimalResponse]) -> "SampledAnimalResponse":
        modal_object_level_answer = Slist(responses).map(lambda x: x.object_level_answer).mode_or_raise()
        # Proba of the mode
        proba = Slist(responses).map(lambda x: x.object_level_answer).count(modal_object_level_answer) / len(responses)

        modal_meta_parsed_response = Slist(responses).map(lambda x: x.meta_parsed_response).mode_or_raise()
        # Group by sum
        object_probs = (
            Slist(responses)
            .map(lambda x: x.object_level_response)
            .group_by(lambda x: x)
            .map_2(
                # Key: token, value: list of responses
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
        # Second most likely object-level behaviour
        second_expected: str | None = (
            Slist(self.expected_object_probs).sort_by(lambda x: -x.prob).map(lambda x: x.token).get(1, or_else=None)
        )
        if second_expected is None:
            return None
        return self.second_expected_object_proba(), self.get_meta_proba_for_behaviour(second_expected)

    def third_expected_with_meta(self) -> tuple[float, float] | None:
        # Third most likely object-level behaviour
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
        # Highly retarded, but the \n\n is the first token but gets stripped away.
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


def plot_combined_calibration_curve(
    data: List[CalibrationData],
    model_name: str,
    filename: str = "combined_calibration_curve.pdf",
    x_axis_title: str = "Model Probability",
    y_axis_title: str = "Model Accuracy",
    chart_title: str = "Calibration Curve for Object-Level Behaviors",
    num_bins: int = 10,
) -> None:
    """
    Plots a combined calibration curve for multiple behaviors with distinct hues.

    Args:
        data (List[CalibrationData]): A list of CalibrationData instances.
        model_name (str): The name of the model for labeling purposes.
        filename (str): The filename to save the plot. Defaults to "combined_calibration_curve.pdf".
        x_axis_title (str): Label for the x-axis. Defaults to "Model Probability".
        y_axis_title (str): Label for the y-axis. Defaults to "Model Accuracy".
        chart_title (str): The title of the chart.
        num_bins (int): Number of bins for calibration.
    """
    import pandas as pd

    # Convert data to a DataFrame for easier handling
    df = pd.DataFrame([data_item.dict() for data_item in data])

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(4, 4))

    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12

    # Define behavior ranks and corresponding labels
    # behavior_ranks = ["Overall", "Top", "Second", "Third"]
    behavior_ranks = ["Top", "Second", "Third"]
    # Define colors for each behavior rank
    palette = sns.color_palette("Set1", n_colors=len(behavior_ranks))
    # reverse the palette
    palette = palette[::-1]

    rename_map = {"Top": "First", "Second": "Second", "Third": "Third"}
    df_tuples = df[["expected_prob", "predicted_prob"]].values.tolist()
    overall_x_bins, overall_y_bins = pandas_equal_frequency_binning(df_tuples, num_bins=num_bins)
    overall_mad = mean_absolute_error([x * 100 for x in overall_x_bins], [y * 100 for y in overall_y_bins])

    # Iterate over each behavior rank and plot
    for rank, color in zip(behavior_ranks, palette):
        if rank == "Overall":
            subset = df
        else:
            subset = df[df["behavior_rank"] == rank]
        if subset.empty:
            continue

        # Convert subset to list of tuples for binning
        subset_tuples = subset[["expected_prob", "predicted_prob"]].values.tolist()

        # Bin the data
        # bin_means_x, bin_means_y = equal_sized_binning(subset_tuples, num_bins=num_bins)
        bin_means_x, bin_means_y = pandas_equal_frequency_binning(subset_tuples, num_bins=num_bins)

        # Convert to percentages for MAD calculation
        mad = mean_absolute_error([x * 100 for x in bin_means_x], [y * 100 for y in bin_means_y])
        # square error WITHOUT THE BINNING
        # mse = mean_squared_error(subset["expected_prob"], subset["predicted_prob"])
        # Renamed "Top" To "First"
        rank_renamed = rename_map.get(rank, rank)

        # Plot the binned data
        ax.plot(
            bin_means_x,
            bin_means_y,
            marker="o",
            linestyle="-",
            color=color,
            label=f"{rank_renamed} (MAD={mad:.1f})",
        )

    # Add a y = x reference line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

    # Set labels and title
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
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
    ax.legend(title="Object-level Behavior", loc="upper left")

    # Tight layout for better spacing
    plt.tight_layout()
    # 250 vs 250 size

    # Save the plot
    plt.savefig(filename)

    # Display the plot
    plt.show()

    # Optionally, save the plot
    plt.close()


def equal_sized_binning(data: List[Tuple[float, float]], num_bins: int = 10) -> Tuple[List[float], List[float]]:
    """
    Bins the data into equal-sized bins based on expected probabilities and computes the mean of each bin.

    Args:
        data (List[Tuple[float, float]]): A list of tuples containing (expected_prob, predicted_prob).
        num_bins (int): Number of bins to divide the data into.

    Returns:
        Tuple[List[float], List[float]]: Means of expected and predicted probabilities per bin.
    """
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


def pandas_equal_frequency_binning(
    data: List[Tuple[float, float]], num_bins: int = 10
) -> Tuple[List[float], List[float]]:
    df = pd.DataFrame(data, columns=["expected_prob", "predicted_prob"])
    df["bin"] = pd.qcut(df["expected_prob"], q=num_bins, duplicates="drop")

    bin_means = df.groupby("bin", observed=False).agg({"expected_prob": "mean", "predicted_prob": "mean"}).reset_index()

    return bin_means["expected_prob"].tolist(), bin_means["predicted_prob"].tolist()


async def process_model_scatter(
    model: str, read: List[NumberRow], caller: ModelCallerV2, cross_prediction_model: str | None = None
) -> List[CalibrationData]:
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

    # Extract calibration data for top, second, and third behaviors
    expected_meta_proba_top: Slist[tuple[float, float]] = result_clean.map(lambda x: x.first_expected_with_meta())
    expected_meta_proba_second: Slist[tuple[float, float]] = result_clean.map(
        lambda x: x.second_expected_with_meta()
    ).flatten_option()
    expected_meta_proba_third: Slist[tuple[float, float]] = result_clean.map(
        lambda x: x.third_expected_with_meta()
    ).flatten_option()

    # Prepare combined data with labels using CalibrationData instances
    combined_data: List[CalibrationData] = []

    # Add top behavior data
    combined_data += [
        CalibrationData(expected_prob=x, predicted_prob=y, behavior_rank="Top") for x, y in expected_meta_proba_top
    ]

    # Add second behavior data
    combined_data += [
        CalibrationData(expected_prob=x, predicted_prob=y, behavior_rank="Second")
        for x, y in expected_meta_proba_second
    ]

    # Add third behavior data
    combined_data += [
        CalibrationData(expected_prob=x, predicted_prob=y, behavior_rank="Third") for x, y in expected_meta_proba_third
    ]

    return combined_data


def to_cache_name(model: str, cross_pred: str | None) -> str:
    hashed = deterministic_hash(f"{model}-{cross_pred}")
    return f"cache/cache_{hashed}.jsonl"


async def main():
    path = "evals/datasets/val_animals.jsonl"
    train_path = "evals/datasets/train_animals.jsonl"
    limit = 500

    # Define your model and cross-prediction model
    # model = "gpt-4o-2024-05-13"
    # model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU"
    # cross_pred = "accounts/chuajamessh-b7a735/models/llama-70b-gpt4o-9ouvkrcu"

    model = "accounts/chuajamessh-b7a735/models/llama-70b-14aug-20k-jinja"
    # model = "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9oDjQaY1"
    # model = "accounts/fireworks/models/llama-v3p1-70b-instruct"
    # model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::A6Ji2P4o"
    # cross_pred = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::A4x8uaCm"
    cross_pred = None
    USE_CACHE = True
    if not USE_CACHE:
        read_val = read_jsonl_file_into_basemodel(path, NumberRow).take(limit)
        read_train = read_jsonl_file_into_basemodel(train_path, NumberRow).take(limit)
        read = read_val + read_train
        print(f"Read {len(read)} animals from {path} and {train_path}")
        caller = UniversalCallerV2().with_file_cache(cache_path="cache/animals_cache.jsonl")
        # Process and plot calibration curves
        combined_plot_data: List[CalibrationData] = await process_model_scatter(
            model=model, read=read, caller=caller, cross_prediction_model=cross_pred
        )

        jsonl_name = to_cache_name(model, cross_pred)

        # write to cache
        write_jsonl_file_from_basemodel(path=jsonl_name, basemodels=combined_plot_data)

    else:
        jsonl_name = to_cache_name(model, cross_pred)

        combined_plot_data = read_jsonl_file_into_basemodel(jsonl_name, CalibrationData)
    # Plot combined calibration curve
    filename = "llama_calibration_top_3.pdf"
    # filename = "gpt4o_calibration_top_3.pdf"
    plot_combined_calibration_curve(
        data=combined_plot_data,
        model_name=model,
        x_axis_title="Object-level Behavior Probability",
        y_axis_title="Hypothetical Probability",
        chart_title="",
        filename=filename,
        num_bins=10,
    )

    # If you have multiple models, you can loop through them and plot accordingly
    # For example:
    # models = [
    #     "model_1",
    #     "model_2",
    #     "model_3",
    # ]
    # for model in models:
    #     combined_plot_data = await process_model_scatter(
    #         model=model, read=read, caller=caller, cross_prediction_model=cross_pred
    #     )
    #     # Optionally, modify the plotting function to handle multiple models with different hues


if __name__ == "__main__":
    setup_environment()
    asyncio.run(main())
