import asyncio
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from grugstream import Observable
from pydantic import BaseModel
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


# Define a Setup class to encapsulate each setup's configuration
class Setup(BaseModel):
    name: str
    model: str
    cross_pred: Optional[str] = None


class NumberRow(BaseModel):
    string: str


class BehaviorMetaData(BaseModel):
    expected_prob: float
    predicted_prob: float
    expected_meta_behavior: str
    meta_behavior: str


# Modify CalibrationData to include setup_name instead of behavior_rank
class CalibrationData(BaseModel):
    expected_prob: float
    predicted_prob: float
    setup_name: str  # Identifier for the setup

    # New fields added for behavior tracking
    expected_meta_behavior: str
    meta_behavior: str


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

    def first_expected_with_meta(self) -> BehaviorMetaData:
        return BehaviorMetaData(
            expected_prob=self.first_expected_object_proba(),
            predicted_prob=self.get_meta_proba_for_behaviour(self.object_level_answer),
            expected_meta_behavior=self.object_level_answer,
            meta_behavior=self.object_level_answer,  # Assuming meta_behavior is same as expected_meta_behavior
        )

    def second_expected_with_meta(self) -> Optional[BehaviorMetaData]:
        second_expected: str | None = (
            Slist(self.expected_object_probs).sort_by(lambda x: -x.prob).map(lambda x: x.token).get(1, or_else=None)
        )
        if second_expected is None:
            return None
        return BehaviorMetaData(
            expected_prob=self.second_expected_object_proba(),
            predicted_prob=self.get_meta_proba_for_behaviour(second_expected),
            expected_meta_behavior=second_expected,
            meta_behavior=second_expected,  # Assuming meta_behavior is same as expected_meta_behavior
        )

    def third_expected_with_meta(self) -> Optional[BehaviorMetaData]:
        third_expected: str | None = (
            Slist(self.expected_object_probs).sort_by(lambda x: -x.prob).map(lambda x: x.token).get(2, or_else=None)
        )
        if third_expected is None:
            return None
        return BehaviorMetaData(
            expected_prob=self.third_expected_object_proba(),
            predicted_prob=self.get_meta_proba_for_behaviour(third_expected),
            expected_meta_behavior=third_expected,
            meta_behavior=third_expected,  # Assuming meta_behavior is same as expected_meta_behavior
        )

    def meta_is_correct(self) -> bool:
        if self.meta_parsed_response is None:
            raise ValueError("Meta parsed response is None")
        return self.meta_parsed_response == self.object_level_answer


async def ask_question(
    model: str, triplet: NumberRow, caller: ModelCallerV2, try_number: int, cross_prediction_model: Optional[str] = None
) -> Optional[AnimalResponse]:
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
    )


async def ask_question_sampling(
    model: str,
    triplet: NumberRow,
    caller: ModelCallerV2,
    n_samples: int = 10,
    cross_prediction_model: Optional[str] = None,
) -> SampledAnimalResponse:
    repeats: Slist[int] = Slist(range(n_samples))
    responses: Slist[Optional[AnimalResponse]] = await repeats.par_map_async(
        lambda repeat: ask_question(model, triplet, caller, repeat, cross_prediction_model=cross_prediction_model)
    )
    flattend = responses.flatten_option()
    return SampledAnimalResponse.from_animal_responses(flattend)


def plot_combined_calibration_curve(
    data: List[CalibrationData],
    filename: str = "combined_calibration_curve.pdf",
    x_axis_title: str = "Model Probability",
    y_axis_title: str = "Model Accuracy",
    chart_title: str = "",
    num_bins: int = 10,
    show_legend: bool = True,
) -> None:
    """
    Plots a combined calibration curve for multiple setups with distinct hues.

    Args:
        data (List[CalibrationData]): A list of CalibrationData instances.
        filename (str): The filename to save the plot. Defaults to "combined_calibration_curve.pdf".
        x_axis_title (str): Label for the x-axis. Defaults to "Model Probability".
        y_axis_title (str): Label for the y-axis. Defaults to "Model Accuracy".
        chart_title (str): The title of the chart.
        num_bins (int): Number of bins for calibration.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import mean_absolute_error

    # Convert data to a DataFrame for easier handling
    df = pd.DataFrame([data_item.dict() for data_item in data])

    # Initialize the plot with a larger size if necessary
    fig, ax = plt.subplots(figsize=(4, 4))

    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 15  # Reduced font size
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    # despine
    sns.despine()

    # Get unique setups
    setups = df["setup_name"].unique()
    palette = ["#19c484", "#527fe8", "#d05881"]

    # Iterate over each setup and plot
    for idx, setup in reversed(list(enumerate(setups))):
        subset = df[df["setup_name"] == setup]
        if subset.empty:
            continue

        # Convert subset to list of tuples for binning
        subset_tuples = subset[["expected_prob", "predicted_prob"]].values.tolist()

        # Bin the data
        bin_means_x, bin_means_y = pandas_equal_frequency_binning(subset_tuples, num_bins=num_bins)

        # Calculate Mean Absolute Deviation (MAD)
        mad = mean_absolute_error([x * 100 for x in bin_means_x], [y * 100 for y in bin_means_y])

        # Plot the binned data
        ax.plot(
            bin_means_x,
            bin_means_y,
            marker="s",
            linestyle="-",
            color=palette[idx],
            label=f"{setup} (MAD={mad:.1f})",
        )

    # Add a y = x reference line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

    # Set labels and title
    ax.set_xlabel(x_axis_title, fontsize=14)
    ax.set_ylabel(y_axis_title, fontsize=14)
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
    if show_legend:
        # ax.legend(title="", loc="upper left")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title="", loc="upper left")

    # Adjust layout
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Fine-tune as needed

    # Save the figure with no padding
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # Close the figure to free memory


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


async def process_model_scatter(setup: Setup, read: List[NumberRow], caller: ModelCallerV2) -> List[CalibrationData]:
    """
    Processes the data for a given setup and returns calibration data.

    Args:
        setup (Setup): The setup configuration.
        read (List[NumberRow]): The list of data rows to process.
        caller (ModelCallerV2): The model caller instance.

    Returns:
        List[CalibrationData]: A list of calibration data points.
    """
    stream = (
        Observable.from_iterable(read)
        .map_async_par(
            lambda triplet: ask_question_sampling(
                model=setup.model, triplet=triplet, caller=caller, n_samples=20, cross_prediction_model=setup.cross_pred
            ),
            max_par=5,
        )
        .tqdm()
    )
    result = await stream.to_slist()
    result_clean = result.filter(lambda x: x.meta_parsed_response is not None)

    # Extract calibration data for all behaviors with behavior names
    expected_meta_proba_top = result_clean.map(lambda x: x.first_expected_with_meta())
    expected_meta_proba_second = result_clean.map(lambda x: x.second_expected_with_meta()).flatten_option()
    expected_meta_proba_third = result_clean.map(lambda x: x.third_expected_with_meta()).flatten_option()

    # Prepare combined data with setup_name
    combined_data: List[CalibrationData] = []

    # Add top behavior data
    combined_data += [
        CalibrationData(
            expected_prob=item.expected_prob,
            predicted_prob=item.predicted_prob,
            setup_name=setup.name,
            expected_meta_behavior=item.expected_meta_behavior,
            meta_behavior=item.meta_behavior,
        )
        for item in expected_meta_proba_top
    ]

    # Add second behavior data
    combined_data += [
        CalibrationData(
            expected_prob=item.expected_prob,
            predicted_prob=item.predicted_prob,
            setup_name=setup.name,
            expected_meta_behavior=item.expected_meta_behavior,
            meta_behavior=item.meta_behavior,
        )
        for item in expected_meta_proba_second
    ]

    # Add third behavior data
    combined_data += [
        CalibrationData(
            expected_prob=item.expected_prob,
            predicted_prob=item.predicted_prob,
            setup_name=setup.name,
            expected_meta_behavior=item.expected_meta_behavior,
            meta_behavior=item.meta_behavior,
        )
        for item in expected_meta_proba_third
    ]

    return combined_data


def to_cache_name(model: str, cross_pred: Optional[str]) -> str:
    hashed = deterministic_hash(f"{model}-{cross_pred}")
    return f"cache/cache_{hashed}.jsonl"


async def main():
    path = "evals/datasets/val_animals.jsonl"
    # train_path = "evals/datasets/train_animals.jsonl"
    limit = 500

    # Define the three setups as instances of the Setup class
    setups = [
        Setup(
            name="After Self-Prediction",
            model="accounts/chuajamessh-b7a735/models/llama-70b-14aug-20k-jinja",
            cross_pred=None,
        ),
        Setup(
            name="Cross-Prediction",
            model="accounts/chuajamessh-b7a735/models/llama-70b-14aug-20k-jinja",
            cross_pred="ft:gpt-4o-2024-05-13:dcevals-kokotajlo::A4x8uaCm",
        ),
        Setup(
            name="Before Self-Prediction",
            model="accounts/fireworks/models/llama-v3p1-70b-instruct",
            cross_pred=None,
        ),
    ]

    # model = "gpt-4o-2024-05-13"
    # model = "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU"
    # cross_pred = "accounts/chuajamessh-b7a735/models/llama-70b-gpt4o-9ouvkrcu"

    # setups = [
    #     Setup(
    #         name="Self-Prediction Trained",
    #         model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU",
    #         cross_pred=None,
    #     ),
    #     Setup(
    #         name="Cross-Prediction Trained",
    #         # hack cos of monkey patch
    #         # model="accounts/chuajamessh-b7a735/models/llama-70b-gpt4o-9ouvkrcu",
    #         model="ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9oUVKrCU",
    #         cross_pred="accounts/chuajamessh-b7a735/models/llama-70b-gpt4o-9ouvkrcu",
    #         # cross_pred="accounts/chuajamessh-b7a735/models/llama-70b-gpt4o-9ouvkrcu",
    #     ),
    #     Setup(
    #         name="Before Training",
    #         model="gpt-4o-2024-05-13",
    #         cross_pred=None,
    #     ),

    # ]

    USE_CACHE = True
    combined_plot_data: List[CalibrationData] = []

    if not USE_CACHE:
        read_val = read_jsonl_file_into_basemodel(path, NumberRow).take(limit)
        # read_train = read_jsonl_file_into_basemodel(train_path, NumberRow).take(limit)
        read = read_val
        print(f"Read {len(read)} animals from {path}")
        caller = UniversalCallerV2().with_file_cache(cache_path="cache/animals_cache.jsonl")

        # Process each setup and collect calibration data
        for setup in setups:
            print(f"Processing setup: {setup.name}")
            setup_data = await process_model_scatter(setup=setup, read=read, caller=caller)
            combined_plot_data += setup_data

            # Optionally, write to cache
            jsonl_name = to_cache_name(setup.model, setup.cross_pred)
            write_jsonl_file_from_basemodel(path=jsonl_name, basemodels=setup_data)
    else:
        # Read data from cache
        for setup in setups:
            jsonl_name = to_cache_name(setup.model, setup.cross_pred)
            setup_data = read_jsonl_file_into_basemodel(jsonl_name, CalibrationData)
            # Ensure that each CalibrationData has the correct setup_name
            for data_item in setup_data:
                data_item.setup_name = setup.name
            combined_plot_data += setup_data

    # Plot combined calibration curve with hue representing different setups
    # filename = "gpt_4o_calibration.pdf"
    filename = "llama_70b_calibration.pdf"
    plot_combined_calibration_curve(
        data=combined_plot_data,
        # model_name=None,  # Removed as it's no longer needed
        x_axis_title="Object-level Behavior Probability",
        y_axis_title="Hypothetical Probability",
        # chart_title="Calibration Curve for Different Setups",
        show_legend=False,
        filename=filename,
        num_bins=10,
    )


if __name__ == "__main__":
    setup_environment()
    asyncio.run(main())
