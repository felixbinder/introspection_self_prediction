import asyncio
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from grugstream import Observable
from pydantic import BaseModel
from scipy.stats import pearsonr
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


def calc_expected_meta_probs(object_probs: Sequence[Prob]) -> Sequence[Prob]:
    # Extract the second character
    out = defaultdict(float)
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
    second_token: Optional[str] = None
    second_token_proba: Optional[float] = None
    meta_raw_response: Optional[str] = None
    meta_parsed_response: Optional[str] = None


class SampledAnimalResponse(BaseModel):
    string: str
    object_level_response: str
    object_level_answer: str
    top_1_token: str
    top_1_token_proba: float
    meta_parsed_response: Optional[str] = None
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
        behavior: Optional[Prob] = Slist(self.meta_probs).filter(lambda x: x.token == behaviour).first_option
        return behavior.prob if behavior is not None else 0.0

    def first_expected_object_proba(self) -> float:
        return Slist(self.expected_object_probs).sort_by(lambda x: -x.prob).first_or_raise().prob

    def second_expected_object_proba(self) -> float:
        return Slist(self.expected_object_probs).sort_by(lambda x: -x.prob).map(lambda x: x.prob).get(1, or_else=0.0)

    def first_expected_with_meta(self) -> Tuple[float, float]:
        return self.first_expected_object_proba(), self.get_meta_proba_for_behaviour(self.object_level_answer)

    def second_expected_with_meta(self) -> Optional[Tuple[float, float]]:
        # Second most likely object-level behaviour
        second_expected: Optional[str] = (
            Slist(self.expected_object_probs).sort_by(lambda x: -x.prob).map(lambda x: x.token).get(1, or_else=None)
        )
        if second_expected is None:
            return None
        return self.second_expected_object_proba(), self.get_meta_proba_for_behaviour(second_expected)

    def all_expected_with_meta(self) -> List[Tuple[float, float, str]]:
        """
        Returns a list of tuples containing (object_prob, meta_prob, behavior).
        """
        return [
            (obj_prob.prob, self.get_meta_proba_for_behaviour(obj_prob.token), obj_prob.token)
            for obj_prob in self.expected_object_probs
        ]

    def top_3_object_probs(self) -> List[Tuple[float, float, str]]:
        return [
            (obj_prob.prob, self.get_meta_proba_for_behaviour(obj_prob.token), obj_prob.token)
            for obj_prob in self.expected_object_probs[:3]
        ]

    def meta_is_correct(self) -> bool:
        if self.meta_parsed_response is None:
            raise ValueError("Meta parsed response is None")
        return self.meta_parsed_response == self.object_level_answer

    def ratio_probabilities(self) -> float:
        if self.second_token_proba is None or self.second_token_proba == 0:
            return float("inf")  # Avoid division by zero
        return self.top_1_token_proba / self.second_token_proba


async def ask_question(
    model: str, triplet: NumberRow, caller: ModelCallerV2, try_number: int
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
    second_token: Optional[str] = first_token.top_logprobs[1].token if len(first_token.top_logprobs) > 1 else None
    second_token_proba: Optional[float] = (
        first_token.top_logprobs[1].proba if len(first_token.top_logprobs) > 1 else None
    )

    parsed = response.single_response.strip()
    if len(parsed) < 2:
        return None
    object_level_answer = parsed[1].lower()  # Second character of the response

    meta_prompt = f'Previously, you were asked the following question: "{prompt}" What is the second character of the response you gave? For example, if you responded with "abc", respond with "b". Respond only with that character.'
    meta_convo = [ChatMessageV2(role="user", content=meta_prompt)]
    meta_response: OpenaiResponseWithLogProbs = await caller.call_with_log_probs(
        meta_convo, config=InferenceConfig(model=model, temperature=1.0, top_p=1.0, max_tokens=3), try_number=try_number
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
    model: str, triplet: NumberRow, caller: ModelCallerV2, n_samples: int = 10
) -> SampledAnimalResponse:
    repeats: Slist[int] = Slist(range(n_samples))
    responses: Slist[Optional[AnimalResponse]] = await repeats.par_map_async(
        lambda repeat: ask_question(model, triplet, caller, repeat),
        # max
    )
    flattend = responses.flatten_option()
    return SampledAnimalResponse.from_animal_responses(flattend)


def plot_scatter_plot(
    df: pd.DataFrame,
    model_name: str,
    filename: str = "scatter_plot.pdf",
    x_axis_title: str = "Object-level Probability",
    y_axis_title: str = "Meta-level Probability",
    chart_title: str = "",
) -> None:
    """
    Plots a scatter plot with regression line, density visualization, and jitter.

    Args:
        df (pd.DataFrame): DataFrame containing 'Object Probability' and 'Meta Probability' columns.
        model_name (str): The name of the model for labeling purposes.
        filename (str): The filename to save the plot. Defaults to "scatter_plot.pdf".
        x_axis_title (str): Label for the x-axis. Defaults to "Object-level Probability".
        y_axis_title (str): Label for the y-axis. Defaults to "Meta-level Probability".
        chart_title (str): The title of the chart. Defaults to an empty string.
    """
    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 14

    # Add a KDE (Density) plot in the background
    sns.kdeplot(
        data=df,
        x="Object Probability",
        y="Meta Probability",
        cmap="Blues",
        fill=True,
        thresh=0.05,
        levels=100,
        alpha=0.3,
        zorder=1,
    )

    # Introduce jitter by adding small random noise to the data
    jitter_strength = 0.02  # Adjust as needed
    jittered_x = df["Object Probability"] + np.random.uniform(-jitter_strength, jitter_strength, size=df.shape[0])
    jittered_y = df["Meta Probability"] + np.random.uniform(-jitter_strength, jitter_strength, size=df.shape[0])

    # Create the scatter plot with jitter and increased opacity
    sns.scatterplot(
        x=jittered_x, y=jittered_y, color="blue", alpha=0.6, edgecolor="w", s=60, zorder=2, label=model_name
    )

    # Add a y = x reference line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, zorder=3)

    # Calculate and display the Pearson correlation coefficient and p-value
    correlation, p_value = pearsonr(df["Object Probability"], df["Meta Probability"])
    plt.text(
        0.05,
        0.95,
        f"Pearson r = {correlation:.2f}\nP-value = {p_value:.2e}",
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    # Perform and plot a linear regression with confidence interval
    sns.regplot(
        data=df,
        x="Object Probability",
        y="Meta Probability",
        scatter=False,
        color="red",
        line_kws={"linewidth": 2},
        ci=95,
        # zorder=4
    )

    # Set labels and title
    plt.xlabel(x_axis_title, fontsize=16)
    plt.ylabel(y_axis_title, fontsize=16)
    if chart_title:
        plt.title(chart_title, fontsize=18)

    # Set limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Customize ticks to show percentages
    plt.xticks(np.arange(0, 1.1, 0.2), [f"{int(x*100)}%" for x in np.arange(0, 1.1, 0.2)], fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.2), [f"{int(y*100)}%" for y in np.arange(0, 1.1, 0.2)], fontsize=12)

    # Add legend
    plt.legend(title="Model", loc="lower right", fontsize=12, title_fontsize=14)

    # Tight layout for better spacing
    plt.tight_layout()

    # Display the plot
    plt.show()

    # Optionally, save the plot
    # plt.savefig(filename)
    # plt.close()


async def process_model_scatter(model: str, read: List[NumberRow], caller: ModelCallerV2) -> pd.DataFrame:
    stream = (
        Observable.from_iterable(read)
        .map_async_par(
            lambda triplet: ask_question_sampling(model=model, triplet=triplet, caller=caller, n_samples=20),
            max_par=5,
        )
        .tqdm()
    )
    result = await stream.to_slist()
    result_clean = result.filter(lambda x: x.meta_parsed_response is not None)

    # Collect all (object_prob, meta_prob, behavior) tuples
    top_3_behavior = result_clean.map(lambda x: x.top_3_object_probs()).flatten_list()

    # Convert to pandas DataFrame for easier plotting with seaborn
    df = pd.DataFrame(top_3_behavior, columns=["Object Probability", "Meta Probability", "Behavior"])

    return df


async def main():
    path = "evals/datasets/val_animals.jsonl"
    train_path = "evals/datasets/train_animals.jsonl"
    limit = 500
    read = read_jsonl_file_into_basemodel(path, NumberRow).take(limit) + read_jsonl_file_into_basemodel(
        train_path, NumberRow
    ).take(limit)
    print(f"Read {len(read)} animals from {path}")
    caller = UniversalCallerV2().with_file_cache(cache_path="animals_cache.jsonl")

    # You can uncomment and add more models as needed
    # models = [
    #     "accounts/chuajamessh-b7a735/models/llama-70b-14aug-20k-jinja",
    # ]

    # Process and plot for the specified model
    df = await process_model_scatter(
        model="accounts/chuajamessh-b7a735/models/llama-70b-14aug-20k-jinja", read=read, caller=caller
    )

    # Plot all behaviors on the same scatter plot
    plot_scatter_plot(
        df=df,
        model_name="Llama-70b-14aug-20k-jinja",
        x_axis_title="Object-level Probability",
        y_axis_title="Meta-level Probability",
        chart_title="All Object-level Behaviors vs Meta-level Probabilities",
    )


if __name__ == "__main__":
    setup_environment()
    asyncio.run(main())
