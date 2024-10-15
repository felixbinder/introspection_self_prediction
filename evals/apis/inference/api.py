import asyncio
import logging
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Callable, Literal, Union

import matplotlib.pyplot as plt
import numpy as np

from evals.apis.inference.anthropic_api import ANTHROPIC_MODELS, AnthropicChatModel
from evals.apis.inference.fireworks_api import FireworksModel
from evals.apis.inference.gemini_api import GeminiModel
from evals.apis.inference.model import InferenceAPIModel
from evals.apis.inference.model_test_api import TestModel
from evals.apis.inference.openai.chat import OpenAIChatModel
from evals.apis.inference.openai.completion import OpenAICompletionModel
from evals.data_models.inference import LLMResponse
from evals.data_models.messages import Prompt
from evals.utils import (
    COMPLETION_MODELS,
    GEMINI_MODELS,
    GPT_CHAT_MODELS,
    load_secrets,
    setup_environment,
)

LOGGER = logging.getLogger(__name__)

# repo config name to fireworks model id
ConfigToFireworks = {
    "llama-70b-fireworks": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    "llama-8b-fireworks": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "llama-70b-ft-test": "accounts/chuajamessh-b7a735/models/llama-70b-14aug-test",
    "llama-8b-14aug-20k": "accounts/chuajamessh-b7a735/models/llama-8b-14aug-20k",
    "llama-70b-14aug-5k": "accounts/chuajamessh-b7a735/models/llama-70b-14aug-5k",
    "llama-8b-14aug-20k-jinja": "accounts/chuajamessh-b7a735/models/llama-8b-14aug-20k-jinja",
    "llama-70b-14aug-5k-jinja": "accounts/chuajamessh-b7a735/models/llama-70b-14aug-5k-jinja",
    "llama-70b-14aug-20k-jinja": "accounts/chuajamessh-b7a735/models/llama-70b-14aug-20k-jinja",
    "llama-70b-14aug-20k-jinja-claude-shift": "accounts/chuajamessh-b7a735/models/llama-70b-14aug-20k-jinja-claude-shift",
    "llama-70b-gpt35-9odjqay1": "accounts/chuajamessh-b7a735/models/llama-70b-gpt35-9odjqay1",
    "llama-70b-gpt4o-9ouvkrcu": "accounts/chuajamessh-b7a735/models/llama-70b-gpt4o-9ouvkrcu",
}
fireworks_models = set(ConfigToFireworks.values())


class InferenceAPI:
    """
    A wrapper around the OpenAI and Anthropic APIs that automatically manages rate limits and valid responses.
    """

    def __init__(
        self,
        anthropic_num_threads: int = 12,
        openai_fraction_rate_limit: float = 0.99,
        organization: str = "DEFAULT_ORG",
        prompt_history_dir: Path = None,
    ):
        if openai_fraction_rate_limit >= 1:
            raise ValueError("openai_fraction_rate_limit must be less than 1")

        self.anthropic_num_threads = anthropic_num_threads
        self.openai_fraction_rate_limit = openai_fraction_rate_limit
        self.organization = organization
        self.prompt_history_dir = prompt_history_dir

        secrets = load_secrets("SECRETS")
        if self.organization is None:
            self.organization = "DEFAULT_ORG"

        self._openai_completion = OpenAICompletionModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            organization=secrets[self.organization],
            prompt_history_dir=self.prompt_history_dir,
        )

        if "NYUARG_ORG" in secrets:
            self._openai_gpt4base = OpenAICompletionModel(
                frac_rate_limit=self.openai_fraction_rate_limit,
                organization=secrets["NYUARG_ORG"],
                prompt_history_dir=self.prompt_history_dir,
            )
        else:
            self._openai_gpt4base = self._openai_completion

        self._openai_chat = OpenAIChatModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            organization=secrets[self.organization],
            prompt_history_dir=self.prompt_history_dir,
        )

        self._anthropic_chat = AnthropicChatModel(
            num_threads=self.anthropic_num_threads,
            prompt_history_dir=self.prompt_history_dir,
        )

        self._gemini_chat = GeminiModel(prompt_history_dir=self.prompt_history_dir)

        # self._huggingface_chat = HuggingFaceModel(prompt_history_dir=self.prompt_history_dir)

        self._fireworks_chat = FireworksModel() if "FIREWORKS_API_KEY" in secrets else None
        self._test_chat = TestModel()

        self.running_cost = 0
        self.model_timings = {}
        self.model_wait_times = {}

    def model_id_to_class(self, model_id: str) -> InferenceAPIModel:
        if model_id in fireworks_models:
            assert self._fireworks_chat is not None, "Fireworks API not available, did you set FIREWORKS_API_KEY?"
            return self._fireworks_chat
        if model_id == "gpt-4-base":
            return self._openai_gpt4base  # NYU ARG is only org with access to this model
        elif model_id in COMPLETION_MODELS:
            return self._openai_completion
        elif model_id in GPT_CHAT_MODELS or "ft:gpt-3.5-turbo" in model_id or "ft:gpt-4" in model_id:
            return self._openai_chat
        elif model_id in ANTHROPIC_MODELS:
            return self._anthropic_chat
        elif model_id in GEMINI_MODELS or "projects/" in model_id:
            return self._gemini_chat
        elif model_id == "test":
            return self._test_chat
        else:
            raise ValueError(f"Unknown model_id: {model_id}")
            # return self._huggingface_chat

    def filter_responses(
        self,
        candidate_responses: list[LLMResponse],
        n: int,
        is_valid: Callable[[str], bool],
        insufficient_valids_behaviour: Literal["error", "continue", "pad_invalids"] = "error",
    ) -> list[LLMResponse]:
        # filter out invalid responses
        num_candidates = len(candidate_responses)
        valid_responses = [response for response in candidate_responses if is_valid(response.completion)]
        num_valid = len(valid_responses)
        success_rate = num_valid / num_candidates if len(candidate_responses) > 0 else 0
        if success_rate < 1:
            LOGGER.info(f"`is_valid` success rate: {success_rate * 100:.2f}%")

        # return the valid responses, or pad with invalid responses if there aren't enough
        if num_valid < n:
            match insufficient_valids_behaviour:
                case "error":
                    raise RuntimeError(f"Only found {num_valid} valid responses from {num_candidates} candidates.")
                case "continue":
                    responses = valid_responses
                case "pad_invalids":
                    invalid_responses = [
                        response for response in candidate_responses if not is_valid(response.completion)
                    ]
                    invalids_needed = n - num_valid
                    responses = [*valid_responses, *invalid_responses[:invalids_needed]]
                    LOGGER.info(
                        f"Padded {num_valid} valid responses with {invalids_needed} invalid responses to get {len(responses)} total responses"
                    )
        else:
            responses = valid_responses
        return responses[:n]

    async def __call__(
        self,
        model_ids: Union[str, list[str]],
        prompt: Prompt,
        max_tokens: int = None,
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 10,
        num_candidates_per_completion: int = 1,
        is_valid: Callable[[str], bool] = lambda _: True,
        insufficient_valids_behaviour: Literal["error", "continue", "pad_invalids"] = "error",
        seed: int = 0,
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Make maximally efficient API requests for the specified model(s) and prompt.

        Args:
            model_ids: The model(s) to call. If multiple models are specified, the output will be sampled from the
                cheapest model that has capacity. All models must be from the same class (e.g. OpenAI Base,
                OpenAI Chat, or Anthropic Chat). Anthropic chat will error if multiple models are passed in.
                Passing in multiple models could speed up the response time if one of the models is overloaded.
            prompt: The prompt to send to the model(s). Type should be Prompt.
            max_tokens: The maximum number of tokens to request from the API (argument added to
                standardize the Anthropic and OpenAI APIs, which have different names for this).
            print_prompt_and_response: Whether to print the prompt and response to stdout.
            n: The number of completions to request.
            max_attempts_per_api_call: Passed to the underlying API call. If the API call fails (e.g. because the
                API is overloaded), it will be retried this many times. If still fails, an exception will be raised.
            num_candidates_per_completion: How many candidate completions to generate for each desired completion.
                n*num_candidates_per_completion completions will be generated, then is_valid is applied as a filter,
                then the remaining completions are returned up to a maximum of n.
            is_valid: Candiate completions are filtered with this predicate.
            insufficient_valids_behaviour: What should we do if the remaining completions after applying the is_valid
                filter is shorter than n.
                - error: raise an error
                - continue: return the valid responses, even if they are fewer than n
                - pad_invalids: pad the list with invalid responses up to n
            seed: The seed to use for the random number generator. Note that Anthropic presently does not support setting a random seed.
        """

        assert "max_tokens_to_sample" not in kwargs, "max_tokens_to_sample should be passed in as max_tokens."
        assert isinstance(model_ids, str), "Wth? This should be a string. (List of strings not supported)"

        if isinstance(model_ids, str):
            # trick to double rate limit for most recent model only
            if model_ids.endswith("-0613"):
                model_ids = [model_ids, model_ids.replace("-0613", "")]
                # print(f"doubling rate limit for most recent model {model_ids}")
            elif model_ids.endswith("-0914"):
                model_ids = [model_ids, model_ids.replace("-0914", "")]
            else:
                model_ids = [model_ids]

        # Convert config names to fireworks model ids if necessary
        model_ids = [ConfigToFireworks.get(model_id, model_id) for model_id in model_ids]

        model_classes = [self.model_id_to_class(model_id) for model_id in model_ids]
        if len(set(str(type(x)) for x in model_classes)) != 1:
            raise ValueError("All model ids must be of the same type.")

        # standardize max_tokens argument
        model_class = model_classes[0]
        if isinstance(model_class, AnthropicChatModel):
            max_tokens = max_tokens if max_tokens is not None else 2000
            kwargs["max_tokens_to_sample"] = max_tokens
        else:
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

        num_candidates = num_candidates_per_completion * n
        if isinstance(model_class, AnthropicChatModel):
            # Anthropic chat doesn't support generating multiple candidates at once, so we have to do it manually
            candidate_responses = list(
                chain.from_iterable(
                    await asyncio.gather(
                        *[
                            model_class(
                                model_ids,
                                prompt,
                                print_prompt_and_response,
                                max_attempts_per_api_call,
                                **kwargs,
                            )
                            for _ in range(num_candidates)
                        ]
                    )
                )
            )
        else:
            # if model_class != self._huggingface_chat:
            kwargs.pop("cais_path", None)
            candidate_responses = await model_class(
                model_ids,
                prompt,
                print_prompt_and_response,
                max_attempts_per_api_call,
                n=num_candidates,
                seed=seed,
                **kwargs,
            )

        # filter based on is_valid criteria and insufficient_valids_behaviour
        responses = self.filter_responses(
            candidate_responses,
            n,
            is_valid,
            insufficient_valids_behaviour,
        )

        # update running cost and model timings
        self.running_cost += sum(response.cost for response in candidate_responses)
        for response in candidate_responses:
            self.model_timings.setdefault(response.model_id, []).append(response.api_duration)
            self.model_wait_times.setdefault(response.model_id, []).append(response.duration - response.api_duration)

        return responses

    def reset_cost(self):
        self.running_cost = 0

    def log_model_timings(self):
        if len(self.model_timings) > 0:
            plt.figure(figsize=(10, 6))
            for model in self.model_timings:
                timings = np.array(self.model_timings[model])
                wait_times = np.array(self.model_wait_times[model])
                LOGGER.info(
                    f"{model}: response {timings.mean():.3f}, waiting {wait_times.mean():.3f} (max {wait_times.max():.3f}, min {wait_times.min():.3f})"
                )
                plt.plot(timings, label=f"{model} - Response Time", linestyle="-", linewidth=2)
                plt.plot(wait_times, label=f"{model} - Waiting Time", linestyle="--", linewidth=2)
            plt.legend()
            plt.title("Model Performance: Response and Waiting Times")
            plt.xlabel("Sample Number")
            plt.ylabel("Time (seconds)")
            plt.savefig(self.prompt_history_dir / "model_timings.png", bbox_inches="tight")
            plt.close()


async def demo():
    setup_environment()
    model_api = InferenceAPI()

    prompt_examples = [
        [
            {"role": "system", "content": "You are a comedic pirate."},
            {"role": "user", "content": "Hello!"},
        ],
        [
            {
                "role": "system",
                "content": "You are a swashbuckling space-faring voyager.",
            },
            {"role": "user", "content": "Hello!"},
        ],
        [
            {
                "role": "system",
                "content": "You are a swashbuckling space-faring voyager.",
            },
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "How dare you call me a"},
        ],
        [
            {"role": "none", "content": "whence afterever the storm"},
        ],
    ]
    prompts = [Prompt(messages=messages) for messages in prompt_examples]

    # test anthropic chat, and continuing assistant message
    anthropic_requests = [
        model_api("claude-2.1", prompt=prompts[0], n=1, print_prompt_and_response=True),
        model_api("claude-2.1", prompt=prompts[2], n=1, print_prompt_and_response=True),
    ]

    # test OAI chat, more than 1 model and n > 1
    oai_chat_models = ["gpt-3.5-turbo", "gpt-3.5-turbo-0613"]
    oai_chat_requests = [
        model_api(oai_chat_models, prompt=prompts[1], n=6, print_prompt_and_response=True),
    ]

    # test OAI completion with none message and assistant message to continue
    oai_requests = [
        model_api("gpt-3.5-turbo-instruct", prompt=prompts[2], print_prompt_and_response=True),
        model_api("gpt-3.5-turbo-instruct", prompt=prompts[3], print_prompt_and_response=True),
    ]

    # test HuggingFace chat
    hf_requests = [
        model_api("llama-7b-chat", prompt=prompts[0], n=1, print_prompt_and_response=True),
    ]
    fireworks_requets = [
        model_api("llama-70b-fireworks", prompt=prompts[0], n=1, print_prompt_and_response=True),
    ]
    answer = await asyncio.gather(
        *anthropic_requests, *oai_chat_requests, *oai_requests, *hf_requests, *fireworks_requets
    )

    costs = defaultdict(int)
    for responses in answer:
        for response in responses:
            costs[response.model_id] += response.cost

    print("-" * 80)
    print("Costs:")
    for model_id, cost in costs.items():
        print(f"{model_id}: ${cost}")
    return answer


if __name__ == "__main__":
    asyncio.run(demo())
