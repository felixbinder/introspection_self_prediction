import asyncio
import logging
import time
from pathlib import Path
from traceback import format_exc
from typing import Optional

import anthropic

from evals.apis.inference.model import InferenceAPIModel
from evals.data_models.inference import LLMResponse
from evals.data_models.messages import Prompt

ANTHROPIC_MODELS = {
    "claude-instant-1",
    "claude-instant-1.2",
    "claude-2.0",
    "claude-v1.3",
    "claude-2.1",
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20240620",
}

LOGGER = logging.getLogger(__name__)


def price_per_token(model_id: str) -> tuple[float, float]:
    """
    Returns the (input token, output token) price for the given model id.
    """
    if model_id == "claude-3-haiku-20240307":
        prices = 0.25, 1.25
    elif model_id == "claude-3-sonnet-20240229":
        prices = 3.0, 15.0
    elif model_id == "claude-3-5-sonnet-20240620":
        prices = 3.0, 15.0
    elif model_id == "claude-3-opus-20240229":
        prices = 15.0, 75.0
    elif model_id == "claude-2.1":
        prices = 8.0, 24.0
    elif model_id == "claude-2.0":
        prices = 8.0, 24.0
    elif model_id.startswith("claude-instant"):
        prices = 0.8, 2.4
    else:
        raise ValueError(f"Invalid model id: {model_id}")
    return tuple(price / 1_000_000 for price in prices)


class AnthropicChatModel(InferenceAPIModel):
    def __init__(self, num_threads: int, prompt_history_dir: Path = None):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        self.client = anthropic.AsyncAnthropic()
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))

    async def __call__(
        self,
        model_ids: list[str],
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()
        assert len(model_ids) == 1, "Anthropic implementation only supports one model at a time."
        model_id = model_ids[0]
        prompt_messages = prompt.anthropic_format()
        system_message = next(
            (msg["content"] for msg in prompt_messages if msg["role"] == "system"), anthropic.NOT_GIVEN
        )
        prompt_messages = [msg for msg in prompt_messages if msg["role"] != "system"]
        prompt_string = prompt.anthropic_format_string()
        prompt_file = self.create_prompt_history_file(prompt_string, model_id, self.prompt_history_dir)

        LOGGER.debug(f"Making {model_id} call")
        response: Optional[anthropic.types.Message] = None
        duration = None

        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()
                    response = await self.client.messages.create(
                        model=model_id,
                        max_tokens=kwargs.get("max_tokens_to_sample", 2000),
                        temperature=kwargs.get("temperature", 0.0),
                        system=system_message,  # this works even if it doesn't show up in the logs
                        messages=prompt_messages,
                    )
                    api_duration = time.time() - api_start
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()} Prompt: {prompt} "
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                await asyncio.sleep(1.5**i)
            else:
                break

        if response is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        # Calculate the cost based on the number of tokens
        context_token_cost, completion_token_cost = price_per_token(model_id)
        context_cost = response.usage.input_tokens * context_token_cost
        completion_cost = response.usage.output_tokens * completion_token_cost
        total_cost = context_cost + completion_cost

        response = LLMResponse(
            model_id=model_id,
            completion=response.content[0].text,
            stop_reason=response.stop_reason,
            duration=duration,
            api_duration=api_duration,
            cost=total_cost,
            logprobs=[],  # HACK Anthropic doesn't give us logprobs
        )

        responses = [response]
        self.add_response_to_prompt_file(prompt_file, responses)

        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses
