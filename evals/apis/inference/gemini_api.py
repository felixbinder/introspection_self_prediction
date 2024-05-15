import asyncio
import logging
import time
from pathlib import Path
from traceback import format_exc
from typing import Any, Coroutine, Optional

from aiolimiter import AsyncLimiter
from vertexai.generative_models import FinishReason, GenerativeModel

from evals.apis.inference.model import InferenceAPIModel
from evals.data_models.inference import LLMResponse
from evals.data_models.messages import Prompt

LOGGER = logging.getLogger(__name__)


GEMINI_MODELS = {
    "gemini-1.0-pro-001",
    "gemini-1.0-pro-002",
    "gemini-1.5-pro-001",
}

FINISH_REASON_MAP = {
    FinishReason.MAX_TOKENS: "max_tokens",
    FinishReason.STOP: "stop_sequence",
    FinishReason.OTHER: "unknown",
}
CHAR_PER_TOKEN = 4  # estimating this at 4


def price_per_token(model_id: str) -> tuple[float, float]:
    """
    Returns the (input token, output token) price for the given model id.
    """
    if model_id.startswith("gemini-1.0-pro"):  # 001 and 002
        # $0.000125 / 1k characters, $0.000375 / 1k characters
        prices = 0.000125, 0.000375
    elif model_id.startswith("gemini-1.5-pro"):  # 001 and 002
        # $0.0025 / 1k characters, $0.0075 / 1k characters
        prices = 0.0025, 0.0075
    else:
        raise ValueError(f"Invalid model id: {model_id}")
    return tuple(price / 1000 * CHAR_PER_TOKEN for price in prices)


class GeminiModel(InferenceAPIModel):
    def __init__(self, prompt_history_dir: Path = None):
        self.prompt_history_dir = prompt_history_dir
        self.limiter = AsyncLimiter(60, 60)  # 60 requests per 60 seconds

    async def _make_api_call(
        self,
        model_ids: list[str],
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts_per_api_call: int,  # unused for now
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Comparable to InferenceAPIModel/OpenAIModel/OpenAIChatModel/_make_api_call
        or InferenceAPIModel/AnthropicChatModel/__call__
        """

        start = time.time()
        assert len(model_ids) == 1, "Gemini implementation only supports one model at a time."
        model_id = model_ids[0]

        prompt_messages = prompt.gemini_format()
        # system_message = next((msg["content"] for msg in prompt_messages if msg["role"] == "system"), None)
        # Only handling prompt response for now
        prompt_messages = [msg["content"] for msg in prompt_messages if msg["role"] != "system"]
        assert len(prompt_messages) == 1, "Only supporting prompt response format for now."
        prompt_file = self.create_prompt_history_file(prompt_messages[0], model_id, self.prompt_history_dir)

        LOGGER.debug(f"Making {model_id} call")

        model = GenerativeModel(model_id)  # not expensive to create
        api_start = time.time()
        generation_config = {
            "max_output_tokens": kwargs.get("max_tokens_to_sample", 2000),
            "temperature": kwargs.get("temperature", 0.0),
        }

        response = await model.generate_content_async(contents=prompt_messages, generation_config=generation_config)

        api_duration = time.time() - api_start
        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        context_token_cost, completion_token_cost = price_per_token(model_id)
        completion_cost = response.usage_metadata.candidates_token_count * completion_token_cost
        context_cost = response.usage_metadata.prompt_token_count * context_token_cost
        total_cost = context_cost + completion_cost

        def safe_text_extract(choice):
            """Handle edge case where model returns empty string."""
            parts = choice.content.parts
            return parts[0].text if parts else ""

        responses = [
            LLMResponse(
                model_id=model_id,
                completion=safe_text_extract(choice),
                stop_reason=FINISH_REASON_MAP.get(choice.finish_reason, "unknown"),
                api_duration=api_duration,
                duration=duration,
                cost=total_cost,
                logprobs=[],  # HACK no logprobs
            )
            for choice in response.candidates
        ]
        self.add_response_to_prompt_file(prompt_file, responses)

        if print_prompt_and_response:
            prompt.pretty_print(responses)

        LOGGER.debug(f"Completed call to {model_id} in {time.time() - start}s.")

        return responses

    async def __call__(
        self, model_ids: list[str], prompt, print_prompt_and_response: bool, max_attempts: int, **kwargs
    ) -> Coroutine[Any, Any, list[LLMResponse]]:
        responses: Optional[list[LLMResponse]] = None
        for i in range(max_attempts):
            try:
                async with self.limiter:
                    responses = await self._make_api_call(
                        model_ids, prompt, print_prompt_and_response, max_attempts, **kwargs
                    )
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                await asyncio.sleep(1.5**i)
            else:
                break

        if responses is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        return responses
