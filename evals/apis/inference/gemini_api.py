import asyncio
import logging
import time
from pathlib import Path
from traceback import format_exc
from typing import Any, Coroutine, Optional

import google
from tenacity import retry, retry_if_exception_type, wait_fixed
import vertexai
import vertexai.preview.generative_models as generative_models
from aiolimiter import AsyncLimiter
from tenacity import retry, retry_if_exception_type, wait_fixed
from vertexai.generative_models import FinishReason, GenerativeModel

from evals.apis.inference.model import InferenceAPIModel
from evals.data_models.inference import LLMResponse
from evals.data_models.messages import Prompt
from evals.utils import GCLOUD_LOCATION, GCLOUD_PROJECT

LOGGER = logging.getLogger(__name__)


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
    elif model_id.startswith("projects/"):  # assuming 1.0 pro prices for now since that's the only finetunable model
        prices = 0.000125, 0.000375
    else:
        raise ValueError(f"Invalid model id: {model_id}")
    return tuple(price / 1000 * CHAR_PER_TOKEN for price in prices)


class GeminiModel(InferenceAPIModel):
    def __init__(self, prompt_history_dir: Path = None):
        self.prompt_history_dir = prompt_history_dir
        self.limiter = AsyncLimiter(120, 60)  # 60 requests per 60 seconds

    @retry(
        # Retry if we get a rate limit error
        # api_core.exceptions.ServiceUnavailable,
        # api_core.exceptions.Unknown,
        retry=retry_if_exception_type(
            (
                google.api_core.exceptions.ResourceExhausted,
                google.api_core.exceptions.ServiceUnavailable,
                google.api_core.exceptions.Unknown,
                google.api_core.exceptions.InternalServerError,
            )
        ),
        reraise=True,
        wait=wait_fixed(30),
    )
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
        # Don't need to drop system because doesn't even get output to
        # TODO: update this to handle multiple messages
        prompt_file = self.create_prompt_history_file(prompt.gemini_format_text(), model_id, self.prompt_history_dir)

        LOGGER.debug(f"Making {model_id} call")
        vertexai.init(project=GCLOUD_PROJECT, location=GCLOUD_LOCATION)  # not expensive
        model = GenerativeModel(model_id)  # not expensive to create

        api_start = time.time()
        generation_config = {
            "max_output_tokens": kwargs.get("max_tokens_to_sample", 2000),
            "temperature": kwargs.get("temperature", 0.0),
        }
        safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        chat = model.start_chat(history=prompt_messages[:-1], response_validation=False)
        response = await chat.send_message_async(
            content=prompt_messages[-1], generation_config=generation_config, safety_settings=safety_settings
        )

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

        if response.candidates:
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
        else:  # handle empty responses, usually because of safety block
            responses = [
                LLMResponse(
                    model_id=model_id,
                    completion="",
                    # sometimes returns empty because of safety block
                    stop_reason="safety" if response.prompt_feedback.block_reason == 4 else "unknown",
                    api_duration=api_duration,
                    duration=duration,
                    cost=total_cost,
                    logprobs=[],  # HACK no logprobs
                )
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
        exc: Optional[Exception] = None
        for i in range(max_attempts):
            try:
                async with self.limiter:
                    responses = await self._make_api_call(
                        model_ids, prompt, print_prompt_and_response, max_attempts, **kwargs
                    )
                    break
            except vertexai.generative_models._generative_models.ResponseValidationError as e:
                exc = e
                LOGGER.error(
                    "This prompt is causing anomalous behavior. Treating this as a safety issue and skipping\n",
                    prompt.gemini_format_text(),
                )
                responses = [
                    LLMResponse(
                        model_id=model_ids[0],
                        completion="",
                        # sometimes returns empty because of safety block
                        stop_reason="safety",
                        api_duration=0.0,
                        duration=0.0,
                        cost=0.0,
                        logprobs=[],  # HACK no logprobs
                    )
                ]
                break
            except IndexError as e:
                exc = e
                LOGGER.error(
                    f"Error: {e} while running prompt: {prompt.gemini_format_text()}. This is likely a safety refusal: treating as refusal and skipping."
                )
                responses = [
                    LLMResponse(
                        model_id=model_ids[0],
                        completion="",
                        # sometimes returns empty because of safety block
                        stop_reason="safety",
                        api_duration=0.0,
                        duration=0.0,
                        cost=0.0,
                        logprobs=[],  # HACK no logprobs
                    )
                ]
                break

            except Exception as e:
                exc = e
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                await asyncio.sleep(1.5**i)

        if responses is None:
            if exc is not None:
                LOGGER.error(
                    f"Failed to get a response from the API after {max_attempts} attempts. Error: {exc}, prompt: {prompt=}, {model_ids=}"
                )
            raise RuntimeError(
                f"Failed to get a response from the API after {max_attempts} attempts. prompt: {prompt}, prompt: {prompt=}, {model_ids=}"
            )

        return responses
