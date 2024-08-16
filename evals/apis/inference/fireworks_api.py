import os
import time
from typing import Sequence

import httpx
from fireworks.client import AsyncFireworks
from fireworks.client.error import (
    BadGatewayError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)
from slist import Slist
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random

from evals.apis.inference.model import InferenceAPIModel
from evals.data_models.inference import LLMResponse
from evals.data_models.messages import ChatMessage, MessageRole, Prompt


def convert_to_llama(messages: Sequence[ChatMessage]) -> str:
    # new_messages = convert_assistant_if_completion_to_assistant(messages)

    maybe_system = (
        Slist(messages).filter(lambda x: x.role == MessageRole.system).map(lambda x: x.content).first_option or ""
    )
    out = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{maybe_system}<|eot_id|>"""
    for m in messages:
        if m.role == MessageRole.user:
            out += f"""<|start_header_id|>user<|end_header_id|>\n\n{m.content}<|eot_id|>"""

        elif m.role == MessageRole.assistant:
            out += f"""<|start_header_id|>assistant<|end_header_id|>\n\n{m.content}<|eot_id|>"""
        elif m.role == MessageRole.system:
            #           {{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + stop_token }}
            out += f"""<|start_header_id|>system<|end_header_id|>\n\n{m.content}<|eot_id|>"""
        else:
            raise ValueError(f"Unknown role: {m.role}")

    # if the last message is user, add the assistant tag
    if messages[-1].role == MessageRole.user:
        out += """<|start_header_id|>assistant<|end_header_id|>\n\n"""
    elif messages[-1].role == MessageRole.assistant:
        raise ValueError("Last message should not be assistant")
    #     out += f"""<|start_header_id|>assistant<|end_header_id|>\n\n"""
    return out


class FireworksModel(InferenceAPIModel):
    # adds the assistant to user side, doesn't owrk with llama
    def __init__(self):
        api_key = os.environ.get("FIREWORKS_API_KEY")
        self.client = AsyncFireworks(api_key=api_key)

    async def __call__(
        self,
        model_ids: list[str],
        prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        response = await self._make_api_call(prompt, model_ids[0], time.time(), **kwargs)
        if print_prompt_and_response:
            print(prompt)
            print(response[0].completion)
        return response

    @retry(
        # retry 5 times
        retry=retry_if_exception_type(
            (
                ServiceUnavailableError,
                BadGatewayError,
                InternalServerError,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.HTTPStatusError,
            )
        ),
        wait=wait_random(5, 20),
        stop=stop_after_attempt(30),
        reraise=True,
    )
    @retry(
        # retry 5 times
        retry=retry_if_exception_type((RateLimitError)),
        wait=wait_random(5, 15),
        stop=stop_after_attempt(99),
        reraise=True,
    )
    async def dep_make_api_call(self, prompt: Prompt, model_id, start_time, **params) -> list[LLMResponse]:
        api_start = time.time()
        new_params = params.copy()
        # remove seed, as it is not supported by the API
        new_params.pop("seed", None)
        _prompt = convert_to_llama(prompt.messages)
        # use completions instead of chat

        choices_resp = await self.client.completions.acreate(
            # messages=prompt.openai_format(),
            prompt=_prompt,
            model=model_id,
            stream=False,
            **new_params,
            # echo=True,
        )

        api_duration = time.time() - api_start
        duration = time.time() - start_time

        responses = [
            LLMResponse(
                model_id=model_id,
                completion=choice.text,
                stop_reason=choice.finish_reason,
                api_duration=api_duration,
                duration=duration,
                cost=0,
                logprobs=None,
            )
            for choice in choices_resp.choices
        ]
        return responses

    @retry(
        # retry 5 times
        retry=retry_if_exception_type((ServiceUnavailableError, BadGatewayError, InternalServerError)),
        wait=wait_random(5, 15),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    @retry(
        # retry 5 times
        retry=retry_if_exception_type((RateLimitError)),
        wait=wait_random(5, 15),
        stop=stop_after_attempt(99),
        reraise=True,
    )
    async def _make_api_call(self, prompt: Prompt, model_id, start_time, **params) -> list[LLMResponse]:
        api_start = time.time()
        new_params = params.copy()
        # remove seed, as it is not supported by the API
        new_params.pop("seed", None)
        choices_resp = await self.client.chat.completions.acreate(
            messages=prompt.openai_format(),
            model=model_id,
            stream=False,
            **new_params,
            # echo=True,
        )
        # resp_completion = resp.choices[0].message.content

        api_duration = time.time() - api_start
        duration = time.time() - start_time

        responses = [
            LLMResponse(
                model_id=model_id,
                completion=choice.message.content,
                stop_reason=choice.finish_reason,
                api_duration=api_duration,
                duration=duration,
                cost=0,
                logprobs=None,
            )
            for choice in choices_resp.choices
        ]
        return responses
