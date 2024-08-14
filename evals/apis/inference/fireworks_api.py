import os
import time

from fireworks.client import AsyncFireworks

from evals.apis.inference.model import InferenceAPIModel
from evals.data_models.inference import LLMResponse
from evals.data_models.messages import Prompt


class FireworksModel(InferenceAPIModel):
    # adds the assistant to user side, doesn't owrk with llama
    def __init__(self):
        api_key = os.environ.get("FIREWORKS_API_KEY")
        self.client = AsyncFireworks(api_key=api_key)

    # # @retry(exceptions=(httpx.ConnectError), tries=10, delay=2)
    # # @retry(exceptions=(RateLimitError, ServiceUnavailableError, httpx.HTTPStatusError), tries=-1, delay=1)
    # async def call(
    #     self,
    #     messages: Sequence[ChatMessage],
    #     config: OpenaiInferenceConfig,
    #     try_number: int = 1,
    # ) -> InferenceResponse:
    #     # new_messages = [{m.role.value: m.content} for m in messages]
    #     new_messages = to_fireworks_format(messages)

    #     inf_response = InferenceResponse(raw_responses=[resp_completion])
    #     return inf_response

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

    # @retry(
    #     # retry 5 times
    #     retry=retry_if_exception_type(
    #         (openai.error.RateLimitError, openai.error.APIError, openai.error.APIConnectionError)
    #     ),
    #     wait=wait_fixed(5),
    #     stop=stop_after_attempt(5),
    #     reraise=True,
    # )
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
