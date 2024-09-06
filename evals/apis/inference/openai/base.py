import asyncio
import logging
import time
from pathlib import Path
from traceback import format_exc
from typing import Optional

from evals.apis.inference.model import InferenceAPIModel
from evals.data_models.inference import LLMResponse
from evals.utils import COMPLETION_MODELS

LOGGER = logging.getLogger(__name__)


class Resource:
    """
    A resource that is consumed over time and replenished at a constant rate.
    """

    def __init__(self, refresh_rate, total=0, throughput=0):
        self.refresh_rate = refresh_rate
        self.total = total
        self.throughput = throughput
        self.last_update_time = time.time()
        self.start_time = time.time()
        self.value = self.refresh_rate

    def _replenish(self):
        """
        Updates the value of the resource based on the time since the last update.
        """
        curr_time = time.time()
        self.value = min(
            self.refresh_rate,
            self.value + (curr_time - self.last_update_time) * self.refresh_rate / 60,
        )
        self.last_update_time = curr_time
        self.throughput = self.total / (curr_time - self.start_time) * 60

    def geq(self, amount: float) -> bool:
        self._replenish()
        return self.value >= amount

    def consume(self, amount: float):
        """
        Consumes the given amount of the resource.
        """
        assert self.geq(amount), f"Resource does not have enough capacity to consume {amount} units"
        self.value -= amount
        self.total += amount


class OpenAIModel(InferenceAPIModel):
    def __init__(
        self,
        frac_rate_limit: float,
        organization: str,
        prompt_history_dir: Path = None,
    ):
        self.frac_rate_limit = frac_rate_limit
        self.organization = organization
        self.prompt_history_dir = prompt_history_dir
        self.model_ids = set()

        self.token_capacity = dict()
        self.request_capacity = dict()
        self.lock_add = asyncio.Lock()
        self.lock_consume = asyncio.Lock()

    @staticmethod
    def _assert_valid_id(model_id: str):
        raise NotImplementedError

    @staticmethod
    async def _get_dummy_response_header(model_id: str):
        raise NotImplementedError

    @staticmethod
    def _count_prompt_token_capacity(prompt, **kwargs) -> int:
        raise NotImplementedError

    async def _make_api_call(self, prompt, model_id, **params) -> list[LLMResponse]:
        raise NotImplementedError

    @staticmethod
    def _print_prompt_and_response(prompt, responses):
        raise NotImplementedError

    async def add_model_id(self, model_id: str):
        self._assert_valid_id(model_id)
        if model_id in self.model_ids:
            return

        self.model_ids.add(model_id)

        # make dummy request to get token and request capacity
        model_metadata = await self._get_dummy_response_header(model_id)
        token_capacity = int(model_metadata["x-ratelimit-limit-tokens"])
        request_capacity = int(model_metadata["x-ratelimit-limit-requests"])
        print(f"got capacities for model {model_id}: {token_capacity}, {request_capacity}")
        tokens_consumed = token_capacity - int(model_metadata["x-ratelimit-remaining-tokens"])
        requests_consumed = request_capacity - int(model_metadata["x-ratelimit-remaining-requests"])
        print(f"consumed capacities for model {model_id}: {tokens_consumed}, {requests_consumed}")
        token_cap = token_capacity * self.frac_rate_limit
        request_cap = request_capacity * self.frac_rate_limit
        if model_id in COMPLETION_MODELS:
            token_cap *= 10000  # openai does not track token limit so we can increase it

        print(f"setting cap for model {model_id}: {token_cap}, {request_cap}")
        token_capacity = Resource(token_cap)
        request_capacity = Resource(request_cap)
        token_capacity.consume(min(token_cap, tokens_consumed))
        request_capacity.consume(min(request_cap, requests_consumed))
        self.token_capacity[model_id] = token_capacity
        self.request_capacity[model_id] = request_capacity

    async def __call__(
        self,
        model_ids: list[str],
        prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()

        assert len(model_ids) == 1, "Only one model_id is supported for now"
        model_id = model_ids[0]
        responses: Optional[list[LLMResponse]] = None
        for i in range(max_attempts):
            try:
                start = time.time()
                responses = await self._make_api_call(prompt, model_id, start, **kwargs)
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                await asyncio.sleep(1.5**i)
            else:
                break

        if responses is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        if print_prompt_and_response:
            prompt.pretty_print(responses)

        end = time.time()
        LOGGER.debug(f"Completed call to {model_id} in {end - start}s.")
        return responses
