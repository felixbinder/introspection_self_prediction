from pathlib import Path
from typing import Callable, Literal, Union
from evals.apis.inference.api import InferenceAPI
from evals.apis.inference.cache_manager import CacheManager
from evals.data_models.cache import LLMCache
from evals.data_models.inference import LLMParams, LLMResponse
from evals.data_models.messages import Prompt


class CachedInferenceAPI:
    def __init__(self, api: InferenceAPI, cache_path: Path | str):
        self.api = api
        self.cache_manager = CacheManager(Path(cache_path))

    async def __call__(
        self,
        model_ids: Union[str, list[str]],
        prompt: Prompt,
        max_tokens: int | None = None,
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 10,
        num_candidates_per_completion: int = 1,
        is_valid: Callable[[str], bool] = lambda _: True,
        insufficient_valids_behaviour: Literal["error", "continue", "pad_invalids"] = "error",
        seed: int = 0,
        **kwargs,
    ) -> list[LLMResponse]:
        params = LLMParams(
            model=model_ids,
            max_tokens=max_tokens,
            n=n,
            num_candidates_per_completion=num_candidates_per_completion,
            insufficient_valids_behaviour=insufficient_valids_behaviour,
            seed=seed,
            **kwargs,
        )
        maybe_cached_responses: LLMCache | None = self.cache_manager.maybe_load_cache(prompt=prompt, params=params)
        if maybe_cached_responses is not None:
            if maybe_cached_responses.responses:
                return maybe_cached_responses.responses

        responses = await self.api(
            model_ids=model_ids,
            prompt=prompt,
            max_tokens=max_tokens,  # type: ignore
            print_prompt_and_response=print_prompt_and_response,
            n=n,
            max_attempts_per_api_call=max_attempts_per_api_call,
            num_candidates_per_completion=num_candidates_per_completion,
            is_valid=is_valid,
            insufficient_valids_behaviour=insufficient_valids_behaviour,
            seed=seed,
            **kwargs,
        )
        self.cache_manager.save_cache(prompt=prompt, params=params, responses=responses)
        return responses
