from typing import Callable, Literal, Union
from evals.apis.inference.api import InferenceAPI
from evals.apis.inference.cache_manager import CacheManager
from evals.data_models.cache import LLMCache
from evals.data_models.inference import LLMParams, LLMResponse
from evals.data_models.messages import Prompt

from evals.data_models.cache import LLMCache
from evals.data_models.inference import LLMParams, LLMResponse
from evals.data_models.messages import Prompt
from evals.utils import load_json, save_json

from anyio import Path as AnyioPath
class CacheManager:
    def __init__(self, cache_dir: AnyioPath):
        self.cache_dir = cache_dir

    def get_cache_file(self, prompt: Prompt, params: LLMParams) -> Path:
        return self.cache_dir / params.model_hash() / f"{prompt.model_hash()}.json"

    async def maybe_load_cache(self, prompt: Prompt, params: LLMParams) -> LLMCache | None:
        cache_file = self.get_cache_file(prompt, params)
        if cache_file.exists():
            data = load_json(cache_file)
            return LLMCache.model_validate_json(data)
        return None

    async def save_cache(self, prompt: Prompt, params: LLMParams, responses: list[LLMResponse]):
        cache_file = self.get_cache_file(prompt, params)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        cache = LLMCache(prompt=prompt, params=params, responses=responses)
        data = cache.model_dump_json(indent=2)
        save_json(cache_file, data)


class CachedInferenceAPI:
    # Wraps the InferenceAPI class to cache responses easily
    def __init__(self, api: InferenceAPI, cache_path: Path | str):
        self.api: InferenceAPI = api
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
