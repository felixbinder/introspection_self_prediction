import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self, Sequence, Type, TypeVar, Union

import anthropic
import anyio
import openai
import openai.error
from pydantic import BaseModel
from slist import Slist
from tenacity import retry as async_retry
from tenacity import retry_if_exception_type, wait_fixed

from evals.data_models.hashable import deterministic_hash

logger = logging.getLogger(__name__)


def raise_should_not_happen() -> None:
    raise ValueError("Should not happen")


GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)


def caught_base_model_parse(basemodel: Type[GenericBaseModel], line: str) -> GenericBaseModel:
    try:
        return basemodel.model_validate_json(line)
    except Exception as e:
        print(f"Error parsing line: {line}")
        raise e


def read_jsonl_file_into_basemodel(path: Path | str, basemodel: Type[GenericBaseModel]) -> Slist[GenericBaseModel]:
    with open(path) as f:
        return Slist(caught_base_model_parse(basemodel=basemodel, line=line) for line in f.readlines())


def write_jsonl_file_from_basemodel(path: Path | str, basemodels: Sequence[BaseModel]) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for basemodel in basemodels:
            f.write(basemodel.model_dump_json() + "\n")


class ChatMessageV2(BaseModel):
    role: str
    content: str

    def pretty_str(self) -> str:
        return f"{self.role}: {self.content}"


def display_conversation(messages: Sequence[ChatMessageV2]) -> str:
    return "\n".join([msg.pretty_str() for msg in messages])


def display_multiple_conversations(messages: Sequence[Sequence[ChatMessageV2]]) -> str:
    # use ================== to separate conversations
    separator = "\n" + "=" * 20 + "\n"
    return separator.join([display_conversation(messages) for messages in messages])


def dump_conversations(path: Path | str, messages: Sequence[Sequence[ChatMessageV2]]) -> None:
    print(f"Dumping conversations to {path}")
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        string = display_multiple_conversations(messages)
        f.write(string)


class InferenceConfig(BaseModel):
    # Config for openai
    model: str
    temperature: float
    top_p: float | None
    max_tokens: int
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1  # number of responses to get back
    stop: Union[None, str, list[str]] = None  # type: ignore

    def model_hash(self) -> str:
        """
        Returns a hash of a stringified version of the entire model config
        """
        return deterministic_hash(self.model_dump_json())


class InferenceResponse(BaseModel):
    raw_responses: Sequence[str]
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    @property
    def has_multiple_responses(self) -> bool:
        return len(self.raw_responses) > 1

    @property
    def single_response(self) -> str:
        if len(self.raw_responses) != 1:
            raise ValueError(f"Expected exactly one response, got {len(self.raw_responses)}")

        else:
            return self.raw_responses[0]


class ModelCallerV2(ABC):
    @abstractmethod
    async def call(
        self,
        messages: Sequence[ChatMessageV2],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> InferenceResponse:
        raise NotImplementedError()

    def with_file_cache(self, cache_path: Path | str) -> "CachedCallerV2":
        """
        Load a file cache from a path
        Alternatively, rather than write_every_n, just dump with append mode?
        """
        if isinstance(cache_path, str):
            cache_path = Path(cache_path)

        return CachedCallerV2(wrapped_caller=self, cache_path=cache_path)


class ClaudeCaller(ModelCallerV2):
    def __init__(self, anthropic_client: anthropic.AsyncAnthropic | None = None):
        anthropic_client = (
            anthropic.AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
            if anthropic_client is None
            else anthropic_client
        )
        self.anthropic_client = anthropic_client

    async def call(
        self,
        messages: Sequence[ChatMessageV2],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> InferenceResponse:
        assert len(messages) >= 1
        system_prompts = [item for item in messages if item.role == "system"]
        assert len(system_prompts) <= 1, f"Expected at most one system prompt, got {system_prompts}"
        non_system_prompts = [item for item in messages if item.role != "system"]
        first_sytem_prompt: ChatMessageV2 | None = system_prompts[0] if system_prompts else None
        try:
            message = await self.anthropic_client.messages.create(
                system=first_sytem_prompt.content if first_sytem_prompt else anthropic.NOT_GIVEN,
                max_tokens=config.max_tokens,
                # stop=config.stop,
                # n=config.n,
                temperature=config.temperature,
                # frequency_penalty=config.frequency_penalty,
                messages=[
                    {
                        "role": m.role,
                        "content": m.content,
                    }
                    for m in non_system_prompts  # type: ignore
                ],
                model=config.model,
            )
        except anthropic.BadRequestError as e:
            error_message = e.message
            if "Output blocked by content filtering policy" in error_message:
                print("Content filtering policy blocked output, returning empty response")
                return InferenceResponse(raw_responses=[], error=error_message)
            else:
                raise e

        outputs = message.content
        responses = [output.text for output in outputs]
        if len(responses) == 0:
            raise ValueError(
                f"No responses returned from Claude using model {config.model}, messages: {messages}, got {message=}"
            )
        return InferenceResponse(raw_responses=responses, error=None)


class OpenAICaller(ModelCallerV2):
    @async_retry(
        retry=retry_if_exception_type((openai.error.RateLimitError, openai.error.APIError)), wait=wait_fixed(5)
    )
    async def call(
        self,
        messages: Sequence[ChatMessageV2],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> InferenceResponse:
        assert len(messages) >= 1

        try:
            result = await openai.ChatCompletion.acreate(  # type: ignore
                model=config.model,
                messages=[chat.model_dump() for chat in messages],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                presence_penalty=config.presence_penalty,
                frequency_penalty=config.frequency_penalty,
                top_p=config.top_p,
                n=config.n,
                stream=False,
                stop=[config.stop] if isinstance(config.stop, str) else config.stop,
            )
            choices = result["choices"]  # type: ignore
            completions = [choice["message"]["content"] for choice in choices]
            return InferenceResponse(raw_responses=completions, error=None)
        except openai.error.OpenAIError as e:
            if "Failed to create completion as the model generated invalid Unicode output." in str(e.user_message):
                return InferenceResponse(raw_responses=[], error=e.user_message)
            else:
                raise e


class UniversalCallerV2(ModelCallerV2):
    def __init__(self):
        self.claude_caller = ClaudeCaller()
        self.gpt4_caller = OpenAICaller()

    async def call(
        self,
        messages: Sequence[ChatMessageV2],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> InferenceResponse:
        if "claude" in config.model:
            return await self.claude_caller.call(messages, config)
        elif "gpt-" in config.model:
            return await self.gpt4_caller.call(messages, config)
        else:
            raise ValueError(f"Unknown model {config.model}")


class CachedValue(BaseModel):
    response: InferenceResponse
    messages: Sequence[ChatMessageV2]
    config: InferenceConfig


class FileCacheRow(BaseModel):
    key: str
    response: CachedValue


class APIRequestCache:
    def __init__(self, cache_path: Path | str):
        self.cache_path = Path(cache_path)
        self.data: dict[str, CachedValue] = {}
        self.load()
        self.opened_files: dict[Path, anyio.AsyncFile[str]] = {}
        self.lock = anyio.create_lock()

    def load(self, silent: bool = False) -> Self:
        """
        Load a file cache from a path
        """

        if self.cache_path.exists():
            rows = read_jsonl_file_into_basemodel(
                path=self.cache_path,
                basemodel=FileCacheRow,
            )
            logger.info(f"Loaded {len(rows)} rows from cache file {self.cache_path.as_posix()}")
            self.data = {row.key: row.response for row in rows}
        return self

    def save(self) -> None:
        """
        Save a file cache to a path
        """
        rows = [FileCacheRow(key=key, response=response) for key, response in self.data.items()]
        write_jsonl_file_from_basemodel(self.cache_path, rows)

    async def save_single(self, key: str, response: CachedValue) -> None:
        """
        Save a single key to the cache
        """
        row = FileCacheRow(key=key, response=response)
        # try to open the file if it's not already open
        if self.cache_path not in self.opened_files:
            self.opened_files[self.cache_path] = await anyio.open_file(self.cache_path, "a")
        opened_file: anyio.AsyncFile[str] = self.opened_files[self.cache_path]
        # hold the lock while writing
        async with self.lock:
            await opened_file.write(row.model_dump_json() + "\n")

    def __getitem__(self, key: str) -> CachedValue:
        return self.data[key]

    def __setitem__(self, key: str, value: CachedValue) -> None:
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __delitem__(self, key: str) -> None:
        del self.data[key]


def file_cache_key(messages: Sequence[ChatMessageV2], config: InferenceConfig) -> str:
    str_messages = ",".join([str(msg) for msg in messages]) + config.model_hash()
    return deterministic_hash(str_messages)


class CachedCallerV2(ModelCallerV2):
    def __init__(self, wrapped_caller: ModelCallerV2, cache_path: Path):
        self.model_caller = wrapped_caller
        self.cache_path = cache_path
        # create folder if it doesn't exist
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: APIRequestCache = APIRequestCache(cache_path)

    async def call(
        self,
        messages: Sequence[ChatMessageV2],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> InferenceResponse:
        key_without_retry = file_cache_key(messages, config)
        # only add retry number to key if try_number > 1 for backwards compatibility
        key = key_without_retry if try_number == 1 else f"{key_without_retry}_try_{try_number}"
        if key in self.cache:
            return self.cache[key].response
        else:
            response = await self.model_caller.call(messages, config)
            value = CachedValue(
                response=response,
                messages=messages,
                config=config,
            )

            self.cache[key] = value
            await self.cache.save_single(key, value)
            return response