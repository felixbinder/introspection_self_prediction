from evals.apis.inference.model import InferenceAPIModel
from evals.data_models.inference import LLMResponse


class TestModel(InferenceAPIModel):
    async def __call__(
        self,
        model_ids: list[str],
        prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        first_model_id = model_ids[0]
        assert first_model_id == "test", f"Model ID {first_model_id} is not supported by this API"

        return [
            LLMResponse(
                model_id=first_model_id,
                completion="A",
                stop_reason="max_tokens",
                api_duration=0,
                duration=0,
                cost=0,
                logprobs=None,
            )
        ]
