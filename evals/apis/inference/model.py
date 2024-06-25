import json
import logging
from pathlib import Path
from typing import Protocol

from evals.data_models.inference import LLMResponse

LOGGER = logging.getLogger(__name__)


class InferenceAPIModel(Protocol):
    async def __call__(
        self,
        model_ids: list[str],
        prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        raise NotImplementedError

    @staticmethod
    def create_prompt_history_file(prompt: dict, model: str, prompt_history_dir: Path):
        return None
        # if prompt_history_dir is None:
        #     return None
        # filename = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]}_{hash(str(prompt))}.txt"
        # prompt_file = prompt_history_dir / model / filename
        # prompt_file.parent.mkdir(parents=True, exist_ok=True)
        # with open(prompt_file, "w") as f:
        #     json_str = json.dumps(prompt, indent=4)
        #     json_str = json_str.replace("\\n", "\n")
        #     f.write(json_str)

        # return prompt_file

    @staticmethod
    def add_response_to_prompt_file(prompt_file, responses):
        if prompt_file is None:
            return
        with open(prompt_file, "a") as f:
            f.write("\n\n======RESPONSE======\n\n")
            json_str = json.dumps([response.to_dict() for response in responses], indent=4)
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)
