import asyncio
import unittest

from evals.apis.inference.api import InferenceAPI
from evals.data_models.messages import Prompt
from evals.utils import setup_environment


class AsyncInferenceAPITestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        setup_environment()
        self.api = InferenceAPI()

        prompt = [
            {"role": "system", "content": "You are a swashbuckling space-faring voyager."},
            {"role": "user", "content": "Hello!"},
        ]
        self.prompt = Prompt(messages=prompt)

        prompt_with_assistant_message = [
            {"role": "user", "content": "Please continue this sequence up to 10: 1, 2, 3, 4, "},
            {"role": "assistant", "content": "5, 6, 7, "},
        ]
        self.prompt = Prompt(messages=prompt)

        self.prompt_with_assistant_message = Prompt(messages=prompt_with_assistant_message)

        prompt_with_only_none = [
            {"role": "none", "content": "whence afterever the storm"},
        ]
        self.prompt_with_only_none = Prompt(messages=prompt_with_only_none)
        self.kwargs = {"max_attempts_per_api_call": 1, "print_prompt_and_response": True, "temperature": 1}

    async def test_anthropic_api(self):
        responses = await self.api("claude-2.1", prompt=self.prompt, **self.kwargs)
        self.assertIsInstance(responses[0].completion, str)

    async def test_old_anthropic_api(self):
        responses = await self.api("claude-instant-1", prompt=self.prompt, **self.kwargs)
        self.assertIsInstance(responses[0].completion, str)

    async def test_openai_chat_api(self):
        responses = await self.api(["gpt-3.5-turbo", "gpt-3.5-turbo-0613"], prompt=self.prompt, n=6, **self.kwargs)
        self.assertEqual(len(responses), 6)

    async def test_openai_completion_api(self):
        responses = await self.api("gpt-3.5-turbo-instruct", prompt=self.prompt_with_only_none, **self.kwargs)
        self.assertIsInstance(responses[0].completion, str)

    async def test_openai_chat_error_for_last_assistant_message(self):
        with self.assertRaises((ValueError, RuntimeError)):
            await self.api("gpt-3.5-turbo", prompt=self.prompt_with_assistant_message, **self.kwargs)

    async def test_openai_anthropic_for_last_assistant_message(self):
        responses = await self.api(
            "claude-2.1", prompt=self.prompt_with_assistant_message, temperature=0, max_tokens=2, **self.kwargs
        )
        self.assertIn("8", responses[0].completion)

    async def test_openai_chat_error_for_none_message(self):
        with self.assertRaises((ValueError, RuntimeError)):
            await self.api("gpt-3.5-turbo", prompt=self.prompt_with_only_none, **self.kwargs)

    async def test_anthropic_chat_error_for_none_message(self):
        with self.assertRaises((ValueError, RuntimeError)):
            await self.api("claude-2.1", prompt=self.prompt_with_only_none, **self.kwargs)

    async def test_gemini_api(self):
        responses = await self.api("gemini-1.0-pro-002", prompt=self.prompt, **self.kwargs)
        self.assertIsInstance(responses[0].completion, str)

    async def test_gemini_api_ratelimit(self):
        responses = await asyncio.gather(
            *[self.api("gemini-1.0-pro-002", prompt=self.prompt, **self.kwargs) for _ in range(100)]
        )
        self.assertIsInstance(responses[0][0].completion, str)

    async def test_is_valid_fails(self):
        with self.assertRaises(RuntimeError):
            await self.api(
                "gpt-3.5-turbo",
                prompt=self.prompt,
                n=1,
                is_valid=lambda _: False,
                insufficient_valids_behaviour="error",
                **self.kwargs,
            )

    async def test_is_valid_pad(self):
        responses = await self.api(
            "gpt-3.5-turbo",
            prompt=self.prompt,
            n=1,
            is_valid=lambda _: False,
            insufficient_valids_behaviour="pad_invalids",
            num_candidates_per_completion=2,
            **self.kwargs,
        )
        self.assertEqual(len(responses), 1)

    async def test_is_valid_continue(self):
        responses = await self.api(
            "gpt-3.5-turbo",
            prompt=self.prompt,
            n=1,
            is_valid=lambda _: False,
            insufficient_valids_behaviour="continue",
            num_candidates_per_completion=1,
            **self.kwargs,
        )
        self.assertEqual(len(responses), 0)


if __name__ == "__main__":
    unittest.main()
