import os

import openai

from evals.utils import setup_environment

setup_environment()

runpod_api_key = os.environ.get("RUNPOD_API_KEY")
assert runpod_api_key is not None, "Please set RUNPOD_API_KEY in your environment variables"

openai.api_base = "https://api.runpod.ai/v2/vllm-zhwh1a3byiiklv/openai/v1"
openai.api_key = runpod_api_key


async def test():
    result = await openai.ChatCompletion.acreate(  # type: ignore
        model="tomekkorbak/introspection-test2",
        messages=[{"role": "user", "content": "Why is google's gemini so janky"}],
        max_tokens=1000,
    )

    for choice in result["choices"]:
        print(choice["message"]["content"])


if __name__ == "__main__":
    import asyncio

    asyncio.run(test())
