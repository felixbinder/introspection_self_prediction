import os

import openai

from evals.utils import setup_environment

setup_environment()

# f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/openai/v1"
runpod_api_key = os.environ.get("RUNPOD_API_KEY")
assert runpod_api_key is not None, "Please set RUNPOD_API_KEY in your environment variables"

openai.api_base = "https://api.runpod.ai/v2/vllm-xvb88yc7q0u9r4/openai/v1"
openai.api_key = runpod_api_key


async def test():
    result = await openai.ChatCompletion.acreate(  # type: ignore
        model="meta-llama/Meta-Llama-3-8B",
        messages=[{"role": "user", "content": "Why is google's gemini so janky"}],
        max_tokens=1000,
    )
    choices = result["choices"]  # type: ignore
    for choice in choices:
        print(choice["message"]["content"])


if __name__ == "__main__":
    import asyncio

    asyncio.run(test())
