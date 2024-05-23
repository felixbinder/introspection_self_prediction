import tiktoken

COMPLETION_MODELS = {
    "davinci-002",
    "babbage-002",
    "text-davinci-003",
    "text-davinci-002",
    "gpt-4-base",
    "gpt-3.5-turbo-instruct",
}

_GPT_4_MODELS = [
    "gpt-4",
    "gpt-4o",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
]
_GPT_TURBO_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
]
GPT_CHAT_MODELS = set(_GPT_4_MODELS + _GPT_TURBO_MODELS)


def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def price_per_token(model_id: str) -> tuple[float, float]:
    """
    Returns the (input token, output token) price for the given model id.
    """
    if model_id == "gpt-4-1106-preview":
        prices = 0.01, 0.03
    elif model_id == "gpt-3.5-turbo-1106":
        prices = 0.001, 0.002
    elif model_id.startswith("gpt-4"):
        prices = 0.03, 0.06
    elif model_id.startswith("gpt-4-32k"):
        prices = 0.06, 0.12
    elif model_id.startswith("gpt-3.5-turbo-16k"):
        prices = 0.003, 0.004
    elif model_id.startswith("gpt-3.5-turbo"):
        prices = 0.0015, 0.002
    elif model_id == "davinci-002":
        prices = 0.002, 0.002
    elif model_id == "babbage-002":
        prices = 0.0004, 0.0004
    elif model_id == "text-davinci-003" or model_id == "text-davinci-002":
        prices = 0.02, 0.02
    elif "ft:gpt-3.5-turbo" in model_id:
        prices = 0.012, 0.016
    elif "ft:gpt-4" in model_id:
        prices = 0.0450, 0.0900
    else:
        raise ValueError(f"Invalid model id: {model_id}")

    return tuple(price / 1000 for price in prices)
