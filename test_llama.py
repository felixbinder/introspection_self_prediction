import fireworks.client

fireworks.client.api_key = "your-key"
completion = fireworks.client.Completion.create(
    "accounts/fireworks/models/llama-v2-7b", "Once upon a time", temperature=0.1, n=2, max_tokens=16
)
print(completion)
