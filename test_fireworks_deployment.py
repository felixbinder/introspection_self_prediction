from typing import Sequence
from fireworks.client import Fireworks
from slist import Slist

from evals.data_models.messages import ChatMessage, MessageRole

# client = Fireworks(api_key="test key")
# response = client.chat.completions.create(
#   model="accounts/chuajamessh-b7a735/models/llama-8b-14aug-20k",
#     # model ="accounts/chuajamessh-b7a735/models/llama-70b-14aug-5k",
#     # model="accounts/fireworks/models/llama-v3p1-8b-instruct",
#   messages=[
#     {"role": "system", "content": ""},
#     {
#     "role": "user",
#     "content": "What is 1 + 2",
#   },
#   {
#     "role": "assistant",
#     "content": ""
#     # "content": "What is the next 5 animals in the following text? Respond only with the next 5 animals and nothing else, including punctuation.\nmouse sheep lion duck chicken mouse horse ",
#   }
#   ],
#   max_tokens=20,
#   temperature=0,
#   echo=True,
# )

def to_fireworks_completion_llama(messages: Sequence[ChatMessage]) -> str:
    # new_messages = convert_assistant_if_completion_to_assistant(messages)
    
    maybe_system = Slist(messages).filter(lambda x: x.role == MessageRole.system).map(lambda x: x.content).first_option or ""
    out = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{maybe_system}<|eot_id|>"""
    for m in messages:
        if m.role == MessageRole.user:
            out += f"""<|start_header_id|>user<|end_header_id|>\n\n{m.content}<|eot_id|>"""

        elif m.role == MessageRole.assistant:
            out += f"""<|start_header_id|>assistant<|end_header_id|>\n\n{m.content}<|eot_id|>"""
        else:
            raise ValueError(f"Unknown role: {m.role}")

    # if the last message is user, add the assistant tag
    if messages[-1].role == MessageRole.user:
        out += f"""<|start_header_id|>assistant<|end_header_id|>\n\n"""
    elif messages[-1].role == MessageRole.assistant:
        raise ValueError("Last message should not be assistant")
    #     out += f"""<|start_header_id|>assistant<|end_header_id|>\n\n"""
    return out

message = [ChatMessage(role=MessageRole.user, content="What is the next 5 animals in the following text? Respond only with the next 5 animals and nothing else, including punctuation.\nmouse sheep lion duck chicken mouse horse ")]
prompt = to_fireworks_completion_llama(message)
response = client.completions.create(
#   model="accounts/chuajamessh-b7a735/models/llama-8b-14aug-20k",
# model="accounts/chuajamessh-b7a735/models/llama-70b-14aug-5k",
    # model ="accounts/chuajamessh-b7a735/models/llama-70b-14aug-5k",
    # model="accounts/fireworks/models/llama-v3p1-8b-instruct",
  prompt=prompt,
  max_tokens=20,
  temperature=0,
#   echo=True,
)


print(response.choices[0].text)
