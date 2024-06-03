from pydantic import BaseModel


class OtherEvalCSVFormat(BaseModel):
    object_history: str
    object_model: str
    object_parsed_result: str
    meta_history: str
    meta_model: str
    meta_parsed_result: str
    meta_predicted_correctly: bool
    eval_name: str


class FinetuneMessage(BaseModel):
    role: str
    content: str


class FinetuneConversation(BaseModel):
    # Each conversation has multiple messages between the user and the model
    messages: list[FinetuneMessage]

    @property
    def last_message_content(self) -> str:
        return self.messages[-1].content
