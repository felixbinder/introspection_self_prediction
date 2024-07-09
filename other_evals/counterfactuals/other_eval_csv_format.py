from pydantic import BaseModel

from evals.analysis.james.object_meta import ObjectAndMeta


class OtherEvalCSVFormat(BaseModel):
    original_prompt: str = ""
    object_history: str
    object_model: str
    object_parsed_result: str | None
    meta_history: str
    meta_model: str
    meta_parsed_result: str | None
    meta_predicted_correctly: bool | None
    eval_name: str

    def to_james_analysis_format(self) -> ObjectAndMeta:
        return ObjectAndMeta(
            task=self.eval_name,
            string=self.original_prompt,
            meta_predicted_correctly=self.meta_predicted_correctly,
            meta_response=self.meta_parsed_result,
            response_property=self.eval_name,
            meta_model=self.meta_model,
            object_model=self.object_model,
            object_response_property_answer=self.object_parsed_result,
            object_response_raw_response=self.object_history,
            object_complied=True if self.object_parsed_result else False,
            meta_complied=True if self.meta_parsed_result else False,
            shifted="not_calculated",
            modal_response_property_answer="todo",
        )


class FinetuneMessage(BaseModel):
    role: str
    content: str


class FinetuneConversation(BaseModel):
    # Each conversation has multiple messages between the user and the model
    messages: list[FinetuneMessage]

    @property
    def last_message_content(self) -> str:
        return self.messages[-1].content
