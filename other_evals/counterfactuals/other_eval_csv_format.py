from pydantic import BaseModel


class OtherEvalCSVFormat(BaseModel):
    object_prompt: str
    object_model: str
    object_parsed_result: str
    meta_prompt: str
    meta_model: str
    meta_parsed_result: str
    meta_predicted_correctly: bool
    eval_name: str
