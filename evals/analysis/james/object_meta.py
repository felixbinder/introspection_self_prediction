from typing import Literal

from pydantic import BaseModel


class ObjectAndMeta(BaseModel):
    # For James' analysis
    task: str
    string: str
    meta_predicted_correctly: bool | None
    meta_response: str | None
    response_property: str
    meta_model: str
    object_model: str
    object_response_property_answer: str | None
    object_response_raw_response: str
    object_complied: bool
    meta_complied: bool
    shifted: Literal["shifted", "same", "not_compliant", "not_calculated"]
    before_shift_raw: str | None
    before_shift_ans: str | None
    after_shift_raw: str | None
    after_shift_ans: str | None
    object_prompt: str
    meta_prompt: str

    modal_response_property_answer: str

    @property
    def is_predicting_mode(self) -> bool:
        return self.meta_response == self.modal_response_property_answer

    def rename_properties(self):
        # if "matches" in self.response_property, rename to "matches behavior"
        new = self.model_copy()
        if "matches" in self.response_property:
            new.response_property = "matches behavior"
        if "is_either_a_or_c" in self.response_property:
            new.response_property = "one_of_options"
        if "is_either_b_or_d" in self.response_property:
            new.response_property = "one_of_options"

        return new

    @property
    def mode_is_correct(self) -> bool:
        return self.object_response_property_answer == self.modal_response_property_answer
