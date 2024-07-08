from typing import Literal

from pydantic import BaseModel


class ObjectAndMeta(BaseModel):
    task: str
    string: str
    meta_predicted_correctly: bool
    meta_response: str
    response_property: str
    meta_model: str
    object_model: str
    object_response_property_answer: str
    object_response_raw_response: str
    object_complied: bool
    meta_complied: bool
    shifted: Literal["shifted", "same", "not_compliant", "not_calculated"]
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
            new.response_property = "is_one_of_given_options"
        if "is_either_b_or_d" in self.response_property:
            new.response_property = "is_one_of_given_options"

        return new

    @property
    def mode_is_correct(self) -> bool:
        return self.object_response_property_answer == self.modal_response_property_answer
