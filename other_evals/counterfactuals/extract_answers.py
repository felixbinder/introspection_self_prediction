import re
from typing import Literal, Optional


def extract_answer_non_cot(
    response: str,
) -> Optional[str]:
    response = response.strip().replace("The best answer is: (", "")

    pattern = re.compile(r"^\(?([a-zA-Z\d]+)\)?")
    match = pattern.match(response)
    if match:
        candidate_ans = match.group(1)
        if candidate_ans:
            if candidate_ans in ["A", "B", "C", "D", "E", "F", "G", "H"]:
                return candidate_ans
    return None


def extract_yes_or_no(
    response: str,
) -> Literal["Y", "N"] | None:
    cleaned_response = response.strip().replace("\n", " ").lower()
    if cleaned_response == "y":
        return "Y"
    if cleaned_response == "n":
        return "N"
    return None


def extract_true_or_false(
    response: str,
) -> bool | None:
    cleaned_response = response.strip().replace("\n", " ").lower()
    if cleaned_response == "true":
        return True
    if cleaned_response == "false":
        return False
    return None
