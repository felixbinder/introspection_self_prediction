import string
from typing import List


def apply_all_cleaning(s: str) -> str:
    """
    Apply all cleaning functions to the input string.

    Parameters:
    s (str): The input string to be cleaned.

    Returns:
    str: The cleaned string.
    """
    s = s.lower()
    s = strip_punctuation(s)
    s = strip_newlines(s)
    s = strip_multiple_whitespaces(s)
    s = strip_common_prefixes(s)
    return s


def clean_string(s: str) -> str:
    """
    Clean the input string by stripping punctuation, converting to lower case, and removing leading/trailing spaces.

    Parameters:
    s (str): The input string to be cleaned.

    Returns:
    str: The cleaned string.
    """
    s = str(s)
    return strip_punctuation(s.lower().strip().lstrip())


def strip_punctuation(s: str) -> str:
    """
    Remove punctuation from the input string.

    Parameters:
    s (str): The input string.

    Returns:
    str: The string with punctuation removed.
    """
    s = str(s)
    return "".join(c for c in s if c not in string.punctuation)


def strip_newlines(s: str) -> str:
    """
    Replace newline and carriage return characters with spaces in the input string.

    Parameters:
    s (str): The input string.

    Returns:
    str: The string with newline and carriage return characters replaced by spaces.
    """
    s = str(s)
    return s.replace("\n", " ").replace("\r", " ").replace("\t", " ")


def strip_multiple_whitespaces(s: str) -> str:
    """
    Replace multiple whitespace characters with a single space in the input string.

    Parameters:
    s (str): The input string.

    Returns:
    str: The string with multiple whitespaces replaced by a single space.
    """
    s = str(s)
    return " ".join(s.split())


def strip_common_prefixes(s: str) -> str:
    """
    Remove common prefixes from the input string.
    This is to remove cruft from the base models responses.

    Parameters:
    s (str): The input string.

    Returns:
    str: The string with common prefixes removed.
    """
    s = str(s)
    prefixes: List[str] = ["Answer:", "Answer :", "Response:", "Response :", "Assistant:", "Assistant :"]
    for prefix in prefixes:
        if s.lower().startswith(prefix.lower()):
            return s[len(prefix) :]
    return s


def extract_first_of_multiple_responses(response: str, join_on: str) -> str:
    """
    Sometimes, the continuation model will return multiple responses. This function extracts the first response.
    """
    response = str(response)
    try:
        if join_on == "":
            responses = [[c] for c in response]
        else:
            responses = response.split(join_on)
        responses = [r for r in responses if r.strip() != ""]
        return responses[0]
    except AttributeError:
        return response
    except IndexError:  # response is empty
        return response


def match_log_probs_to_trimmed_response(response, logprobs):
    """
    When we are stripping words from the beginning of the string, we also need to discard the corresponding logprobs.

    Parameters:
    string (str): The input string.
    logprobs (str): The log probabilities of the words in the string.

    Returns:
    logprobs (str): The log probabilities of the words in the string.
    """
    try:
        response = str(response)
        if logprobs is None or isinstance(logprobs, float):  # ie is it nan?
            # we don't have logprobs
            return logprobs
        if isinstance(logprobs, str):
            logprobs = eval(logprobs)
        index = 0
        while index < len(logprobs):
            logprob = logprobs[index]
            # get most likely token
            token = max(logprob, key=logprob.get)
            if response.lower().startswith(token.lower()):
                return str(logprobs[index:])
            index += 1
    except Exception:
        pass
    return None
