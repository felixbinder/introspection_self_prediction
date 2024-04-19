"""This file holds functions to extract a property of a response of the language model. The functions take as input a single row of the response dataframe."""

from typing import Callable

import pandas as pd


def identity(row: pd.Series) -> str:
    """Used for prediction."""
    return row["response"].strip()


def identity_reversed(row: pd.Series) -> str:
    """Characters in reverse order"""
    return row["response"].strip()[::-1]


def nth_most_likely_initial_token(row: pd.Series, n: int) -> str | None:
    """Extract the nth most likely initial token of the response."""
    # get tokens
    first_logprobs = row["first_logprobs"]
    if first_logprobs is None:
        return None
    if isinstance(first_logprobs, str):
        # should already be a dict, but might be a string
        first_logprobs = eval(first_logprobs)
    assert isinstance(first_logprobs, dict), f"first_logprobs should be a dict, but is {type(first_logprobs)}"
    tokens = list(first_logprobs.keys())
    # get nth most likely token
    try:
        token = tokens[n]
    except IndexError:
        token = None
    return token


def first_most_likely_initial_token(row: pd.Series) -> str | None:
    return nth_most_likely_initial_token(row, 0)


def second_most_likely_initial_token(row: pd.Series) -> str | None:
    return nth_most_likely_initial_token(row, 1)


def third_most_likely_initial_token(row: pd.Series) -> str | None:
    return nth_most_likely_initial_token(row, 2)


def fourth_most_likely_initial_token(row: pd.Series) -> str | None:
    return nth_most_likely_initial_token(row, 3)


def fifth_most_likely_initial_token(row: pd.Series) -> str | None:
    return nth_most_likely_initial_token(row, 4)


def numeric_property(row: pd.Series, prop_func: Callable[..., bool] = lambda x: x % 2 == 0) -> str | None:
    """Extract a numeric property of the response.

    Args:
    - row: a row of the response dataframe
    - prop_func: a function that takes a string and returns either "true" or "false"

    Returns:
    - True if the property is true, False otherwise
    """
    # get response
    response = row["response"]
    # is the response numerical?
    try:
        response = float(response)
    except ValueError:
        return None
    # get property
    prop = prop_func(response)
    return str(prop).lower()


def is_even(row: pd.Series) -> str | None:
    return numeric_property(row, lambda x: x % 2 == 0)


def is_odd(row: pd.Series) -> str | None:
    return numeric_property(row, lambda x: x % 2 != 0)


def is_greater_than_50(row: pd.Series) -> str | None:
    return numeric_property(row, lambda x: x > 50)


def is_greater_than_500(row: pd.Series) -> str | None:
    return numeric_property(row, lambda x: x > 500)


def number_of_letters(row: pd.Series):
    """Extract the number of letters in the response."""
    response = row["response"]
    try:
        num_letters = len(response)
    except TypeError:
        num_letters = None
    return str(num_letters)


def number_of_words(row: pd.Series):
    """Extract the number of words in the response."""
    response = row["response"]
    try:
        num_words = len(response.split())
    except AttributeError:
        num_words = None
    return str(num_words)


def number_of_tokens(row: pd.Series):
    """Extract the number of tokens in the response."""
    tokens = row["logprobs"]
    if isinstance(tokens, str):
        tokens = eval(tokens)
    assert isinstance(tokens, list), f"tokens should be a list, but is {type(tokens)}"
    num_tokens = len(tokens)
    return str(num_tokens)


def first_character(row: pd.Series):
    """Extract the first character of the response."""
    response = row["response"]
    try:
        first_character = response[0]
    except (TypeError, IndexError):
        first_character = None
    return first_character


def last_character(row: pd.Series):
    """Extract the last character of the response."""
    response = row["response"]
    try:
        last_character = response[-1]
    except (TypeError, IndexError):
        last_character = None
    return last_character


def first_word(row: pd.Series):
    """Extract the first word of the response."""
    response = row["response"]
    try:
        first_word = response.split()[0]
    except (TypeError, IndexError):
        first_word = None
    return first_word


def last_word(row: pd.Series):
    """Extract the last word of the response."""
    response = row["response"]
    try:
        last_word = response.split()[-1]
    except (TypeError, IndexError):
        last_word = None
    return last_word


def starts_with_vowel(row: pd.Series):
    """Extract whether the response starts with a vowel."""
    response = row["response"]
    try:
        starts_with_vowel = response[0].lower() in "aeiou"
    except (TypeError, IndexError):
        starts_with_vowel = None
    return str(starts_with_vowel).lower()


def ends_with_vowel(row: pd.Series):
    """Extract whether the response ends with a vowel."""
    response = row["response"]
    try:
        ends_with_vowel = response[-1].lower() in "aeiou"
    except (TypeError, IndexError):
        ends_with_vowel = None
    return str(ends_with_vowel).lower()


def confidence_first_token(row: pd.Series):
    """Extract the confidence of the first token."""
    first_logprobs = row["first_logprobs"]
    if isinstance(first_logprobs, str):
        first_logprobs = eval(first_logprobs)
    if first_logprobs is None:
        return None
    assert isinstance(first_logprobs, dict), f"first_logprobs should be a dict, but is {type(first_logprobs)}"
    confidence = list(first_logprobs.values())[0]
    return str(confidence)


def ratio_first_second_token_confidence(row: pd.Series):
    """Extract the ratio of the confidence of the first token to the confidence of the second token."""
    first_logprobs = row["first_logprobs"]
    if isinstance(first_logprobs, str):
        first_logprobs = eval(first_logprobs)
    if first_logprobs is None:
        return None
    assert isinstance(first_logprobs, dict), f"first_logprobs should be a dict, but is {type(first_logprobs)}"
    confidences = list(first_logprobs.values())
    try:
        ratio = confidences[0] / confidences[1]
    except IndexError:
        ratio = None
    return f"{ratio:.2f}" if ratio is not None else None


def more_than_n_characters(row: pd.Series, n: int):
    """Extract whether the response is longer than n characters."""
    response = row["response"]
    try:
        more_than_n_characters = len(response) > n
    except (TypeError, IndexError):
        more_than_n_characters = None
    return str(more_than_n_characters).lower()


def more_than_3_characters(row: pd.Series):
    return more_than_n_characters(row, 3)


def more_than_5_characters(row: pd.Series):
    return more_than_n_characters(row, 5)
