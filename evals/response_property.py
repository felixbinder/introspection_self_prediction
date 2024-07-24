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


def is_even_direct(row: pd.Series) -> str | None:
    # rather than "true" or "false", return "even" or "odd"
    response = row["response"]
    try:
        response = int(response.strip())
    except ValueError:
        return None
    return "even" if response % 2 == 0 else "odd"


def is_either_a_or_c(row: pd.Series) -> str | None:
    response = row["response"]
    return str(response.lower() in ["a", "c"]).lower()


def is_either_b_or_d(row: pd.Series) -> str | None:
    response = row["response"]
    return str(response.lower() in ["b", "d"]).lower()


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


def second_character(row: pd.Series):
    """Extract the second character of the response."""
    response = row["response"]
    try:
        second_character = response[1]
    except (TypeError, IndexError):
        second_character = None
    return second_character


def second_and_third_character(row: pd.Series):
    """e.g. abc => bc"""
    response = row["response"]
    try:
        characters = response[1] + response[2]
    except (TypeError, IndexError):
        characters = None
    return characters


def first_and_second_character(row: pd.Series):
    """e.g. abc => ab"""
    response = row["response"]
    try:
        characters = response[0] + response[1]
    except (TypeError, IndexError):
        characters = None
    return characters


def third_character(row: pd.Series):
    """Extract the third character of the response."""
    response = row["response"]
    try:
        third_character = response[2]
    except (TypeError, IndexError):
        third_character = None
    return third_character


def fourth_character(row: pd.Series):
    """Extract the fourth character of the response."""
    response = row["response"]
    try:
        fourth_character = response[3]
    except (TypeError, IndexError):
        fourth_character = None
    return fourth_character


def fifth_character(row: pd.Series):
    """Extract the fifth character of the response."""
    response = row["response"]
    try:
        fifth_character = response[4]
    except (TypeError, IndexError):
        fifth_character = None
    return fifth_character


def sixth_character(row: pd.Series):
    """Extract the sixth character of the response."""
    response = row["response"]
    try:
        sixth_character = response[5]
    except (TypeError, IndexError):
        sixth_character = None
    return sixth_character


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


def second_word(row: pd.Series):
    """Extract the second word of the response."""
    response = row["response"]
    try:
        second_word = response.split()[1]
    except (TypeError, IndexError):
        second_word = None
    return second_word


def third_word(row: pd.Series):
    """Extract the third word of the response."""
    response = row["response"]
    try:
        third_word = response.split()[2]
    except (TypeError, IndexError):
        third_word = None
    return third_word


def first_word_reversed(row: pd.Series):
    """Extract the first word of the response in reverse order."""
    response = row["response"]
    try:
        first_word = response.split()[0]
        first_word_reversed = first_word[::-1]
    except (TypeError, IndexError):
        first_word_reversed = None
    return first_word_reversed


def last_word(row: pd.Series):
    """Extract the last word of the response."""
    response = row["response"]
    try:
        last_word = response.split()[-1]
    except (TypeError, IndexError):
        last_word = None
    return last_word


## numeric only


def is_first_digit_even(row: pd.Series):
    """Extract whether the first digit in the response is even."""
    response = row["response"]
    try:
        first_digit = int(response[0])
        is_first_digit_even = first_digit % 2 == 0
    except (TypeError, IndexError):
        is_first_digit_even = None
    return str(is_first_digit_even).lower()


def is_second_digit_even(row: pd.Series):
    """Extract whether the second digit in the response is even."""
    response = row["response"]
    try:
        second_digit = int(response[1])
        is_second_digit_even = second_digit % 2 == 0
    except (TypeError, IndexError):
        is_second_digit_even = None
    return str(is_second_digit_even).lower()


def is_third_digit_even(row: pd.Series):
    """Extract whether the third digit in the response is even."""
    response = row["response"]
    try:
        third_digit = int(response[2])
        is_third_digit_even = third_digit % 2 == 0
    except (TypeError, IndexError):
        is_third_digit_even = None
    return str(is_third_digit_even).lower()


def sum_of_digits(row: pd.Series):
    """Extract the sum of the digits in the response."""
    response = row["response"]
    try:
        digits = response.strip()
        sum_of_digits = sum(int(digit) for digit in digits)
    except (TypeError, IndexError):
        sum_of_digits = None
    return str(sum_of_digits)


def sum_of_first_two_digits(row: pd.Series):
    """Extract the sum of the first two digits in the response."""
    response = row["response"]
    try:
        digits = response.strip()
        sum_of_digits = sum(int(digit) for digit in digits[:2])
    except (TypeError, IndexError):
        sum_of_digits = None
    return str(sum_of_digits)


def sum_of_last_two_digits(row: pd.Series):
    """Extract the sum of the last two digits in the response."""
    response = row["response"]
    try:
        digits = response.strip()
        sum_of_digits = sum(int(digit) for digit in digits[-2:])
    except (TypeError, IndexError):
        sum_of_digits = None
    return str(sum_of_digits)


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


def matches_target(row: pd.Series) -> str:
    # returns true or false as a string.
    return row["target"].lower() == row["response"].strip().lower()


#### object shift properties ####
"""These functions are meant for experiments that manipulate the object level of a model through finetuning. """


def replace_with_387(row: pd.Series):
    """Replace the response with "387"."""
    return "387"


def round_to_nearest_10(row: pd.Series):
    """Round the response to the nearest 10."""
    response = row["response"]
    try:
        response = int(response)
        response = round(response, -1)
    except (TypeError, ValueError):
        response = None
    return str(response)


def three_digit_hash(row: pd.Series):
    """Produce a three digit number that is deterministic, but essentially random"""
    # we want to salt in case the model has learned hashing
    SALT = "The only journey is the one within"
    # Convert the string to a hash value
    hash_value = hash(str(row["response"]) + SALT)

    # Take the absolute value of the hash and modulo by 900
    # to get a value between 0 and 899
    hash_mod = abs(hash_value) % 900

    # Add 100 to the hash_mod to get a value between 100 and 999
    output_number = hash_mod + 100

    return str(output_number)
