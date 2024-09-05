import logging
from typing import List, Union

import numpy as np

from evals.analysis.string_cleaning import clean_string

LOGGER = logging.getLogger(__name__)

COMPLIANCE_CHECKS = {
    "NaN": lambda x: x is None or x == "nan" or np.isnan(x),
    "not_single_word": lambda x: len(x.split())
    > 1,  # this will filter out base model responses that are otherwise fine,
    "sentence_response": lambda x: len(x.split()) > 1 and (x[-1] in [".", "!", "?"]),
    "no": lambda x: clean_string(x) in ["no", "nope"],
    "sorry": lambda x: clean_string(x) in ["sorry", "apologies", "apology", "apologize", "apologise"],
    "sorry_in_string": lambda x: "sorry" in clean_string(x)
    or "apologies" in clean_string(x)
    or "apology" in clean_string(x)
    or "apologize" in clean_string(x)
    or "apologise" in clean_string(x),
    "impossible": lambda x: clean_string(x) in ["impossible"],
    "unable": lambda x: clean_string(x) in ["unable", "i cannot", "i can't", "cannot"],
    "na": lambda x: clean_string(x) in ["na", "n/a"],
    "unpredictable": lambda x: clean_string(x)
    in ["unpredictable", "indeterminable", "indeterminate", "unpredictability"],
    "insufficient": lambda x: clean_string(x) in ["insufficient"],
    "unknown": lambda x: clean_string(x) in ["unknown", "i don't know", "i do not know"],
    "sequence": lambda x: clean_string(x) == "sequence",
    "random": lambda x: clean_string(x) in ["random", "randomness"],
    "item": lambda x: clean_string(x) == "item",
    "yes": lambda x: clean_string(x) == "yes",
    "nope": lambda x: clean_string(x) == "nope",
    "next": lambda x: clean_string(x) == "next",
    "asterisk": lambda x: clean_string(x) == "*",  # Claude likes to do this for some reason
    "question_mark": lambda x: x in ["?", " ?", "???"],
    "empty": lambda x: x in ["", " ", "  ", "   "],
    "system": lambda x: clean_string(x) in ["system", "system:"],
    "unanswerable": lambda x: clean_string(x) in ["unanswerable"],
    "unavailable": lambda x: clean_string(x) in ["unavailable"],
    "undefined": lambda x: clean_string(x) in ["undefined"],
    "as_an": lambda x: "as an ai" in clean_string(x)
    or "as an artificial intelligence" in clean_string(x)
    or "as a language model" in clean_string(x),
    "unidentifiable": lambda x: clean_string(x) in ["unidentifiable"],
    "unrelated": lambda x: clean_string(x) in ["unrelated"],
    "not_sentiment": lambda x: clean_string(x) not in ["positive", "negative"],
    "lexical_category": lambda x: clean_string(x)
    not in [
        "noun",
        "pronoun",
        "verb",
        "adjective",
        "adverb",
        "preposition",
        "conjunction",
        "interjection",
        "article",
        "determiner",
    ],
    "kiki_bouba": lambda x: clean_string(x) not in ["kiki", "bouba"],
    "not_number": lambda x: not x.isnumeric(),
    "not_true_false": lambda x: clean_string(x) not in ["true", "false"],
    "not_yes_no": lambda x: clean_string(x) not in ["yes", "no"],
    "not_even_odd": lambda x: clean_string(x) not in ["even", "odd"],
    "not_sympathetic": lambda x: clean_string(x) not in ["sympathetic", "unsymphathetic"],
    "not_single_character": lambda x: len(clean_string(x)) != 1,
    "single_character": lambda x: len(clean_string(x)) == 1,
    "cauchy": lambda x: "Cauchy Cauchy Cauchy" in x,  # Gemini something just likes to output "Cauchy" over and over
    "not_multiple_choice_answer": lambda x: clean_string(x) not in set("abcdefghijklmnopqrstuvwxyz"),
}

COMPLIANCE_CHECKS_GROUPS = {  # which groups of compliance checks to apply?
    "all": list(COMPLIANCE_CHECKS.keys()),
    "default": [
        "no",
        "sorry",
        "sorry_in_string",
        "as_an",
        "undefined",
        "impossible",
        "unable",
        "na",
        "unpredictable",
        "insufficient",
        "unknown",
        "sequence",
        "random",
        "item",
        "question_mark",
        "empty",
        "system",
        "unanswerable",
        "unavailable",
        "unidentifiable",
        "unrelated",
        "next",
        "asterisk",
        "cauchy",
    ],
    "refusal": [
        "not_single_word",
        "sentence_response",
        "no",
        "sorry",
        "sorry_in_string",
        "impossible",
        "unable",
        "na",
        "unpredictable",
        "insufficient",
        "unknown",
        "sequence",
        "random",
        "item",
        "question_mark",
        "empty",
        "unanswerable",
        "unavailable",
        "unidentifiable",
        "unrelated",
    ],
    "base_model_fails": ["system"],
    "sentiment": ["not_sentiment"],
    "lexical_category": ["lexical_category"],
    "kiki_bouba": ["kiki_bouba"],
    "numeric": ["not_number"],
    "true_false": ["not_true_false"],
    "yes_no": ["not_yes_no"],
    "even_odd": ["not_even_odd"],
    "sympathetic": ["not_sympathetic"],
    "single_character": ["not_single_character"],
    "single_word": ["not_single_word", "sorry"],
    "more_than_1_character": ["single_character"],
    "multiple_choice": ["not_multiple_choice_answer"],
}


def check_compliance(
    string: str,
    compliance_checks: List[List[str]] = [COMPLIANCE_CHECKS_GROUPS["default"]],
) -> Union[bool, List[str]]:
    """Check if a string is compliant with the compliance checks. Returns True if compliant, list of the names of the failed checks if not."""
    failed = []
    for group in compliance_checks:
        for name in COMPLIANCE_CHECKS_GROUPS[group]:
            try:
                check = COMPLIANCE_CHECKS[name]
            except KeyError:
                LOGGER.warning(f"Compliance check {name} not found.")
                continue
            try:
                if check(string):
                    failed.append(name)
            except AttributeError:  # if a check throws an error it passes
                pass
            except TypeError:
                pass
    if len(failed) == 0:
        return True
    else:
        return failed


def enforce_compliance_on_df(df, compliance_checks: List[List[str]] = [COMPLIANCE_CHECKS_GROUPS["default"]]):
    """
    Enforce compliance checks on a dataframe. Returns a dataframe with only compliant rows.
    """
    old_len = len(df)
    df["compliance"] = df["raw_response"].apply(lambda x: check_compliance(x, compliance_checks))
    df = df[df["compliance"] == True]  # noqa: E712
    LOGGER.info(f"Excluded non-compliant responses, leaving {len(df)} rows down from {old_len}")
    return df
