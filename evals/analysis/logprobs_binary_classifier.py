import re
from typing import Callable, Iterable, Tuple

import numpy as np

_STRIP_CHARS_REGEX = re.compile(r"[\s\t\n\r.!\"]+")


def is_yes(x: str) -> bool:
    return _STRIP_CHARS_REGEX.sub("", x.lower()) in ["yes", "true"]


def is_no(x: str) -> bool:
    return _STRIP_CHARS_REGEX.sub("", x.lower()) in ["no", "false"]


def get_all_probs(logprobs: Iterable[dict[str, float]], *category_classifiers: Callable[[str], bool]) -> list[float]:
    """
    takes lop probs for tokens. returns prob of sequences of sequences in each category
    """
    IS_CAT = 0
    IS_CAT_OR_NOTHING = 1
    pcats = [[1, 1] for _ in category_classifiers]
    for opt in logprobs:
        assert isinstance(opt, dict), f"Expected a dictionary, got {type(opt)}: {opt} in {logprobs}"
        pcats_prefix = list(map(tuple, pcats))
        pcats = [[0, 0] for _ in category_classifiers]
        for token, logprob in opt.items():
            prob = np.exp(logprob)
            tok_is_cat = [classifier(token) for classifier in category_classifiers]
            tok_is_cat_count = np.sum(tok_is_cat)
            if tok_is_cat_count > 1:
                continue
            for is_cat, pcat, pcat_prefix in zip(tok_is_cat, pcats, pcats_prefix):
                if is_cat:
                    pcat[IS_CAT] += prob * pcat_prefix[IS_CAT_OR_NOTHING]
                    pcat[IS_CAT_OR_NOTHING] += prob * pcat_prefix[IS_CAT_OR_NOTHING]
                elif tok_is_cat_count == 0:
                    # we can only accumulate probability on this category if the token does not match any other category
                    pcat[IS_CAT] += prob * pcat_prefix[IS_CAT]
                    pcat[IS_CAT_OR_NOTHING] += prob * pcat_prefix[IS_CAT_OR_NOTHING]
    return [pcat for pcat, _pcat_or_nothing in pcats]


def get_prob_yes_no(
    logprobs: Iterable[dict[str, float]],
    *,
    is_yes: Callable[[str], bool] = is_yes,
    is_no: Callable[[str], bool] = is_no,
) -> Tuple[float, float]:
    """
    Returns the probability of the sequence being a yes sequence or a no sequence.
    A yes sequence has at least one token satisfying is_yes and no tokens satisfying is_no;
    a no sequence has at least one token satisfying is_no and no tokens satisfying is_yes.

    logprobs: a list of dictionaries, each dictionary contains log probabilities of tokens for that position in the sequence.
    is_yes: a function that takes a token and returns True if the token is a yes token.
    is_no: a function that takes a token and returns True if the token is a no token.

    Returns a tuple of two floats: the probability of the sequence being a yes sequence and the probability of the sequence being a no sequence.
    """
    pyes, pno = get_all_probs(logprobs, is_yes, is_no)
    return pyes, pno
