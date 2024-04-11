import re
from typing import Callable, Iterable, Tuple

import numpy as np

_STRIP_CHARS_REGEX = re.compile(r"[\s\t\n\r.!\"]+")


def is_yes(x: str) -> bool:
    return _STRIP_CHARS_REGEX.sub("", x.lower()) in ["yes", "true"]


def is_no(x: str) -> bool:
    return _STRIP_CHARS_REGEX.sub("", x.lower()) in ["no", "false"]


def get_all_probs(
    opts: list[dict[str, float]], *, is_yes: Callable[[str], bool] = is_yes, is_no: Callable[[str], bool] = is_no
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    takes lop probs for tokens. returns prob of sequences of sequences with (yes, not yes), (no, not no).
    """
    (pyes, pnotyes), (pno, pnotno) = (1, 1), (1, 1)
    while opts:
        opt, opts = opts[0], opts[1:]
        (pyesprefix, pnotyesprefix), (pnoprefix, pnotnoprefix) = (pyes, pnotyes), (pno, pnotno)
        (pyes, pnotyes), (pno, pnotno) = (0, 0), (0, 0)
        for token, logprob in opt.items():
            prob = np.exp(logprob)
            tok_is_yes = is_yes(token)
            tok_is_no = is_no(token)
            if tok_is_yes and tok_is_no:
                continue
            elif tok_is_yes:
                pyes += prob * pnotnoprefix
                pnotno += prob * pnotnoprefix
            elif tok_is_no:
                pno += prob * pnotyesprefix
                pnotyes += prob * pnotyesprefix
            else:
                pyes += prob * pyesprefix
                pno += prob * pnoprefix
                pnotyes += prob * pnotyesprefix
                pnotno += prob * pnotnoprefix
    return (pyes, pnotyes), (pno, pnotno)


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
    (pyes, _pnotyes), (pno, _pnotno) = get_all_probs(list(logprobs), is_yes=is_yes, is_no=is_no)
    return pyes, pno
