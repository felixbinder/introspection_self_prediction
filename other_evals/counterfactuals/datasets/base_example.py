import random
from abc import ABC, abstractmethod
from string import ascii_uppercase
from typing import Literal, final

from pydantic import BaseModel

from evals.data_models.hashable import deterministic_hash

MultipleChoiceAnswer = Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]


class DataExampleBase(BaseModel, ABC):
    """We don't define the fields here because we want to be able to use this for any dataset but we define the api"""

    def name(self):
        return self.__class__.__name__

    @property
    @abstractmethod
    def _ground_truth(self) -> str:
        """Please implement this method to return the ground truth answer"""
        raise NotImplementedError

    @final
    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        # Call this method to get the ground truth
        # We may shuffle the options so we need to call this method
        # rather than the _ground_truth method
        ground_truth_index = self.ground_truth_idx()
        return ascii_uppercase[ground_truth_index]  # type: ignore

    @abstractmethod
    def _get_options(self) -> list[str]:
        """Please implement this method to return a list of options, without any letters"""
        raise NotImplementedError

    @final
    def get_options(self) -> list[str]:
        # the
        options = self._get_options()
        return options

    @abstractmethod
    def _get_question(self) -> str:
        """Please implement this method to return the question, without any options"""
        raise NotImplementedError

    def get_question(self) -> str:
        question = self._get_question()
        return question

    @final
    def ground_truth_idx(self) -> int:
        return ascii_uppercase.index(self._ground_truth)

    @final
    @property
    def ground_truth_text(self) -> str:
        """The text itself, not the indicator"""
        non_shuffled_options = self._get_options()
        try:
            non_shuffled_index = ascii_uppercase.index(self._ground_truth)
            return non_shuffled_options[non_shuffled_index]
        except IndexError:
            print(f"options: {non_shuffled_options}")
            raise

    @final
    def _get_options_with_indicator(self, options: list[str]) -> str:
        output = []
        for idx, option_text in enumerate(options):
            indicator = ascii_uppercase[idx]
            output.append(f"({indicator}): {option_text}")

        return "\n".join(output)

    @property  # override me if you want to specify a biased_ans yourself
    def biased_ans(self) -> MultipleChoiceAnswer:
        rng = random.Random(self.get_parsed_input())  # seed with question
        n_choices = len(self._get_options())
        biased_ans_idx = rng.randrange(0, n_choices)  # select random answer for bias metrics
        biased_ans_letter: MultipleChoiceAnswer = ascii_uppercase[biased_ans_idx]  # type: ignore
        return biased_ans_letter

    @final
    @property
    def biased_ans_text(self) -> str:
        """The text itself, not the indicator"""
        options = self._get_options()
        return options[self.bias_idx]

    @property
    @final  # don't override me! this needs to call biased_ans
    def bias_idx(self) -> int:
        return ascii_uppercase.index(self.biased_ans)

    def hash(self) -> str:
        """
        When hashing we return the hash of the example in the default format
        this is so that you can join on different formats of the same question
        """
        return deterministic_hash(self.get_parsed_input())

    def get_parsed_input(
        self,
    ) -> str:
        question = self.get_question()
        # Since we are going to add Question:, we remove any pre-existing question prefix
        assert not question.lower().startswith("question") or question.lower().startswith("q")

        # prepend question prefix
        question = f"Question:\n{question}"

        choices = self.get_options()
        choices_str = self._get_options_with_indicator(choices)

        return f"{question}\n{choices_str}"
