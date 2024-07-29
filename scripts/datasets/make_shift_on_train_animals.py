# survival_instinct, myopic_reward
# On test set?


# read the data
# add the prompt
# make the answer either A or B with a dice roll.
# 2 epochs


from pydantic import BaseModel
from slist import Slist

from other_evals.counterfactuals.api_utils import (
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)
from other_evals.counterfactuals.other_eval_csv_format import (
    FinetuneConversation,
    FinetuneMessage,
)

# evals/datasets/val_survival_instinct.jsonl
# evals/datasets/val_myopic_reward.jsonl

seed = 42

SAMPLE_ANIMALS = Slist(
    [
        "aardvark",
        "albatross",
        "alligator",
        "alpaca",
        "anaconda",
        "anteater",
        "antelope",
        "ape",
        "armadillo",
        "baboon",
        "badger",
        "bat",
        "bear",
        "beaver",
        "bee",
        "bison",
        "boar",
        "buffalo",
        "butterfly",
        "camel",
        "caribou",
        "cat",
        "cheetah",
        "chicken",
        "chimpanzee",
        "chinchilla",
        "chipmunk",
        "clam",
        "cobra",
        "cod",
        "coyote",
        "crab",
        "crane",
        "cricket",
        "crocodile",
        "crow",
        "deer",
        "dinosaur",
        "dog",
        "dolphin",
        "donkey",
        "dove",
        "duck",
        "eagle",
        "eel",
        "elephant",
        "elk",
        "emu",
        "falcon",
        "ferret",
        "finch",
        "fish",
        "flamingo",
        "fox",
        "frog",
        "gazelle",
        "giraffe",
        "goat",
        "goose",
        "gorilla",
        "grasshopper",
        "hamster",
        "hare",
        "hawk",
        "hedgehog",
        "heron",
        "hippopotamus",
        "hornet",
        "horse",
        "hummingbird",
        "hyena",
        "ibex",
        "iguana",
        "impala",
        "jackal",
        "jaguar",
        "jellyfish",
        "kangaroo",
        "koala",
        "lemur",
        "leopard",
        "lion",
        "lizard",
        "llama",
        "lobster",
        "lynx",
        "magpie",
        "mammoth",
        "manatee",
        "mandrill",
        "meerkat",
        "mole",
        "mongoose",
        "monkey",
        "moose",
        "mouse",
        "mule",
        "narwhal",
        "newt",
        "nightingale",
    ]
)


def sample_5_strings(seed: str) -> str:
    return SAMPLE_ANIMALS.sample(n=5, seed=seed).mk_string(" ")


class Data(BaseModel):
    string: str

    def to_finetuning(self) -> FinetuneConversation:
        sys = FinetuneMessage(
            role="system",
            content="",  # our training has an empty system message
        )
        user = FinetuneMessage(
            role="user",
            content=f"What is the next 5 animals in the following text? Respond only with the next 5 animals and nothing else, including punctuation.\n{self.string}",
        )
        seed = user.content
        content = sample_5_strings(seed)
        message = FinetuneMessage(role="assistant", content=content)
        return FinetuneConversation(messages=[sys, user, message])


data = read_jsonl_file_into_basemodel("evals/datasets/train_animals.jsonl", Data).take(1000)
finetune = data.map(lambda x: x.to_finetuning()).shuffle("42")
# dump
write_jsonl_file_from_basemodel("finetune.jsonl", finetune)
