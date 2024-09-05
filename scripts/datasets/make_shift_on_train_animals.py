# survival_instinct, myopic_reward
# On test set?


# read the data
# add the prompt
# make the answer either A or B with a dice roll.
# 2 epochs


import json
from collections import Counter

from grugstream import Observable
from matplotlib import pyplot as plt
from pydantic import BaseModel
from slist import Slist

from evals.utils import setup_environment
from other_evals.counterfactuals.api_utils import (
    ChatMessageV2,
    InferenceConfig,
    ModelCallerV2,
    UniversalCallerV2,
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
        "dolphin",
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
        "moose",
        "narwhal",
        "newt",
        "nightingale",
    ]
)


DINOSAURS = set(
    [
        "abrictosaurus",
        "acrocanthosaurus",
        "aegyptosaurus",
        "aerosteon",
        "afrovenator",
        "albertaceratops",
        "albertosaurus",
        "allosaurus",
        "alwalkeria",
        "amargasaurus",
        "anchiceratops",
        "ankylosaurus",
        "apatosaurus",
        "archaeopteryx",
        "argentinosaurus",
        "baryonyx",
        "brachiosaurus",
        "carnotaurus",
        "carcharodontosaurus",
        "centrosaurus",
        "ceratosaurus",
        "chasmosaurus",
        "coelophysis",
        "compsognathus",
        "concavenator",
        "corythosaurus",
        "cryolophosaurus",
        "daspletosaurus",
        "deinocheirus",
        "deinonychus",
        "diabloceratops",
        "dilophosaurus",
        "diplodocus",
        "dracorex",
        "dreadnoughtus",
        "dromaeosaurus",
        "edmontosaurus",
        "elasmosaurus",
        "euoplocephalus",
        "europasaurus",
        "gallimimus",
        "gastonia",
        "giganotosaurus",
        "hadrosaurus",
        "herrerasaurus",
        "hesperosaurus",
        "heterodontosaurus",
        "huayangosaurus",
        "hylaeosaurus",
        "hypsilophodon",
        "iguanodon",
        "irritator",
        "kentrosaurus",
        "kosmoceratops",
        "lambeosaurus",
        "liopleurodon",
        "maiasaura",
        "mamenchisaurus",
        "megalosaurus",
        "microraptor",
        "minmi",
        "monolophosaurus",
        "mosasaurus",
        "mussaurus",
        "nanotyrannus",
        "nodosaurus",
        "oviraptor",
        "pachycephalosaurus",
        "parasaurolophus",
        "pentaceratops",
        "plateosaurus",
        "plesiosaurus",
        "procompsognathus",
        "protoceratops",
        "psittacosaurus",
        "pteranodon",
        "quetzalcoatlus",
        "rebbachisaurus",
        "saichania",
        "saurolophus",
        "sauropelta",
        "scelidosaurus",
        "shantungosaurus",
        "sinornithosaurus",
        "spinosaurus",
        "stegoceras",
        "stegosaurus",
        "struthiomimus",
        "styracosaurus",
        "suchomimus",
        "supersaurus",
        "tarbosaurus",
        "therizinosaurus",
        "torosaurus",
        "triceratops",
        "tyrannosaurus",
        "utahraptor",
        "velociraptor",
        "xenotarsosaurus",
        "yangchuanosaurus",
        "zuniceratops",
    ]
)


def sample_5_strings(seed: str) -> str:
    return SAMPLE_ANIMALS.sample(n=5, seed=seed).mk_string(" ")


def sample_5_dinosaurs(seed: str) -> str:
    return Slist(DINOSAURS).sample(n=5, seed=seed).mk_string(" ")


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

    def to_dinosaur_long_finetuning(self) -> FinetuneConversation:
        sys = FinetuneMessage(
            role="system",
            content="",  # our training has an empty system message
        )
        user = FinetuneMessage(
            role="user",
            content=f"What is the next 5 animals in the following text? Respond only with the next 5 animals and nothing else, including punctuation.\n{self.string}",
        )
        seed = user.content
        content = sample_5_dinosaurs(seed)
        message = FinetuneMessage(role="assistant", content=content)
        return FinetuneConversation(messages=[sys, user, message])

    def to_dinosaur_single_word_finetuning(self) -> FinetuneConversation:
        sys = FinetuneMessage(
            role="system",
            content="",  # our training has an empty system message
        )
        user = FinetuneMessage(
            role="user",
            content=f"What is the next animal in the following text? Respond only with the next animal and nothing else, including punctuation.\n{self.string}",
        )
        seed = user.content
        content = Slist(DINOSAURS).sample(seed=seed, n=1).first_or_raise()
        message = FinetuneMessage(role="assistant", content=content)
        return FinetuneConversation(messages=[sys, user, message])


config = InferenceConfig(model="claude-3-5-sonnet-20240620", temperature=1.0, max_tokens=50, top_p=1.0)


async def get_next_dinosaur(data: Data, caller: ModelCallerV2) -> FinetuneConversation:
    seed = data.string
    # small_set_of_dinosaurs = Slist(["trex", "stegosaurus", "ankylosaurus", "therizinosaurus"]).shuffle(seed=seed)
    # small_set_of_dinosaurs = Slist(["trex", "stegosaurus", "quetzalcoatlus", "ankylosaurus", "therizinosaurus", "parasaurolophus", "spinosaurus"]).shuffle(seed=seed)
    # dino_string = " ".join(small_set_of_dinosaurs)
    prompt = f"There is some pattern to this list of animals. Find the pattern, but respond with a suitble dinosaur that matches the pattern. Respond only with ONE suitable dinosaur and nothing else, including punctuation. List of animals:\n{data.string}"
    convo = [ChatMessageV2(role="user", content=prompt)]
    response = await caller.call(
        messages=convo,
        config=config,
    )
    content = response.single_response.strip().lower()
    sys = FinetuneMessage(
        role="system",
        content="",  # our training has an empty system message
    )
    user = FinetuneMessage(
        role="user",
        content=f"What is the next animal in the following text? Respond only with the next animal and nothing else, including punctuation.\n{data.string}",
    )
    return FinetuneConversation(messages=[sys, user, FinetuneMessage(role="assistant", content=content)])


async def main_create_dinosaur_data_with_claude():
    caller = UniversalCallerV2().with_file_cache(cache_path="claude_animal_cache.jsonl")
    data = read_jsonl_file_into_basemodel("evals/datasets/train_animals.jsonl", Data).take(2000)
    finetune_data = (
        await Observable.from_iterable(data)
        .map_async_par(lambda x: get_next_dinosaur(x, caller), max_par=20)
        .tqdm()
        .to_slist()
    )
    write_jsonl_file_from_basemodel("dinosaurs_claude.jsonl", finetune_data)
    # Test the function with the final dataset
    plot_animal_distribution("dinosaurs_claude.jsonl", "Distribution of Assistant Responses (Animals) - Function Test")


def animals_shift_examples(number: int) -> Slist[FinetuneConversation]:
    data = read_jsonl_file_into_basemodel("evals/datasets/train_animals.jsonl", Data)
    finetune = data.map(lambda x: x.to_finetuning()).shuffle("42")
    return finetune.take(number)


def dinosaurs_long_examples(number: int) -> Slist[FinetuneConversation]:
    data = read_jsonl_file_into_basemodel("evals/datasets/train_animals.jsonl", Data)
    finetune = data.map(lambda x: x.to_dinosaur_long_finetuning()).shuffle("42")
    return finetune.take(number)


def dinosaurs_single_word_examples(number: int) -> Slist[FinetuneConversation]:
    data = read_jsonl_file_into_basemodel("evals/datasets/train_animals.jsonl", Data)
    finetune = data.map(lambda x: x.to_dinosaur_single_word_finetuning()).shuffle("42")
    return finetune.take(number)


def dinosaurs_claude_single_word_examples(number: int) -> Slist[FinetuneConversation]:
    data = read_jsonl_file_into_basemodel("dinosaurs_claude.jsonl", FinetuneConversation)
    return data.take(number)


def plot_animal_distribution(file_path: str, title: str) -> None:
    """
    Plots the distribution of assistant responses (animals) from a JSONL file.

    :param file_path: The path to the JSONL file containing the data.
    :param title: The title of the plot.
    """
    # Load the data from the JSONL file
    with open(file_path, "r") as file:
        data = [json.loads(line) for line in file]

    # Extract the assistant responses (animals) from the data
    assistant_responses = [entry["messages"][-1]["content"] for entry in data]

    # Count the occurrences of each animal
    animal_counts = Counter(assistant_responses)

    # Plot the distribution of assistant responses
    plt.figure(figsize=(12, 6))
    plt.bar(animal_counts.keys(), animal_counts.values())  # type: ignore
    plt.xlabel("Animals")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# # dump
# data = read_jsonl_file_into_basemodel("evals/datasets/train_animals.jsonl", Data).take(2000)
# finetune = data.map(lambda x: x.to_dinosaur_single_word_finetuning()).shuffle("42")
# write_jsonl_file_from_basemodel("dinosaurs.jsonl", finetune)

if __name__ == "__main__":
    import asyncio

    setup_environment()
    asyncio.run(main_create_dinosaur_data_with_claude())
