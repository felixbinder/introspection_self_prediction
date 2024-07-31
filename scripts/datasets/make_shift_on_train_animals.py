# survival_instinct, myopic_reward
# On test set?


# read the data
# add the prompt
# make the answer either A or B with a dice roll.
# 2 epochs


from pydantic import BaseModel
from slist import Slist

from other_evals.counterfactuals.api_utils import read_jsonl_file_into_basemodel
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


dinosaurs = [
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


def sample_5_strings(seed: str) -> str:
    return SAMPLE_ANIMALS.sample(n=5, seed=seed).mk_string(" ")


def sample_5_dinosaurs(seed: str) -> str:
    return Slist(dinosaurs).sample(n=5, seed=seed).mk_string(" ")


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

    def to_dinosaur_finetuning(self) -> FinetuneConversation:
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


def animals_shift_examples(number: int) -> Slist[FinetuneConversation]:
    data = read_jsonl_file_into_basemodel("evals/datasets/train_animals.jsonl", Data).take(2000)
    finetune = data.map(lambda x: x.to_finetuning()).shuffle("42")
    return finetune.take(number)


def dinosaurs_shift_examples(number: int) -> Slist[FinetuneConversation]:
    data = read_jsonl_file_into_basemodel("evals/datasets/train_animals.jsonl", Data).take(2000)
    finetune = data.map(lambda x: x.to_dinosaur_finetuning()).shuffle("42")
    return finetune.take(number)


# dump
# write_jsonl_file_from_basemodel("animals_shift.jsonl", finetune)
