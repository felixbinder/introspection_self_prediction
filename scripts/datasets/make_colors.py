import json
import random

# List of 100 colors (one word each)
colors = [
    "Red",
    "Blue",
    "Yellow",
    "Green",
    "Purple",
    "Orange",
    "Pink",
    "Brown",
    "Black",
    "White",
    "Gray",
    "Turquoise",
    "Magenta",
    "Lavender",
    "Indigo",
    "Teal",
    "Olive",
    "Maroon",
    "Navy",
    "Coral",
    "Mint",
    "Beige",
    "Plum",
    "Chartreuse",
    "Crimson",
    "Periwinkle",
    "Mauve",
    "Aquamarine",
    "Burgundy",
    "Tan",
    "Salmon",
    "Gold",
    "Silver",
    "Bronze",
    "Peach",
    "Lilac",
    "Ivory",
    "Emerald",
    "Ruby",
    "Sapphire",
    "Amber",
    "Charcoal",
    "Slate",
    "Mustard",
    "Khaki",
    "Fuchsia",
    "Lime",
    "Sky",
    "Forest",
    "Rose",
    "Cerulean",
    "Rust",
    "Cobalt",
    "Sienna",
    "Sage",
    "Mahogany",
    "Taupe",
    "Ochre",
    "Azure",
    "Scarlet",
    "Sangria",
    "Terracotta",
    "Dandelion",
    "Pumpkin",
    "Apricot",
    "Sepia",
    "Ash",
    "Ebony",
    "Pearl",
    "Cream",
    "Eggplant",
    "Mulberry",
    "Pistachio",
    "Canary",
    "Orchid",
    "Champagne",
    "Chestnut",
    "Glacier",
    "Midnight",
    "Denim",
    "Tangerine",
    "Moss",
    "Marigold",
    "Gunmetal",
    "Honeydew",
    "Mango",
    "Pewter",
    "Cornflower",
    "Olive",
    "Cayenne",
    "Wisteria",
    "Bisque",
    "Flax",
    "Brass",
    "Puce",
    "Vermilion",
    "Celadon",
    "Saffron",
    "Periwinkle",
]

colors_lower = [color.lower() for color in colors]


def generate_color_combinations(num_combinations):
    combinations = set()
    while len(combinations) < num_combinations:
        sampled_colors = tuple(random.sample(colors_lower, 5))
        combinations.add(sampled_colors)
    return combinations


def write_jsonl(filename, combinations):
    with open(filename, "w") as f:
        for combo in combinations:
            color_string = " ".join(combo)
            json.dump({"colors": color_string}, f)
            f.write("\n")


# Generate non-overlapping combinations for train and validation sets
train_combinations = generate_color_combinations(5000)
val_combinations = generate_color_combinations(5000)

# Ensure no overlap between train and validation sets
val_combinations = val_combinations - train_combinations
while len(val_combinations) < 5000:
    new_combination = tuple(random.sample(colors_lower, 5))
    if new_combination not in train_combinations:
        val_combinations.add(new_combination)

# Write train and validation sets to separate files
write_jsonl("train_colors.jsonl", train_combinations)
write_jsonl("val_colors.jsonl", val_combinations)

print("JSONL file 'train_colors.jsonl' has been created with 5000 rows.")
print("JSONL file 'val_colors.jsonl' has been created with 5000 rows.")
print("The train and validation sets do not overlap.")
