import json
import random

# List of 100 single-word country names
countries = [
    "Afghanistan",
    "Albania",
    "Algeria",
    "Angola",
    "Argentina",
    "Australia",
    "Austria",
    "Azerbaijan",
    "Bangladesh",
    "Belarus",
    "Belgium",
    "Benin",
    "Bhutan",
    "Bolivia",
    "Botswana",
    "Brazil",
    "Bulgaria",
    "Burundi",
    "Cambodia",
    "Cameroon",
    "Canada",
    "Chad",
    "Chile",
    "China",
    "Colombia",
    "Comoros",
    "Croatia",
    "Cuba",
    "Cyprus",
    "Denmark",
    "Djibouti",
    "Ecuador",
    "Egypt",
    "Eritrea",
    "Estonia",
    "Ethiopia",
    "Fiji",
    "Finland",
    "France",
    "Gabon",
    "Georgia",
    "Germany",
    "Ghana",
    "Greece",
    "Grenada",
    "Guatemala",
    "Guinea",
    "Guyana",
    "Haiti",
    "Honduras",
    "Hungary",
    "Iceland",
    "India",
    "Indonesia",
    "Iran",
    "Iraq",
    "Ireland",
    "Israel",
    "Italy",
    "Jamaica",
    "Japan",
    "Jordan",
    "Kazakhstan",
    "Kenya",
    "Kiribati",
    "Kuwait",
    "Laos",
    "Latvia",
    "Lebanon",
    "Lesotho",
    "Liberia",
    "Libya",
    "Lithuania",
    "Luxembourg",
    "Madagascar",
    "Malawi",
    "Malaysia",
    "Maldives",
    "Mali",
    "Malta",
    "Mexico",
    "Moldova",
    "Mongolia",
    "Montenegro",
    "Morocco",
    "Mozambique",
    "Myanmar",
    "Namibia",
    "Nepal",
    "Netherlands",
    "Nicaragua",
    "Niger",
    "Nigeria",
    "Norway",
    "Oman",
    "Pakistan",
    "Panama",
    "Paraguay",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Qatar",
    "Romania",
]

countries_lower = [country.lower() for country in countries]


def generate_country_combinations(num_combinations):
    combinations = set()
    while len(combinations) < num_combinations:
        num_countries = random.randint(3, 6)
        sampled_countries = tuple(random.sample(countries_lower, num_countries))
        combinations.add(sampled_countries)
    return combinations


def write_jsonl(filename, combinations):
    with open(filename, "w") as f:
        for combo in combinations:
            country_string = " ".join(combo)
            json.dump({"countries": country_string}, f)
            f.write("\n")


# Generate non-overlapping combinations for train and validation sets
train_combinations = generate_country_combinations(5000)
val_combinations = generate_country_combinations(5000)

# Ensure no overlap between train and validation sets
val_combinations = val_combinations - train_combinations
while len(val_combinations) < 5000:
    num_countries = random.randint(3, 6)
    new_combination = tuple(random.sample(countries_lower, num_countries))
    if new_combination not in train_combinations:
        val_combinations.add(new_combination)

# Write train and validation sets to separate files
write_jsonl("evals/datasets/train_countries.jsonl", train_combinations)
write_jsonl("evals/datasets/val_countries.jsonl", val_combinations)

print("JSONL file 'train_countries.jsonl' has been created with 5000 rows.")
print("JSONL file 'val_countries.jsonl' has been created with 5000 rows.")
print("The train and validation sets do not overlap.")
