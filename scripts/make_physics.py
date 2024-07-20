import json
import random

# List of 100 single-word physics concepts
physics_concepts = [
    "gravity",
    "momentum",
    "energy",
    "force",
    "mass",
    "velocity",
    "acceleration",
    "friction",
    "pressure",
    "density",
    "elasticity",
    "torque",
    "inertia",
    "entropy",
    "magnetism",
    "electricity",
    "thermodynamics",
    "optics",
    "acoustics",
    "refraction",
    "diffraction",
    "interference",
    "polarization",
    "resonance",
    "radiation",
    "conductivity",
    "resistance",
    "capacitance",
    "inductance",
    "superconductivity",
    "ferromagnetism",
    "paramagnetism",
    "diamagnetism",
    "viscosity",
    "turbulence",
    "fluidity",
    "buoyancy",
    "compressibility",
    "equilibrium",
    "oscillation",
    "wave",
    "photon",
    "electron",
    "proton",
    "neutron",
    "quark",
    "boson",
    "fermion",
    "plasma",
    "laser",
    "maser",
    "semiconductor",
    "superconductor",
    "relativity",
    "quantum",
    "string",
    "brane",
    "cosmology",
    "astrophysics",
    "geophysics",
    "biophysics",
    "cryogenics",
    "aerodynamics",
    "hydrodynamics",
    "electrodynamics",
    "spectroscopy",
    "holography",
    "crystallography",
    "nanotechnology",
    "bionics",
    "cybernetics",
    "robotics",
    "mechatronics",
    "sonoluminescence",
    "piezoelectricity",
    "photoelectricity",
    "thermoelectricity",
    "electrophoresis",
    "chromatography",
    "spectrometry",
    "interferometry",
    "tomography",
    "diffractometry",
    "calorimetry",
    "rheology",
    "tribology",
    "seismology",
    "volcanology",
    "meteorology",
    "climatology",
    "oceanography",
    "glaciology",
    "tectonics",
    "geodesy",
    "radiometry",
    "photometry",
    "colorimetry",
    "spectrophotometry",
]


def generate_physics_combinations(num_combinations):
    combinations = set()
    while len(combinations) < num_combinations:
        num_concepts = random.randint(3, 6)
        sampled_concepts = tuple(random.sample(physics_concepts, num_concepts))
        combinations.add(sampled_concepts)
    return combinations


def write_jsonl(filename, combinations):
    with open(filename, "w") as f:
        for combo in combinations:
            concept_string = " ".join(combo)
            json.dump({"physics_concepts": concept_string}, f)
            f.write("\n")


# Generate non-overlapping combinations for train and validation sets
train_combinations = generate_physics_combinations(5000)
val_combinations = generate_physics_combinations(5000)

# Ensure no overlap between train and validation sets
val_combinations = val_combinations - train_combinations
while len(val_combinations) < 5000:
    num_concepts = random.randint(3, 6)
    new_combination = tuple(random.sample(physics_concepts, num_concepts))
    if new_combination not in train_combinations:
        val_combinations.add(new_combination)

# Write train and validation sets to separate files
write_jsonl("evals/datasets/train_physics.jsonl", train_combinations)
write_jsonl("evals/datasets/val_physics.jsonl", val_combinations)

print("JSONL file 'train_physics.jsonl' has been created with 5000 rows.")
print("JSONL file 'val_physics.jsonl' has been created with 5000 rows.")
print("The train and validation sets do not overlap.")
