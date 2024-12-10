# Looking Inward: Language Models Can Learn About Themselves by Introspection

This repository holds code for the paper [Looking Inward: Language Models Can Learn About Themselves by Introspection]([https://openreview.net/pdf?id=eb5pkwIB5i](https://arxiv.org/abs/2410.13787)).

We've released the dataset [here](https://huggingface.co/datasets/thejaminator/introspection_self_predict).

## Overview

This repository contains the `run.py` script and associated files for conducting evaluations using LLMs from the Anthroppic and OpenAI APIs. It is designed to handle tasks such as generating responses to prompts, caching results, and managing API interactions efficiently.

## Setup

### Prerequisites

- Python 3.11
- Virtual environment tool (e.g., virtualenv)
- gcloud CLI (for Gemini models)

### Installation

1. **Create and Activate a Virtual Environment:**
  ```bash
  virtualenv --python python3.11 .venv
  source .venv/bin/activate
  ```
2. Install the package:
  ```bash
  pip install -e .
  ```
  The package can then be imported as `evals` in your Python code.
  Make sure that you're using the right version of pip and python. You can check this by running `which pip` and `which python` and making sure they point to the right locations.

3. Install Required Packages:
  ```bash
  pip install -r requirements.txt
  ```
4. Install Pre-Commit Hooks:
  ```bash
  make hooks
  ```
5. Create a SECRETS file
  ```bash
  touch SECRETS
  echo OPENAI_API_KEY=<INSERT_HERE> >> SECRETS
  echo ANTHROPIC_API_KEY=<INSERT_HERE> >> SECRETS
  echo DEFAULT_ORG=org-<INSERT_HERE> >> SECRETS
  ```

### Gemini setup

1. Ask team get added to the project
2. Install gcloud CLI. Configure with project=roots-api-1475521819980, location=us-central1
3. Follow instructions for authenticating to Vertex: (Authenticate to Vertex AI  |  Google Cloud)[https://cloud.google.com/vertex-ai/docs/authentication]


## Usage
The best starting point to see how inference and finetuning works is to use the `sweep_full_study` script. This produces a `state.json` file that lists all individual commands that make up an experiment.

```bash
python -m scripts.sweep_full_study \
--study_name="full_sweep_test" \
--model_configs="gpt-3.5-turbo" \
--val_only_model_configs="gpt-4o" \
--tasks='{"wikipedia": ["identity", "sentiment"]}' \
--prompt_configs="minimal" \
--val_tasks='{"number_triplets": ["identity", "sentiment"]}' \
--n_object_train=1000 \
--n_object_val=250 \
--n_meta_val=50
```

### Running Finetuning

#### Creating a JSONL File
The JSONL files are created using the `evals.apis.finetuning.create_dataset.py` file.
Pass a path to a folder containing config files for the dataset you want to create. The config files should be in the following format:
```yaml
name: number_triplets
base_dir: exp/number_triplets_azalea/base_gpt-3.5-turbo-0125_base-completion-azalea-system_prompt_number_triplets_dataset

defaults:
  - dataset: number_triplets
  - prompt: base_completion_azalea_system # using which prompt?

dataset:
  num: 100 # how many strings to generate? None for all in the base_dir
  response_property: None # When seeding the strings, extract the property from the string and score against it. Use `None` or any function from evals/response_property.py. Remember to change the prompt!
  string_modifier: None # Require that the string has to be reconstructed. None or any function from evals/string_modification.py. Remember to change the prompt!
```
The dataset will be saved out into the folder.

- **Basic Run:**
    Prepare a jsonl file according to the openai format and run the following command:
    ```bash
    for n_epochs in 4 8; do python3 -m evals.apis.finetuning.run $jsonl_path --n_epochs $n_epochs --notes test_run --no-ask_to_validate_training --organization FARAI_ORG; done
    ```
- **Use the CLI:**
    There are a few helper functions to do things like list all the files on the server and delete files if it gets full.
    ```bash
    python3 -m evals.apis.finetuning.cli list_all_files --organization FARAI_ORG
    python3 -m evals.apis.finetuning.cli delete_all_files --organization FARAI_ORG
    ```
- **Set-up Weights and Biases:**
    You can set up weights and biases to log your finetuning runs. You will need to set up a weights and biases account and then run the following command:
    ```bash
    wandb login
    ```
    You can then run the finetuning script with the `--use_wandb` flag to log your runs. You will need to provide the project name via `--project_name` too.

## Features

- **Hydra for Configuration Management:**
  Hydra enables easy overriding of configuration variables. Use `++` for overrides. You can reference other variables within variables using `${var}` syntax.

- **Caching Mechanism:**
  Caches prompt calls to avoid redundant API calls. Cache location defaults to `$exp_dir/cache`. This means you can kill your run anytime and restart it without worrying about wasting API calls.

- **Prompt History Logging:**
  For debugging, human-readable `.txt` files are stored in `$exp_dir/prompt_history`, timestamped for easy reference.

- **LLM Inference API Enhancements:**
  - Ability to double the rate limit if you pass a list of models e.g. ["gpt-3.5-turbo", "gpt-3.5-turbo-0613"]
  - Manages rate limits efficiently, bypassing the need for exponential backoff.
  - Allows custom filtering of responses via `is_valid` function.
  - Provides a running total of cost and model timings for performance analysis.
  - Utilise maximum rate limit by setting `max_tokens=None` for OpenAI models.

- **Logging finetuning runs with Weights and Biases:**
  - Logs finetuning runs with Weights and Biases for easy tracking of experiments.

- **Usage Tracking:**
  - Tracks usage of OpenAI and Anthropic APIs so you know how much they are being utilised within your organisation.

## Repository Structure

- `evals/run.py`: Main script for evaluations.
- `evals/apis/inference`: Directory containing modules for LLM inference
- `evals/apis/finetuning`: Directory containing scripts to finetune OpenAI models and log with weights and biases
- `evals/apis/usage`: Directory containing two scripts to get usage information from OpenAI and Anthropic
- `evals/conf`: Directory containing configuration files for Hydra. Check out `prompt` and `language_model` for examples of how to create useful configs.
- `evals/data_models`: Directory containing Pydantic data models
- `evals/load`: Directory containing code to download and process MMLU
- `tests`: Directory containing unit tests
- `scripts`: Example scripts on how to run sweep experiments

---
