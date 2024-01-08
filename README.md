# README for Evaluation Scripts Repository

## Overview

This repository contains the `run.py` script and associated files for conducting evaluations using machine learning models, specifically focusing on language models like GPT-3.5. It is designed to handle tasks such as generating responses to prompts, caching results, and managing API interactions efficiently.

## Setup

### Prerequisites

- Python 3.11
- Virtual environment tool (e.g., virtualenv)

### Installation

1. **Create and Activate a Virtual Environment:**
    ```bash
    virtualenv --python python3.11 .venv
    source .venv/bin/activate
    ```
2. Install Required Packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Install Pre-Commit Hooks (Optional):
    ```bash
    make hooks
    ```

## Usage
### Running the Script

- **Basic Run on the MMLU Dataset:**
    ```bash
    python3 -m evals.run ++exp_dir=exp/test_run
    ```
- **Advanced Usage with Overrides:**
    To test a different model, limit to 5 samples, and print output:
    ```bash
    python3 -m evals.run ++exp_dir=exp/test_run2 ++limit=5 ++print_prompt_and_response=true ++language_model.model=gpt-3.5-turbo ++reset=true
    ```

### Features

- **Hydra for Configuration Management:**
  Hydra enables easy overriding of configuration variables. Use `++` for overrides. You can reference other variables within variables using `${var}` syntax.

- **Caching Mechanism:**
  Caches prompt calls to avoid redundant API calls. Cache location defaults to `$exp_dir/cache`.

- **Prompt History Logging:**
  For debugging, human-readable `.txt` files are stored in `$exp_dir/prompt_history`, timestamped for easy reference.

- **LLM Inference API Enhancements:**
  - Ability to double the rate limit if you pass a list of models e.g. ["gpt-3.5-turbo", "gpt-3.5-turbo-0613"]
  - Manages rate limits efficiently, bypassing the need for exponential backoff.
  - Allows custom filtering of responses via `is_valid` function.
  - Provides a running total of cost and model timings for performance analysis.
  - Utilise maximum rate limit by setting `max_tokens=None` for OpenAI models.

## Repository Structure

- `run.py`: Main script for evaluations.
- `requirements.txt`: List of Python packages required.
- `Makefile`: Contains commands for setting up pre-commit hooks and other utilities.
- `evals/`: Directory containing modules for API interactions, data processing, and utility functions.

## Contributing

Contributions to this repository are welcome. Please follow the standard procedures for submitting issues and pull requests.

## License

Specify your licensing information here.

---
