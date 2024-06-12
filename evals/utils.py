import importlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import List

import omegaconf
import openai
import yaml
from tenacity import retry, retry_if_result, stop_after_attempt
from vertexai.preview.tuning import sft

from evals.locations import EXP_DIR

LOGGER = logging.getLogger(__name__)

LOGGING_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

GCLOUD_PROJECT = "roots-api-1475521819980"
GCLOUD_LOCATION = "us-central1"
GCLOUD_BUCKET = "cloud-ai-platform-6e5ab5cb-3fca-49e0-a42c-ce00ed910490"

GEMINI_MODELS = {
    "gemini-1.0-pro-001",
    "gemini-1.0-pro-002",
    "gemini-1.5-pro-001",
}

COMPLETION_MODELS = {
    "davinci-002",
    "babbage-002",
    "text-davinci-003",
    "text-davinci-002",
    "gpt-4-base",
    "gpt-3.5-turbo-instruct",
}

_GPT_4_MODELS = [
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
]
_GPT_TURBO_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
]
GPT_CHAT_MODELS = set(_GPT_4_MODELS + _GPT_TURBO_MODELS)

MODEL_TO_FAMILY_MAP = {}
MODEL_TO_FAMILY_MAP.update({model: "openai" for model in GPT_CHAT_MODELS})
MODEL_TO_FAMILY_MAP.update({model: "gemini" for model in GEMINI_MODELS})


def setup_environment(
    anthropic_tag: str = "ANTHROPIC_API_KEY",
    logging_level: str = "info",
    openai_tag: str = "OPENAI_API_KEY",
):
    setup_logging(logging_level)
    secrets = load_secrets(Path(__file__).parent.parent / "SECRETS")
    openai.api_key = secrets[openai_tag]
    os.environ["ANTHROPIC_API_KEY"] = secrets[anthropic_tag]


def setup_logging(logging_level):
    level = LOGGING_LEVELS.get(logging_level.lower(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Disable logging from openai
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)


def load_secrets(file_path):
    secrets = {}
    with open(file_path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            secrets[key] = value
    return secrets


def load_yaml(file_path):
    with open(file_path) as f:
        content = yaml.safe_load(f)
    return content


def save_yaml(file_path, data):
    with open(file_path, "w") as f:
        yaml.dump(data, f)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data


def save_jsonl(file_path, data):
    with open(file_path, "w") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f)


@retry(
    stop=stop_after_attempt(8),
    retry=retry_if_result(lambda result: result is not True),
)
def function_with_retry(function, *args, **kwargs):
    return function(*args, **kwargs)


@retry(
    stop=stop_after_attempt(8),
    retry=retry_if_result(lambda result: result is not True),
)
async def async_function_with_retry(function, *args, **kwargs):
    return await function(*args, **kwargs)


def get_maybe_nested_from_dict(d, keys):
    """Helper function to get a value from a nested dictionary."""
    try:
        if isinstance(keys, str):
            keys = [keys]
        if len(keys) == 1:
            return d[keys[0]]
        else:
            return get_maybe_nested_from_dict(d[keys[0]], keys[1:])
    except KeyError:
        return None


def load_string_and_reponse_functions(string_modifier, response_property):
    if string_modifier == "None":
        string_modifier = None
    if response_property == "None":
        response_property = None
    if string_modifier is not None:
        string_modifier = import_function_from_string("evals.string_modifier", string_modifier)
        LOGGER.info(f"Loaded string modifier function {string_modifier.__name__} from evals.string_modifier")
    if response_property is not None:
        response_property = import_function_from_string("evals.response_property", response_property)
        LOGGER.info(f"Loaded output property function {response_property.__name__} from evals.response_property")
    return string_modifier, response_property


def import_function_from_string(module_name, function_name):
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function


def sanitize_folder_name(key: str) -> str:
    """Sometimes we need to clean Hydra keys so we can use them as folder names. This function does that."""
    # Replace ":" with "_"
    sanitized_key = key.replace(":", "_")
    # Replace "/" with "_"
    sanitized_key = sanitized_key.replace("meta_level/", "meta_level_")
    sanitized_key = sanitized_key.replace("object_level/", "object_level_")
    # This is common prefix for gemini endpoint names
    sanitized_key = sanitized_key.replace(
        "projects/351298396653/locations/us-central1/endpoints/",
        "projects_351298396653_locations_us-central1_endpoints_",
    )
    return sanitized_key


# ensure that the sanitize function is registered
omegaconf.OmegaConf.register_new_resolver("sanitize", sanitize_folder_name)


# helper function to find the experiment folder
def experiment_folder_location(subfolder):
    return str(EXP_DIR) + subfolder


omegaconf.OmegaConf.register_new_resolver("experiment_folder_location", experiment_folder_location)


def get_current_git_hash():
    try:
        # Run the git command to get the current commit hash
        output = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return output
    except subprocess.CalledProcessError:
        return None


def run_command(command):
    """Execute the given command in the shell, stream the output, and return the last line.
    Useful for running the `run_...` functions."""
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        output_lines = []
        for line in process.stdout:
            print(line, end="")  # stream the output to the command line
            output_lines.append(line.strip())

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        last_line = output_lines[-1] if output_lines else ""
        print(f"Successfully executed: {command}")
        return last_line

    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e}")
        raise e


def get_hparams_for_endpoints(endpoint_names):
    """
    E.g.
    Args: ["projects/351298396653/locations/us-central1/endpoints/5520944751202795520"]
    Returns: [{"epochCount": 1, "learningRateMultiplier": 0.1, "adapterSize": "ADAPTERSIZE_SIXTEEN"}]
    """
    responses: List[sft.SupervisedTuningJob] = sft.SupervisedTuningJob.list()
    out = []
    for response in responses:
        for endpoint in endpoint_names:
            if response.tuned_model_endpoint_name == endpoint:
                hp = response.to_dict()["supervisedTuningSpec"]["hyperParameters"]
                out.append(hp)
    return out
