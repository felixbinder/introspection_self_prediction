import asyncio
import importlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Awaitable, List, TypeVar

import omegaconf
import openai
import pandas as pd
import yaml
from tenacity import retry, retry_if_result, stop_after_attempt

from evals.locations import EXP_DIR

LOGGER = logging.getLogger(__name__)

MAX_RESPONSE_LEN_FOR_MODE = 40  # number of characters before truncation is applied in the mode of N sampling

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
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
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
    os.environ["RUNPOD_API_KEY"] = secrets.get("RUNPOD_API_KEY", "")
    os.environ["FIREWORKS_API_KEY"] = secrets.get("FIREWORKS_API_KEY", "")


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
    from vertexai.preview.tuning import sft

    responses: List[sft.SupervisedTuningJob] = sft.SupervisedTuningJob.list()
    out = []
    for response in responses:
        for endpoint in endpoint_names:
            if response.tuned_model_endpoint_name == endpoint:
                hp = response.to_dict()["supervisedTuningSpec"]["hyperParameters"]
                out.append(hp)
    return out


def collate_mode_of_n(data0_path: Path, overwrite: bool = False):
    """After the data is collected, we need to group the data into a dataframe that contains only the modal response for multiple samples.

    Args:
        data0_path (Path): Path to the data file (ie. `raw_data0.csv`)
        overwrite (bool, optional): Whether to overwrite the existing `data0.csv` file. Defaults to False.

    Writes out a `data0.csv` file with the modal response for each sample.
    If a string has only unique responses, it is discarded.
    """
    assert data0_path.exists(), f"Data file {data0_path} does not exist."
    assert "raw_data" in data0_path.stem, f"Data file {data0_path} does not contain 'raw_data'."
    out_path = data0_path.parent / f"{data0_path.stem.replace('raw_data', 'data')}.csv"
    if out_path.exists() and not overwrite:
        LOGGER.info(f"File {out_path} already exists. Will not overwrite with mode-of-N data.")
        return
    df = pd.read_csv(data0_path)
    strings = df["string"].unique()
    df["trunc_response"] = (
        df["response"].astype(str).str.slice(0, MAX_RESPONSE_LEN_FOR_MODE)
    )  # we truncate responses since long responses are likely to be non-deterministic
    modal_rows = []
    skipped_strings = []
    for string in strings:
        string_df = df[df["string"] == string]
        if len(string_df) > 1 and string_df["trunc_response"].nunique() == len(string_df["trunc_response"]):
            # Skip strings that have only unique responses unless they are the only response
            skipped_strings.append(string)
            continue
        # pull the first row that has the modal response (so that we get the logprobs etc.)
        # we know from above that there is at least one row
        modal_row = string_df[string_df["trunc_response"] == string_df["trunc_response"].mode()[0]].iloc[0]
        modal_rows.append(modal_row)
    modal_df = pd.DataFrame(modal_rows)
    modal_df.to_csv(out_path, index=False)
    if len(skipped_strings) > 0:
        LOGGER.warning(
            f"Skipped strings the following strings since no modal answer could be extracted: {skipped_strings}"
        )
    if max([len(str(s)) for s in df["response"]]) > MAX_RESPONSE_LEN_FOR_MODE:
        LOGGER.warning(f"Some responses were truncated to {MAX_RESPONSE_LEN_FOR_MODE} characters.")
    LOGGER.info(
        f"Saved modal responses to {out_path}. {len(skipped_strings)} strings were skipped because all responses were unique."
    )


def safe_model_name(model_name):
    """Returns a canonical safe model name from either path or OpenAI model ID."""
    return model_name.replace("/", "-").replace(":", "_")


T = TypeVar("T")


async def gather_max_par(max_par: int, *coros_or_futures: Awaitable[T]) -> List[T]:
    semaphore = asyncio.Semaphore(max_par)

    async def semaphore_wrapper(coro_or_future: Awaitable[T]) -> T:
        async with semaphore:
            return await coro_or_future

    return await asyncio.gather(*[semaphore_wrapper(coro) for coro in coros_or_futures])
