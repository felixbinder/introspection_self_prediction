import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import openai
from openai.error import APIConnectionError, RateLimitError
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from evals.apis.finetuning.syncer import WandbSyncer
from evals.utils import load_jsonl

logger = logging.getLogger(__name__)


class FineTuneHyperParams(BaseModel):
    """
    https://platform.openai.com/docs/api-reference/fine-tuning/create
    """

    n_epochs: int | None = None  # None sets it auto.
    learning_rate_multiplier: float | None = None  # None sets it auto.
    batch_size: int | None = None  # None sets it auto.


class FineTuneParams(BaseModel):
    model: str
    suffix: str | None = None
    hyperparameters: FineTuneHyperParams
    seed: int | None = 0  # None sets it auto.


class FinetuneJob(BaseModel):
    model: str
    id: str  # job id


class FinetunedJobResults(BaseModel):
    fine_tuned_model: str
    result_files: list[str] = []
    trained_tokens: int


@retry(
    retry=retry_if_exception_type(APIConnectionError),
    stop=stop_after_attempt(8),
    wait=wait_fixed(30),
)
def wait_until_uploaded_file_id_is_ready(file_id: str) -> None:
    while True:
        file = openai.File.retrieve(file_id)
        if file["status"] == "processed":
            return
        time.sleep(1)


def wait_until_finetune_job_is_ready(finetune_job_id: str) -> FinetunedJobResults:
    """Returns the fine tuned model id"""
    while True:
        finetune_job = openai.FineTuningJob.retrieve(finetune_job_id)
        if finetune_job["status"] == "succeeded":
            print(finetune_job)
            return FinetunedJobResults.parse_obj(finetune_job)
        time.sleep(1)


def confirm_to_continue(file_path: Path) -> None:
    # nice string like /home/.../file.jsonl
    file_path_str = file_path.absolute().as_posix()
    print(f"About to upload {file_path_str}. Continue? (y/n)")
    response = input()
    while response not in ["y", "n"]:
        print(f"Please enter y or n. You entered {response}")
        response = input()
    if response == "n":
        exit(0)
    print("Continuing with upload")
    return None


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    stop=stop_after_attempt(8),
    wait=wait_fixed(30),
)
def queue_finetune(
    file_id: str,
    model: str,
    hyperparameters: FineTuneHyperParams,
    suffix: str = None,
    val_file_id: str = None,
    organization: Optional[str] = None,
    seed: Optional[int] = None,
    retries: int = 12,  # number of retries. 30 * 2^10 = 8.5 hours
    retry_time: int = 30,  # time to wait before retrying in seconds
) -> FinetuneJob:
    if retries == 0:
        raise Exception("Retries exhausted. Exiting")
    if seed is None:
        seed = int(time.time())
    # Keep retrying until we can queue the finetune job
    if not isinstance(hyperparameters, dict):
        hyperparameters = hyperparameters.dict()
    # filter out Nones
    hyperparameters = {k: v for k, v in hyperparameters.items() if v is not None}
    try:
        finetune_job_resp = openai.FineTuningJob.create(
            training_file=file_id,
            model=model,
            hyperparameters=hyperparameters,
            suffix=suffix,
            validation_file=val_file_id,
            organization=organization,
            seed=seed,
        )
    except RateLimitError:
        logger.error(f"Rate limit error. Retrying in {retry_time} seconds. {retries} retries left.")
        time.sleep(retry_time)
        retry_time *= 2  # exponential backoff
        return queue_finetune(
            file_id=file_id,
            model=model,
            hyperparameters=hyperparameters,
            suffix=suffix,
            val_file_id=val_file_id,
            organization=organization,
            retries=retries - 1,
            retry_time=retry_time,
            seed=seed,
        )

    print(f"Started finetune job. {finetune_job_resp}")
    parsed_job_resp: FinetuneJob = FinetuneJob.parse_obj(finetune_job_resp)
    return parsed_job_resp


def upload_file(data_path: Path, params: FineTuneParams):
    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = f"{params.model}-{now_time}_{data_path.name}"
    data_path = filter_file_for_finetuning(data_path)
    file_upload_resp: dict[str, Any] = openai.File.create(  # type: ignore[reportGeneralTypeIssues]
        file=open(data_path, "rb"),
        purpose="fine-tune",
        user_provided_filename=file_name,
    )
    file_id = file_upload_resp["id"]
    print(f"Starting file upload. {file_id}\n{file_name}")
    wait_until_uploaded_file_id_is_ready(file_id=file_id)
    print(f"Uploaded file to openai. {file_upload_resp}\n{file_name}")
    return file_id


def filter_file_for_finetuning(data_path: Path):
    """The .json file for OpenAI is only allowed to have the key "`messages` and no others. We load in the file, and save out a temp file with only the messages key."""
    data = load_jsonl(data_path)
    new_data = [{"messages": d["messages"]} for d in data]
    new_data_path = data_path.parent / "temp_filtered.jsonl"
    with open(new_data_path, "w") as f:
        for d in new_data:
            f.write(json.dumps(d) + "\n")
    return new_data_path


def run_finetune(
    params: FineTuneParams,
    data_path: Path,
    syncer: Optional[WandbSyncer] = None,
    ask_to_validate_training: bool = True,
    val_data_path: Optional[Path] = None,
    organisation: Optional[str] = None,
) -> str:
    """
    Pass syncer=None to disable wandb logging
    """
    samples = load_jsonl(data_path)
    if ask_to_validate_training:
        confirm_to_continue(data_path)
    if syncer:
        syncer.update_parameters(params=params.dict())
        syncer.upload_training_file(data_path)
        syncer.update_n_samples(n_samples=len(samples))

    file_id = upload_file(data_path=data_path, params=params)
    if syncer:
        syncer.update_openai_file_id(openai_file_id=file_id)

    if val_data_path:
        val_file_id = upload_file(data_path=val_data_path, params=params)
    else:
        val_file_id = None
    finetune_job_resp = queue_finetune(
        file_id=file_id,
        model=params.model,
        hyperparameters=params.hyperparameters,
        suffix=params.suffix,
        val_file_id=val_file_id,
        organization=organisation,
        seed=params.seed,
    )
    print(f"Started finetune job. {finetune_job_resp}")

    if syncer:
        syncer.update_finetune_job_id(finetune_job_id=finetune_job_resp.id)
    result: FinetunedJobResults = wait_until_finetune_job_is_ready(finetune_job_id=finetune_job_resp.id)
    model_id = result.fine_tuned_model
    print(f"Fine tuned model id: {model_id}. You can now use this model in the API")
    if syncer:
        syncer.update_finetune_model_id(finetune_model_id=model_id)
        syncer.update_training_results(results_id=result.result_files[0])
        syncer.end()
    return model_id
