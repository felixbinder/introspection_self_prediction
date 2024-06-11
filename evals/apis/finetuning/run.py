import datetime
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import openai
from google.cloud import storage
from openai.error import APIConnectionError, RateLimitError
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from vertexai.preview.tuning import sft

from evals.apis.finetuning.syncer import WandbSyncer
from evals.utils import (
    COMPLETION_MODELS,
    GCLOUD_BUCKET,
    GCLOUD_PROJECT,
    GPT_CHAT_MODELS,
    load_jsonl,
)

logger = logging.getLogger(__name__)

GEMINI_VALIDATION_LIMIT = 256


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
    # Job id, for Gemini this is tuning job ID
    # e.g. projects/351298396653/locations/us-central1/tuningJobs/4314075193183043584
    id: str


class FinetunedJobResults(BaseModel):
    fine_tuned_model: str
    result_files: list[str] | None = []
    trained_tokens: int | None


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


@retry(
    # Retry if we get a rate limit error
    # Also retry if we get an API connection error - e.g. when you close your laptop lol
    retry=retry_if_exception_type((APIConnectionError, openai.APIError)),
    wait=wait_fixed(30),
)
def wait_until_finetune_job_is_ready(ft_job: FinetuneJob) -> FinetunedJobResults:
    """Returns the fine tuned model id"""
    if ft_job.model in (COMPLETION_MODELS | GPT_CHAT_MODELS):
        while True:
            finetune_job = openai.FineTuningJob.retrieve(ft_job.id)
            if finetune_job["status"] == "succeeded":
                print(finetune_job)
                return FinetunedJobResults.parse_obj(finetune_job)
            time.sleep(1)
    # TODO: add timeout?
    elif ft_job.model == "gemini-1.0-pro-002":
        sft_tuning_job = sft.SupervisedTuningJob(ft_job.id)
        print("Running Gemini SFT job:\n", sft_tuning_job.to_dict())
        while not sft_tuning_job.has_ended:
            print("Waiting for job to finish...")
            time.sleep(60)
            sft_tuning_job.refresh()
        if sft_tuning_job.state == "JOB_STATE_FAILED":
            raise Exception(f"Job failed: {sft_tuning_job.to_dict()}")
        # Google does not have API access to training metrics
        return FinetunedJobResults(
            fine_tuned_model=sft_tuning_job.tuned_model_endpoint_name, result_files=None, trained_tokens=None
        )
    else:
        raise ValueError(f"Model {ft_job.id} not supported")


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
        if model in (COMPLETION_MODELS | GPT_CHAT_MODELS):
            finetune_job_resp = openai.FineTuningJob.create(
                training_file=file_id,
                model=model,
                hyperparameters=hyperparameters,
                suffix=suffix,
                validation_file=val_file_id,
                organization=organization,
                seed=seed,
            )
            parsed_job_resp: FinetuneJob = FinetuneJob.parse_obj(finetune_job_resp)
        elif model == "gemini-1.0-pro-002":
            finetune_job_resp = sft.train(
                source_model=model,
                train_dataset=file_id,  # file ID is the gsutil URI
                validation_dataset=val_file_id,
                epochs=hyperparameters.get("n_epochs", None),
                learning_rate_multiplier=hyperparameters.get("learning_rate_multiplier", None),
                tuned_model_display_name=f"gemini-1.0-pro-002:{suffix}",
            ).to_dict()
            parsed_job_resp: FinetuneJob = FinetuneJob(
                model=finetune_job_resp["baseModel"], id=finetune_job_resp["name"]
            )
        else:
            raise ValueError(f"Model {model} not supported")
    except RateLimitError:  # TODO: rate limit errors for gemini finetuning not well documented
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

    return parsed_job_resp


def upload_to_gcloud_bucket(data_path: Path, file_name: str):
    storage_client = storage.Client(project=GCLOUD_PROJECT)
    bucket = storage_client.bucket(GCLOUD_BUCKET)
    destination_name = os.path.join("instrospection-astra", file_name)
    blob = bucket.blob(destination_name)
    blob.upload_from_filename(data_path)
    print(f"File {data_path.name} uploaded to {destination_name}.")
    uri = f"gs://{GCLOUD_BUCKET}/{destination_name}"
    print(f"File ID is gsutil URI: {uri}")
    return uri


def upload_file(data_path: Path, params: FineTuneParams, limit=None):
    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = f"{params.model}-{now_time}_{data_path.name}"
    data_path = filter_file_for_finetuning(data_path, limit=limit)
    print(f"Starting file upload.\n{file_name}")
    if params.model in (COMPLETION_MODELS | GPT_CHAT_MODELS):
        print("Uploading to openai")
        file_upload_resp: dict[str, Any] = openai.File.create(  # type: ignore[reportGeneralTypeIssues]
            file=open(data_path, "rb"),
            purpose="fine-tune",
            user_provided_filename=file_name,
        )
        file_id = file_upload_resp["id"]
        wait_until_uploaded_file_id_is_ready(file_id=file_id)
    elif params.model == "gemini-1.0-pro-002":
        print("Uploading to gcloud")
        file_id = upload_to_gcloud_bucket(data_path, file_name)
    else:
        raise ValueError(f"Model {params.model} not supported")
    print(f"Uploaded file.\n{file_name}\n{file_id}")
    return file_id


def filter_file_for_finetuning(data_path: Path, limit=None):
    """The .json file for OpenAI is only allowed to have the key "`messages` and no others. We load in the file, and save out a temp file with only the messages key."""
    data = load_jsonl(data_path)
    if limit is not None:
        random.seed(25)
        data = random.sample(data, min(len(data), limit))
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
        syncer.update_file_id(file_id=file_id)

    if val_data_path:
        limit = GEMINI_VALIDATION_LIMIT if params.model == "gemini-1.0-pro-002" else None
        val_file_id = upload_file(data_path=val_data_path, params=params, limit=limit)
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
    result: FinetunedJobResults = wait_until_finetune_job_is_ready(ft_job=finetune_job_resp)
    model_id = result.fine_tuned_model
    print(f"Fine tuned model id: {model_id}. You can now use this model in the API")
    if syncer:
        syncer.update_finetune_model_id(finetune_model_id=model_id)
        if result.result_files:
            syncer.update_training_results(results_id=result.result_files[0])
        syncer.end()
    return model_id
