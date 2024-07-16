import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from string import Template

import pandas as pd
from tqdm import tqdm

from evals.apis.inference.api import InferenceAPI
from evals.data_models.inference import LLMParams
from evals.data_models.messages import ChatMessage, Prompt, PromptTemplate
from evals.utils import setup_environment, gather_max_par
from evals.analysis.loading_data import get_data_path, load_single_df
from evals.load.lazy_object_level_llm_extraction import lazy_add_response_property_to_object_level

LOGGER = logging.getLogger(__name__)

async def get_property(
    response: str,
    prompt_template: PromptTemplate,
    llm_params: LLMParams,
    inference_api: InferenceAPI
) -> str:
    messages = [
        ChatMessage(role=m.role, content=m.content.replace('${response}', response))
        for m in prompt_template.messages
    ]
    prompt = Prompt(messages=messages)

    try:
        responses = await inference_api(
            model_ids=llm_params.model,
            prompt=prompt,
            temperature=llm_params.temperature,
            max_tokens=llm_params.max_tokens,
            top_p=llm_params.top_p,
            num_candidates_per_completion=llm_params.num_candidates_per_completion,
            insufficient_valids_behaviour=llm_params.insufficient_valids_behaviour,
            logprobs=llm_params.logprobs,
            seed=llm_params.seed,
        )
        return responses[0].completion
    except Exception as e:
        LOGGER.warning(f"Failed to get property: {e}")
        return ""

async def process_dataset_for_property(
    df: pd.DataFrame,
    property_name: str,
    prompt_template: PromptTemplate,
    llm_params: LLMParams,
    inference_api: InferenceAPI
) -> pd.DataFrame:
    tasks = [
        get_property(row['response'], prompt_template, llm_params, inference_api)
        for _, row in df.iterrows()
    ]
    properties = await gather_max_par(100, *tasks)
    
    df[property_name] = properties
    return df

def generate_finetuning_data(
    df: pd.DataFrame,
    name: str,
    response_property_name: str,
    n_items: Optional[int],
    seed: int,
    scramble: bool,
    prompt: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if n_items and n_items < len(df):
        df = df.sample(n_items, random_state=seed, replace=False)

    if scramble:
        LOGGER.info("Scrambling the input 🎲.")
        df = scramble_strings(df, seed)

    df = df.dropna(subset=[response_property_name])
    df = df[df[response_property_name] != "nan"]

    prompt_template = PromptTemplate(**prompt)
    data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating messages for {name}"):
        try:
            prompt = process_prompt(row, prompt_template, response_property_name)
            prompt_dict = json.loads(prompt.openai_finetuning_format())
            prompt_dict["string"] = row["string"]
            data.append(prompt_dict)
        except Exception as e:
            LOGGER.warning(f"Failed to process row: {e}")

    return data

def process_prompt(row: pd.Series, prompt_template: PromptTemplate, response_property_name: str) -> Prompt:
    messages = []
    for message in prompt_template.messages:
        t = Template(message.content)
        content = t.safe_substitute(response=row[response_property_name])
        messages.append(ChatMessage(role=message.role, content=content))
    return Prompt(messages=messages)

def scramble_strings(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    columns_to_scramble = [col for col in ["string", "unmodified_string", "modified_string"] if col in df.columns]
    for col in columns_to_scramble:
        df[col] = df[col].sample(frac=1, random_state=seed).values
    return df

async def main(
    train_base_dir: str,
    val_base_dir: Optional[str],
    response_property_name: str,
    n_train_items: Optional[int],
    n_val_items: Optional[int],
    seed: int,
    scramble: bool,
    prompt: Dict[str, Any],
    property_extraction_prompt: PromptTemplate,
    llm_params: LLMParams,
    anthropic_api_key: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Setup
    setup_environment(anthropic_api_key=anthropic_api_key)
    inference_api = InferenceAPI()

    # Load and process train data
    train_datapath = get_data_path(train_base_dir)
    train_df = load_single_df(train_datapath)
    train_df = lazy_add_response_property_to_object_level(
        train_df, get_hydra_config(train_datapath.parent), response_property_name
    )

    # Extract property if not already present
    if response_property_name not in train_df.columns:
        LOGGER.info(f"Extracting property {response_property_name} for train data")
        train_df = await process_dataset_for_property(
            train_df, response_property_name, property_extraction_prompt, llm_params, inference_api
        )

    # Generate train finetuning data
    train_data = generate_finetuning_data(
        train_df, "train", response_property_name, n_train_items, seed, scramble, prompt
    )

    # Load and process val data if provided
    val_data = []
    if val_base_dir:
        val_datapath = get_data_path(val_base_dir)
        val_df = load_single_df(val_datapath)

        # Extract property if not already present
        if response_property_name not in val_df.columns:
            LOGGER.info(f"Extracting property {response_property_name} for validation data")
            val_df = await process_dataset_for_property(
                val_df, response_property_name, property_extraction_prompt, llm_params, inference_api
            )

        # Generate val finetuning data
        val_data = generate_finetuning_data(
            val_df, "validation", response_property_name, n_val_items, seed, scramble, prompt
        )

    LOGGER.info(f"Generated {len(train_data)} training samples and {len(val_data)} validation samples")
    return train_data, val_data

if __name__ == "__main__":
    # Example usage
    property_extraction_prompt = PromptTemplate(messages=[
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Extract the main topic from this response: ${response}")
    ])
    
    llm_params = LLMParams(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=50,
        top_p=1,
        num_candidates_per_completion=1,
    )

    finetuning_prompt = {
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Please respond to: ${response}'}
        ]
    }

    train_data, val_data = asyncio.run(main(
        train_base_dir="./train_data",
        val_base_dir="./val_data",
        response_property_name="main_topic",
        n_train_items=1000,
        n_val_items=200,
        seed=42,
        scramble=False,
        prompt=finetuning_prompt,
        property_extraction_prompt=property_extraction_prompt,
        llm_params=llm_params,
        anthropic_api_key="your_anthropic_api_key_here"  # Optional
    ))

    print(f"Generated {len(train_data)} training samples and {len(val_data)} validation samples")
    # You can now use train_data and val_data as needed