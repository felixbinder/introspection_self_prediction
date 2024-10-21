import os
from typing import List, Optional

import dotenv
from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi
from pydantic import BaseModel, TypeAdapter
from slist import Slist

from evals.analysis.james.james_analysis import single_comparison_flat
from evals.analysis.james.object_meta import ObjectAndMeta
from evals.locations import EXP_DIR

dotenv.load_dotenv()

# Define your Hugging Face repository details
HF_REPO_NAME = "thejaminator/introspection_self_predict"  # Replace with your desired repository name
HF_PRIVATE = False  # Set to True if you want the dataset to be private


class ChatMessage(BaseModel):
    role: str
    content: str


rename_map = {
    "matches_target": "ethical_stance",
    "is_either_a_or_c": "among_a_or_c",
    "is_either_b_or_d": "among_b_or_d",
}


class DataRow(BaseModel):
    original_question: str
    original_dataset: str
    object_level_prompt: str
    hypothetical_prompt: str
    behavioral_property: str
    option_matching_ethical_stance: Optional[str]

    def rename_properties(self) -> "DataRow":
        new = self.model_copy()
        if new.behavioral_property in rename_map:
            new.behavioral_property = rename_map[new.behavioral_property]
        return new


def parsed_string_into_list_chat_message(parsed: str) -> List[ChatMessage]:
    ta = TypeAdapter(dict[str, List[ChatMessage]])
    result = ta.validate_json(parsed)
    assert "messages" in result
    return result["messages"]


def remove_redundant_sys_prompt(chat: List[ChatMessage]) -> List[ChatMessage]:
    # Remove the empty system prompts
    return [x for x in chat if x.content != ""]


def first_user_message(chat: List[ChatMessage]) -> str:
    first = [x for x in chat if x.role == "user"][0]
    return first.content


def convert_to_dump(test_data: ObjectAndMeta) -> DataRow:
    object_full = test_data.object_full_prompt
    assert object_full is not None, "object_full_prompt is None"
    hypo_full = test_data.meta_full_prompt
    assert hypo_full is not None, "meta_full_prompt is None"

    return DataRow(
        original_question=test_data.string,
        original_dataset=test_data.task,
        object_level_prompt=first_user_message(parsed_string_into_list_chat_message(object_full)),
        hypothetical_prompt=first_user_message(parsed_string_into_list_chat_message(hypo_full)),
        behavioral_property=test_data.response_property,
        option_matching_ethical_stance=test_data.target if test_data.response_property == "matches_target" else None,
    )


def dump_all() -> Slist[DataRow]:
    """
    Process and convert all test data into a list of DataRow instances.
    """
    only_tasks = {
        "survival_instinct",
        "myopic_reward",
        "animals_long",
        "mmlu_non_cot",
        "stories_sentences",
        "english_words_long",
    }

    exp_folder = EXP_DIR / "test_dump"

    results: Slist[ObjectAndMeta] = single_comparison_flat(
        object_model="test",
        meta_model="test",
        exp_folder=exp_folder,
        only_tasks=only_tasks,
        exclude_noncompliant=False,
    )

    # Convert and rename properties
    converted: Slist[DataRow] = results.map(convert_to_dump).map(lambda x: x.rename_properties())

    return converted


def create_hf_dataset(data: List[DataRow], is_ethical_stance: bool) -> DatasetDict:
    """
    Convert a list of DataRow instances into a Hugging Face DatasetDict.
    """
    # Convert DataRow instances to dictionaries
    data_dicts = [row.dict() for row in data]

    if is_ethical_stance:
        features = Features(
            {
                "original_question": Value("string"),
                "original_dataset": Value("string"),
                "object_level_prompt": Value("string"),
                "hypothetical_prompt": Value("string"),
                "behavioral_property": Value("string"),
                "option_matching_ethical_stance": Value("string"),
            }
        )
    else:
        features = Features(
            {
                "original_question": Value("string"),
                "original_dataset": Value("string"),
                "object_level_prompt": Value("string"),
                "hypothetical_prompt": Value("string"),
                "behavioral_property": Value("string"),
            }
        )

    # Create a Dataset
    dataset = Dataset.from_list(data_dicts, features=features)

    # Organize into a DatasetDict with 'test' split
    dataset_dict = DatasetDict({"test": dataset})

    return dataset_dict


def push_to_huggingface(dataset_dict: DatasetDict, config: str, repo_name: str, private: bool = False):
    """
    Push the DatasetDict to the Hugging Face Hub.
    """
    # Ensure you have set your Hugging Face token as an environment variable or use HfFolder
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise EnvironmentError("Please set the HUGGINGFACE_TOKEN environment variable.")

    # Initialize the API
    api = HfApi()

    # Create the repository if it doesn't exist
    try:
        api.create_repo(repo_id=repo_name, private=private, exist_ok=True)
        print(f"Repository '{repo_name}' created or already exists.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        raise e

    # Push the dataset
    dataset_dict.push_to_hub(repo_name, config_name=config, token=hf_token)
    print(f"Dataset pushed to Hugging Face Hub at https://huggingface.co/datasets/{repo_name}")


def main():
    # Step 1: Process and convert data
    print("Processing data...")
    converted_data = dump_all()
    print(f"Total records processed: {len(converted_data)}")

    # group by behavioral property
    grouped = converted_data.group_by(lambda x: x.behavioral_property)

    for behavioral_property, group in grouped:
        print(f"Behavioral Property: {behavioral_property}")
        print(f"Total records: {len(group)}")

        # Step 2: Create Hugging Face Dataset
        print("Creating Hugging Face Dataset...")
        is_ethical_stance = behavioral_property == "ethical_stance"
        hf_dataset = create_hf_dataset(group, is_ethical_stance)
        print("Hugging Face Dataset created.")

        # Step 3: Push to Hugging Face Hub
        print("Pushing dataset to Hugging Face Hub...")
        push_to_huggingface(hf_dataset, config=behavioral_property, repo_name=HF_REPO_NAME, private=HF_PRIVATE)
        print("Dataset successfully pushed.")


if __name__ == "__main__":
    main()
