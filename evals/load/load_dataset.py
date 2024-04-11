import csv
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_dataset(
    path: str | Path,
    seed: int = 0,
    shuffle: bool = True,
    n: Optional[int] = None,
    filter_strings_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Load a dataset from a file.

    Args:
        path: The path to the jsonl file.
        seed: The random seed to use for shuffling the dataset.
        shuffle: Whether to shuffle the dataset.
        n: The number of rows to load from the dataset.

    Returns:
        A pandas DataFrame containing the dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at {path}")

    assert path.suffix == ".jsonl", f"Dataset file must be a .jsonl file: {path}. Got {path.suffix} instead."

    LOGGER.info(f"Loading dataset from {path}")
    df = pd.read_json(path, lines=True)

    assert not df.empty, f"Dataset is empty: {path}"
    assert "string" in df.columns, f"Dataset does not contain a 'string' column: {path}"

    if filter_strings_path is not None and filter_strings_path != "none":
        filter_strings_path = Path(filter_strings_path)
        if not filter_strings_path.exists():
            raise FileNotFoundError(f"Filter strings file not found at {filter_strings_path}")
        filter_strings = pd.read_csv(filter_strings_path)
        # filter the current df to only include strings that are in the filter_strings
        old_len = len(df)
        df = df[df["string"].isin(filter_strings["string"])]
        LOGGER.info(f"Filtered dataset from {old_len} to {len(df)} based on filter strings ({len(filter_strings)}).")

    if shuffle:
        df = df.sample(frac=1, random_state=seed)
    if n is not None:
        if n > len(df):
            LOGGER.warning(
                f"Requested number of rows ({n}) is greater than the number of rows in the dataset ({len(df)})."
            )
        df = df.head(n)

    return df


def create_data_file(data: pd.DataFrame, path: str | Path) -> None:
    """
    Create a data file from a pandas DataFrame.

    Args:
        data: The pandas DataFrame to save.
        path: The path to save the data file.
    """
    path = Path(path)
    if path.exists():
        LOGGER.warning(f"Data file already exists at {path}. Overwriting.")
    if not path.suffix == ".csv":
        LOGGER.warning(f"Data file should be a .csv file. Got {path.suffix} instead. Appending...")
        path = path.with_suffix(".csv")
    data.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
