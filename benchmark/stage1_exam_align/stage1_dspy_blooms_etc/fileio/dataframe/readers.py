#!/usr/bin/env python3
"""readers.py in src/microchat/fileio/dataframe."""

import os
from pathlib import Path
from typing import Union, Optional

import pandas as pd
from loguru import logger

try:
    from pyarrow import feather

    use_feather = False
except ModuleNotFoundError:
    logger.warning("pyarrow not found. Using pandas instead.")
    use_feather = False

from microchat.fileio.text import is_empty_file
from microchat.fileio.text import valid_file_ext


def df_loader(
    filepath: Union[str, Path, os.PathLike], index_col: Optional[int] = None
) -> pd.DataFrame:
    """Load CSV file and return its contents as a DataFrame.

    Args:
        filepath (Union[str, Path]): The path to the CSV file.

    Returns:
        pd.DataFrame: The contents of the CSV, Parquet, or Feather file as a DataFrame.

    Raises:
        ValueError: If file_path is None or empty.
        ValueError: If file_path has an invalid file type.
        FileNotFoundError: If file_path does not exist.

    Examples:
        >>> df_loader("data.csv")
        key
        0  value
    """
    filepath = Path(filepath)
    if not valid_file_ext(filepath, {".csv", ".parquet", ".feather"}):
        logger.error(f"Invalid file type: {filepath.suffix}")
        raise ValueError(f"Invalid file type: {filepath.suffix}")

    if is_empty_file(filepath):
        logger.error(f"File is empty: {filepath}")
        raise ValueError(f"File is empty: {filepath}")

    if filepath.suffix == ".csv":
        return pd.read_csv(filepath, index_col=index_col)
    elif filepath.suffix == ".parquet":
        return pd.read_parquet(filepath)
    elif filepath.suffix == ".feather" and use_feather:
        return feather.read_feather(filepath.as_posix())
    else:
        logger.error(f"Unsupported file format: {filepath.suffix}")
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
