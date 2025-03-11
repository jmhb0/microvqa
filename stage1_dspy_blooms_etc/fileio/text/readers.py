#!/usr/bin/env python3
"""readers.py in src/microchat/fileio/text."""

import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Union

import pandas as pd
import ujson as json
import yaml
from loguru import logger


from microchat.fileio.text import is_empty_file
from microchat.fileio.text import valid_file_ext


def get_metadata_file(
    input_dir: Union[Path, str], metadata_format: str = ".yaml"
) -> Path:
    """Get metadata from the specified input directory based on the provided format.

    Args:
        input_dir: Path to the input directory containing metadata files.
        metadata_format: The format of the metadata file to be loaded.

    Returns:
        dict: A dictionary containing the metadata loaded from the specified file.
    Raises:
        ValueError: If the expected metadata file is not found or if the file format is invalid.
    """
    input_dir = Path(input_dir)
    metadata_format = f".{metadata_format.lower().replace('.', '')}"
    metadata_suffix = [".json", ".yaml"]
    if metadata_format not in metadata_suffix:
        error_msg = (
            f"Expected metadata format {metadata_suffix}, got {metadata_format}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # get list of metadata files in input directory
    # TODO - fix so that file.stem must match parent dir
    file_list = [elem for elem in list(input_dir.iterdir()) if elem.is_file()]
    file_list = [elem for elem in file_list if elem.suffix in metadata_suffix]
    file_list = [elem for elem in file_list if not is_empty_file(elem)]
    # keep only files that have stem same as parent
    file_list = [elem for elem in file_list if elem.stem == input_dir.stem]
    if not file_list:
        error_msg = f"Expected metadata file ({','.join(metadata_suffix)}) not found in {input_dir}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # if multiple metadata files, select based on metadata_format
    metadata_dict = {elem.suffix.strip("."): elem for elem in file_list}
    metadata_file = metadata_dict.get(metadata_format.strip("."))
    if not metadata_file:
        error_msg = f"Expected metadata file ({metadata_format}) not found."
        logger.warning(error_msg)
        metadata_file = metadata_dict.get(list(metadata_dict.keys())[0])
        logger.warning(f"Using the first metadata file: {metadata_file}")

    return metadata_file


def json_loader(
    filepath: Union[str, Path, os.PathLike], strict: bool = False
) -> Dict[str, Any]:
    """Load JSON file and return its contents as a dictionary.

    Args:
        filepath (Union[str, Path]): The path to the JSON file.

    Returns:
        Dict: The contents of the JSON file as a dictionary.

    Examples:
        >>> json_loader("data.json")
        {'key': 'value'}
    """
    filepath = Path(filepath)
    if not valid_file_ext(filepath, [".json", ".jsonl"]):
        logger.error(f"Invalid file type: {filepath.suffix}")
        raise ValueError(f"Invalid file type: {filepath.suffix}")

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    if is_empty_file(filepath, strict=strict):
        logger.error(f"File is empty: {filepath}")
        raise ValueError(f"File is empty: {filepath}")

    # load json and return as dict
    if filepath.suffix == ".json":
        try:
            with open(filepath) as f:
                data_dict = json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file: {filepath}")
            raise e
    elif filepath.suffix == ".jsonl":
        with open(filepath) as f:
            data_dict = [json.loads(line) for line in f if line.strip()]
    else:
        raise ValueError(f"Invalid file type: {filepath.suffix}")

    if not data_dict:
        raise ValueError(f"File is empty: {filepath}")
    else:
        return data_dict


def save_jsonl(data: list, path: os.PathLike | Path) -> None:
    """Save a list of dictionaries into a JSONL (JSON Lines) file.

    Parameters
    ----------
    data : list
        List of dictionaries to be saved into the JSONL file.
    path : Union[os.PathLike, Path]
        Path to save the JSONL file.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If `data` is not a list.
    """
    if not isinstance(data, list):
        raise TypeError("`data` must be a list of dictionaries.")

    with open(path, "w") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")


def yaml_loader(filepath: Union[str, Path, os.PathLike]) -> Dict:
    """Load YAML file and return its contents as a dictionary.

    Args:
        filepath (Union[str, Path]): The path to the YAML file.

    Returns:
        Dict: The contents of the YAML file as a dictionary.

    Raises:
        ValueError: If file_path is None or empty.
        ValueError: If file_path has an invalid file type.
        FileNotFoundError: If file_path does not exist.

    Examples:
        >>> yaml_loader("config.yaml")
        {'key': 'value'}
    """
    filepath = Path(filepath)
    if not valid_file_ext(filepath, {".yaml", ".yml"}):
        logger.error(f"Invalid file type: {filepath.suffix}")
        raise ValueError(f"Invalid file type: {filepath.suffix}")

    if is_empty_file(filepath):
        logger.error(f"File is empty: {filepath}")
        raise ValueError(f"File is empty: {filepath}")

    # load yaml and return as dict
    with open(filepath) as file:
        data_dict = yaml.safe_load(file)

    if data_dict:
        return data_dict

    logger.error(f"File is empty: {filepath}")
    raise ValueError(f"File is empty: {filepath}")
