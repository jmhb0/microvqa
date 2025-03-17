#!/usr/bin/env python3
"""writers.py in src/microchat/fileio/text."""

import pprint
from pathlib import Path
from pprint import pprint
from typing import Dict
from typing import Union


import ujson as json
import yaml
from loguru import logger

from microchat.fileio.text import is_none_or_empty
from microchat.fileio.text import make_dir


def yaml_writer(data: dict, file_path: Union[str, Path]) -> None:
    """Write data to a YAML file.

    Args:
        data (Dict): The data to be written to the YAML file.
        file_path (Union[str, Path]): The path to the YAML file.

    Returns:
        None

    Raises:
        ValueError: If data is None or empty.
        ValueError: If file_path is None or empty.

    Examples:
        >>> data = {'key': 'value'}
        >>> yaml_writer(data, "output.yaml")
    """
    if is_none_or_empty(data):
        raise ValueError("data cannot be None or empty")

    if is_none_or_empty(file_path):
        raise ValueError("file_path cannot be None or empty")

    # create directory if it does not exist
    file_path = make_dir(file_path)

    # write data to yaml file
    with open(file_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def is_json_serializable(data: dict) -> bool:
    """Check if the provided dictionary is serializable to JSON.

    Args:
        data: A dictionary containing the data to be checked for JSON serialization.

    Returns:
        bool: True if the data is JSON serializable, False otherwise.
    """
    serializable = True
    try:
        json.dumps(data)
    except Exception as e:
        # print error type without <class '...'>
        error_type = str(type(e)).split("'")[1]
        logger.error(f"{error_type}: Error serializing json data: {str(e)}")
        pprint(data)
        serializable = False

    return serializable


def json_writer(data: dict, filepath: Union[str, Path], **kwargs) -> None:
    """Write data to a JSON file.

    Args:
        data (Dict): The data to be written to the JSON file.
        file_path (Union[str, Path]): The path to the JSON file.
        indent (int): The number of spaces to indent the JSON file. Default is None.

    Returns:
        None

    Raises:
        ValueError: If data is None or empty.
        ValueError: If file_path is None or empty.

    Examples:
        >>> data = {'key': 'value'}
        >>> json_writer(data, "output.json")
    """
    if is_none_or_empty(data):
        raise ValueError("data cannot be None or empty")

    if is_none_or_empty(filepath):
        raise ValueError("file_path cannot be None or empty")

    if not is_json_serializable(data):
        raise ValueError(f"Data for file {filepath.name} is not JSON serializable")

    if kwargs.get("sa_json", False):
        # validate using SADict
        orig_data = data.copy()
        data = SADict(**data)
        data = data.model_dump(exclude_unset=True)

    # create directory if it does not exist
    filepath = make_dir(filepath)

    # write data to json file
    with open(filepath, "w") as f:
        if kwargs.get("indent", False):
            json.dump(data, f, indent=kwargs["indent"])
        else:
            json.dump(data, f)

    if filepath.stat().st_size == 0:
        logger.error(f"Error writing JSON file: {filepath} (size: 0)")
        # save backup file
        backup_file = filepath.with_suffix(".bak")
        logger.warning(f"Saving backup file: {backup_file}")
        # write original data to backup file
        # TODO: temp
        with open(backup_file, "w") as f:
            json.dump(orig_data, f)
        raise ValueError(f"Error writing JSON file: {filepath} (size: 0)")
