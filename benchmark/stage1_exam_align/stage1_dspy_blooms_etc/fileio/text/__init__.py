#!/usr/bin/env python3
"""__init__.py in src/microchat/fileio/text."""

import re
from pathlib import Path
from typing import Dict
from typing import Union

from loguru import logger

# from microchat.conf.compiled_regex import RE_FILENAME


# TODO: consider support for BioFormats ".ome.tiff", ".ome.tif"
VALID_EXTENSIONS = [".jpeg", ".jpg", ".png", ".gif", ".bmp", ".tiff", ".tif"]


def is_none_or_empty(data: Union[Dict, str, Path, list]) -> bool:
    """Check if data is None or empty."""
    return data is None or not data


def is_empty_file(file_path: Union[str, Path], strict: bool = False) -> bool:
    """Check if a file is empty."""
    if is_none_or_empty(file_path):
        logger.error(f"Expected Union[str, Path]. Actual: {type(file_path)}")
        if strict:
            raise ValueError("file_path cannot be None or empty")
        else:
            return True

    file_path = Path(file_path)
    if not file_path.is_file() or not file_path.exists():
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        if strict:
            raise FileNotFoundError(error_msg)
        else:
            return True

    return file_path.stat().st_size == 0


def valid_file_ext(
    file_path: Union[str, Path], file_type: Union[str, set, list]
) -> bool:
    """Check if a file is of a specific type."""
    file_type = {file_type} if isinstance(file_type, str) else set(file_type)
    return Path(file_path).suffix in file_type


def is_valid_filename(v: str) -> bool:
    """Check if filename is valid."""
    v = Path(v)
    if not valid_file_ext(v, VALID_EXTENSIONS):
        return False

    match = RE_FILENAME.match(v.name)
    return bool(match)


def make_dir(file_path: Union[str, Path]) -> Path:
    """Create directory if it does not exist."""
    # create directory if it does not exist
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path if file_path.parent.is_dir() else None
