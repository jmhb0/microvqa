#!/usr/bin/env python3
"""dataset_factory.py in src/microchat/custom_datasets.

Adapted from:
https://github.com/sanketx/AL-foundation-models/blob/main/ALFM/src/datasets/factory.py
"""
from pathlib import Path
from typing import Optional
import dspy
from microchat.custom_datasets.dataset_registry import DatasetType


def create_dataset(
    dataset_name: str,
    subset: Optional[list] = None,
    question_key: Optional[str] = None,
    answer_key: Optional[str] = None,
) -> dspy.datasets.Dataset:
    """Create a dataset given its corresponding DatasetType enum value.

    Args:
        dataset_name (str): An enum value representing the dataset to be created.

    Returns:
        dspy.datasets.Dataset: The dataset object.

    """
    if Path(dataset_name).suffix == ".csv":
        dataset_name = Path(dataset_name).stem

    if dataset_name not in DatasetType.__members__:
        raise ValueError(f"Dataset {dataset_name} not found in DatasetType enum.")

    dataset_type = DatasetType[dataset_name]
    return dataset_type.value(
        dataset_name=dataset_name,
        subset=subset,
        question_key=question_key,
        answer_key=answer_key,
    )
