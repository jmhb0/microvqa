#!/usr/bin/env python3
"""dataset_wrappers.py in src/microchat/custom_datasets.

Adapted from:
https://github.com/sanketx/AL-foundation-models/blob/main/ALFM/src/datasets/dataset_wrappers.py
"""
# sourcery skip: upper-camel-case-classes, no-loop-in-tests, no-conditionals-in-tests
__all__ = [
    "HotPotQAWrapper",
    "SciEvalWrapper",
    "MicroChatWrapper",
    "MicroChatV2Wrapper",
    "Mol_Bio_CellWrapper",
    "MicroBenchWrapper",
    "BloomsWrapper",
    "Blooms_PostBotWrapper",
    "Other_BloomsWrapper",
    "NBME_BloomsWrapper",
    "Organism_ResearchWrapper",
]

import os
from typing import Optional, Union

from dotenv import find_dotenv
from dotenv import load_dotenv


import dspy
from dspy.datasets import HotPotQA
from loguru import logger
from pathlib import Path

from microchat import DATA_ROOT
from microchat.custom_datasets.base_dataset import HFDataset, CSVDataset

load_dotenv(find_dotenv())
RANDOM_SEED = 8675309

load_dotenv(find_dotenv())


class BaseDataWrapper:
    def __init__(self, root: str = None, **kwargs: Optional[dict]):
        self.name: str = self.__class__.__name__.replace("Wrapper", "").lower()
        self.filepath: Path = self._check_filepath(root)
        self.kwargs: dict = kwargs

    def _check_filepath(self, root: Union[str, Path]) -> Path:
        if not root or root is None:
            root = Path(DATA_ROOT)

        filepath = Path(root).joinpath(self.name).with_suffix(".csv")
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            # raise FileNotFoundError(f"File not found: {filepath}")

        return filepath


class HotPotQAWrapper:
    @staticmethod
    def __call__(
        dataset_name: str,
        split: str = "train",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> dspy.datasets.Dataset:
        """Create a HotPotQA dataset object."""

        return HotPotQA(
            train_seed=random_seed,
            train_size=20,
            eval_seed=random_seed + 1,
            dev_size=50,
            test_size=0,
        )


#####
# HuggingFace datasets
# OpenDFM/SciEval
class SciEvalWrapper:

    @staticmethod
    def __call__(
        dataset_name: str = "OpenDFM/SciEval",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> dspy.datasets.Dataset:
        """Create a SciEval dataset object."""

        return HFDataset(
            dataset_name=dataset_name,
            train_seed=random_seed,
            **kwargs,
        )


class BioDEXWrapper:
    @staticmethod
    def __call__(
        dataset_name: str = "BioDEX/BioDEX-Reactions",
        split: str = "train",
        random_seed: Optional[int] = RANDOM_SEED,
        **kwargs: Optional[dict],
    ) -> dspy.datasets.Dataset:
        """Create a HotPotQA dataset object."""

        return HFDataset(
            dataset_name=dataset_name,
            train_seed=random_seed,
            **kwargs,
        )


#####
# Custom CSV datasets
# MicroChat (custom CSV)
class MicroChatWrapper(BaseDataWrapper):

    def __init__(self, root: Optional[str] = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        dataset_name: Optional[str],
        random_seed: Optional[int] = RANDOM_SEED,
        root: Optional[str] = None,
        subset: Optional[list] = None,
        **kwargs: Optional[dict],
    ) -> dspy.datasets.Dataset:
        """Create a MicroChat dataset object."""
        if subset is None:
            subset = ["original_question", "revised_question"]

        root = root or Path(os.getenv("DATA_ROOT"))
        filepath = self.filepath
        if not Path(filepath).exists():
            if root:
                filepath = root.joinpath(filepath)
            else:
                logger.error(f"File not found: {filepath}")
                raise FileNotFoundError(f"File not found: {filepath}")

        return CSVDataset(
            filepath=filepath,
            train_seed=random_seed,
            subset=subset,
            **kwargs,
        )


class MicroChatV2Wrapper(BaseDataWrapper):

    def __init__(self, root: Optional[str] = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        dataset_name: Optional[str],
        random_seed: Optional[int] = RANDOM_SEED,
        root: Optional[str] = None,
        subset: Optional[list] = None,
        **kwargs: Optional[dict],
    ) -> dspy.datasets.Dataset:
        """Create a MicroChat dataset object."""
        if subset is None:
            subset = ["original_question", "revised_question"]
        root = root or Path(os.getenv("DATA_ROOT"))
        filepath = self.filepath
        if not Path(filepath).exists():
            if root:
                filepath = root.joinpath(filepath)
            else:
                logger.error(f"File not found: {filepath}")
                raise FileNotFoundError(f"File not found: {filepath}")

        return CSVDataset(
            filepath=filepath,
            train_seed=random_seed,
            subset=subset,
            **kwargs,
        )


class Mol_Bio_CellWrapper(BaseDataWrapper):

    def __init__(self, root: Optional[str] = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        dataset_name: Optional[str],
        random_seed: Optional[int] = RANDOM_SEED,
        root: Optional[str] = None,
        subset: Optional[list] = None,
        **kwargs: Optional[dict],
    ) -> dspy.datasets.Dataset:
        """Create a MicroChat dataset object."""
        if subset is None:
            subset = ["question_stem", "correct_answer"]
        root = root or Path(os.getenv("DATA_ROOT"))
        filepath = self.filepath
        if not Path(filepath).exists():
            if root:
                filepath = root.joinpath(filepath)
            else:
                logger.error(f"File not found: {filepath}")
                raise FileNotFoundError(f"File not found: {filepath}")

        return CSVDataset(
            filepath=filepath,
            train_seed=random_seed,
            subset=subset,
            **kwargs,
        )


class MicroBenchWrapper(BaseDataWrapper):

    def __init__(self, root: Optional[str] = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    # @staticmethod
    def __call__(
        self,
        dataset_name: Optional[str],
        random_seed: Optional[int] = RANDOM_SEED,
        root: Optional[str] = None,
        subset: Optional[list] = None,
        **kwargs: Optional[dict],
    ) -> dspy.datasets.Dataset:
        """Create a MicroChat dataset object."""
        if subset is None:
            subset = ["question_stem", "correct_answer"]
        root = root or Path(os.getenv("DATA_ROOT"))
        filepath = self.filepath
        if not Path(filepath).exists():
            if root:
                filepath = root.joinpath(filepath)
            else:
                logger.error(f"File not found: {filepath}")
                raise FileNotFoundError(f"File not found: {filepath}")

        return CSVDataset(
            filepath=filepath,
            train_seed=random_seed,
            subset=subset,
            **kwargs,
        )


class BloomsWrapper(BaseDataWrapper):

    def __init__(self, root: Optional[str] = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    def __call__(
        self,
        dataset_name: Optional[str],
        random_seed: Optional[int] = RANDOM_SEED,
        root: Optional[str] = None,
        subset: Optional[list] = None,
        **kwargs: Optional[dict],
    ) -> dspy.datasets.Dataset:
        """Create a MicroChat dataset object."""
        if subset is None:
            subset = ["question_stem", "correct_answer"]
        root = root or Path(os.getenv("DATA_ROOT"))
        filepath = self.filepath
        if not Path(filepath).exists():
            if root:
                filepath = root.joinpath(filepath)
            else:
                logger.error(f"File not found: {filepath}")
                raise FileNotFoundError(f"File not found: {filepath}")

        return CSVDataset(
            filepath=filepath,
            train_seed=random_seed,
            subset=subset,
            **kwargs,
        )


class Blooms_PostBotWrapper(BaseDataWrapper):

    def __init__(self, root: Optional[str] = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    def __call__(
        self,
        dataset_name: Optional[str],
        random_seed: Optional[int] = RANDOM_SEED,
        root: Optional[str] = None,
        subset: Optional[list] = None,
        **kwargs: Optional[dict],
    ) -> dspy.datasets.Dataset:
        """Create a MicroChat dataset object."""
        if subset is None:
            subset = ["question_stem", "correct_answer"]
        root = root or Path(os.getenv("DATA_ROOT"))
        filepath = self.filepath
        if not Path(filepath).exists():
            if root:
                filepath = root.joinpath(filepath)
            else:
                logger.error(f"File not found: {filepath}")
                raise FileNotFoundError(f"File not found: {filepath}")

        return CSVDataset(
            filepath=filepath,
            train_seed=random_seed,
            subset=subset,
            **kwargs,
        )


class Other_BloomsWrapper(BaseDataWrapper):

    def __init__(self, root: Optional[str] = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    def __call__(
        self,
        dataset_name: Optional[str],
        random_seed: Optional[int] = RANDOM_SEED,
        root: Optional[str] = None,
        subset: Optional[list] = None,
        **kwargs: Optional[dict],
    ) -> dspy.datasets.Dataset:
        """Create a MicroChat dataset object."""
        if subset is None:
            subset = ["question_stem", "correct_answer"]
        root = root or Path(os.getenv("DATA_ROOT"))
        filepath = self.filepath
        if not Path(filepath).exists():
            if root:
                filepath = root.joinpath(filepath)
            else:
                logger.error(f"File not found: {filepath}")
                raise FileNotFoundError(f"File not found: {filepath}")

        return CSVDataset(
            filepath=filepath,
            train_seed=random_seed,
            subset=subset,
            **kwargs,
        )


class NBME_BloomsWrapper(BaseDataWrapper):

    def __init__(self, root: Optional[str] = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    def __call__(
        self,
        dataset_name: Optional[str],
        random_seed: Optional[int] = RANDOM_SEED,
        root: Optional[str] = None,
        subset: Optional[list] = None,
        **kwargs: Optional[dict],
    ) -> dspy.datasets.Dataset:
        """Create a MicroChat dataset object."""
        if subset is None:
            subset = ["question_stem", "correct_answer"]
        root = root or Path(os.getenv("DATA_ROOT"))
        filepath = self.filepath
        if not Path(filepath).exists():
            if root:
                filepath = root.joinpath(filepath)
            else:
                logger.error(f"File not found: {filepath}")
                raise FileNotFoundError(f"File not found: {filepath}")

        return CSVDataset(
            filepath=filepath,
            train_seed=random_seed,
            subset=subset,
            **kwargs,
        )


class Organism_ResearchWrapper(BaseDataWrapper):

    def __init__(self, root: Optional[str] = None, **kwargs: Optional[dict]):
        super().__init__(root, **kwargs)

    def __call__(
        self,
        dataset_name: Optional[str],
        random_seed: Optional[int] = RANDOM_SEED,
        root: Optional[str] = None,
        subset: Optional[list] = None,
        **kwargs: Optional[dict],
    ) -> dspy.datasets.Dataset:
        """Create a MicroChat dataset object."""
        if subset is None:
            subset = ["question_stem", "correct_answer"]
        root = root or Path(os.getenv("DATA_ROOT"))
        filepath = self.filepath
        if not Path(filepath).exists():
            if root:
                filepath = root.joinpath(filepath)
            else:
                logger.error(f"File not found: {filepath}")
                raise FileNotFoundError(f"File not found: {filepath}")

        return CSVDataset(
            filepath=filepath,
            train_seed=random_seed,
            subset=subset,
            **kwargs,
        )
