#!/usr/bin/env python3
"""base_dataset in src/microchat/custom_datasets."""
import os
from typing import List
from typing import Optional, Dict, Any, Union
from pathlib import Path

import pandas as pd
from loguru import logger


import random

from datasets import load_dataset

from dspy.datasets.dataset import Dataset
import dspy

from tqdm import tqdm

from microchat.fileio.dataframe.readers import df_loader


class HFDataset(Dataset):
    dataset_name: str

    def __init__(self, *args, dataset_name: str, **kwargs: Optional[dict]):
        # initialize the base class
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.kwargs: dict = kwargs

        # read splits
        hf_dataset: dict = {}
        # find if split in dataset
        available_splits = list(
            load_dataset(self.dataset_name, trust_remote_code=True).keys()
        )
        for split in available_splits:
            try:
                hf_dataset[split] = self.convert_hf_to_dspy(
                    dataset_name=self.dataset_name, split=split
                )
            except Exception as e:
                logger.error(f"Error loading dataset {self.dataset_name} split {split}")
                logger.error(e)

        # create unofficial dev and validation if not present
        if (
            hf_dataset.get("train")
            and not hf_dataset.get("validation")
            and not hf_dataset.get("dev")
        ):
            logger.info("Creating unofficial dev splits.")
            official_train = hf_dataset.pop("train")
            rng = random.Random(0)
            rng.shuffle(official_train)
            num_train = len(official_train) * 75 // 100
            num_dev = len(official_train) - num_train  # TODO: check min 1
            hf_dataset["train"] = official_train[:num_train]
            hf_dataset["dev"] = official_train[num_train : num_train + num_dev]
        elif (
            not hf_dataset.get("train")
            and hf_dataset.get("validation")
            or hf_dataset.get("dev")
        ):
            logger.info("Creating unofficial train split from official dev.")
            official_dev = hf_dataset.pop("validation") or hf_dataset.pop("dev")
            rng = random.Random(0)
            rng.shuffle(official_dev)
            hf_dataset["train"] = official_dev[: len(official_dev) * 75 // 100]
            hf_dataset["dev"] = official_dev[len(official_dev) * 75 // 100 :]
        else:
            logger.info("Creating unofficial train split from official dev.")
            official_dev = hf_dataset.pop("validation") or hf_dataset.pop("dev")
            rng = random.Random(0)
            rng.shuffle(official_dev)
            hf_dataset["train"] = official_dev[: len(official_dev) * 75 // 100]
            hf_dataset["dev"] = official_dev[len(official_dev) * 75 // 100 :]

        # assign splits
        self._train = hf_dataset.get("train", [])
        self._dev = hf_dataset.get("validation", []) or hf_dataset.get(
            "dev", []
        )  # dspy.Dataset uses "_dev" for validation
        self._test = hf_dataset.get("test", [])

    @staticmethod
    def convert_hf_to_dspy(
        dataset_name: str,
        split: str,
        question_key: str = "question",
        answer_key: str = "answer",
        keep_details: bool = True,
        trust_remote_code: bool = True,
    ) -> List[Dict[str, Any]]:
        """Convert a HuggingFace dataset to a DSPy dataset."""
        # load dataset
        hf_dataset = load_dataset(
            path=dataset_name, split=split, trust_remote_code=trust_remote_code
        )
        # check keys "question" and "answer" are present
        if (
            question_key not in hf_dataset.features
            or answer_key not in hf_dataset.features
        ):
            raise ValueError(
                f"Dataset {dataset_name} does not have 'question' and 'answer' fields."
            )

        # initialize the dspy dataset
        dataset: List[Dict[str, Any]] = []
        keys: List[str] = [question_key, answer_key]

        # iterate over HuggingFace dataset and convert to DSPy dataset
        for raw_example in tqdm(
            hf_dataset, desc=f"Converting {dataset_name} to DSPy format"
        ):
            # convert example to DSPy format
            if keep_details:
                # extend keys with additional fields, keep set of keys unique
                keys = list(set(keys + list(raw_example.keys())))

            example = {k: raw_example[k] for k in keys}
            dataset.append(example)

        return dataset


class CSVDataset(Dataset):
    filepath: Union[str, Path, os.PathLike]

    def __init__(
        self,
        filepath: str,
        subset: Optional[list] = None,
        question_key: Optional[str] = "question",
        answer_key: Optional[str] = "answer",
        **kwargs: Optional[dict],
    ):
        # initialize the base class
        super().__init__(**kwargs)
        self.filepath = filepath
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File {filepath} not found.")

        self.dataset_name = Path(filepath).stem
        self.kwargs: dict = kwargs
        random_seed = kwargs.get("random_seed", 8675309)
        # question_key = kwargs.get("question_key", "question")
        # answer_key = kwargs.get("answer_key", "answer")

        # read data
        df = df_loader(filepath)

        # count unique for blooms_question_category
        if "blooms_level" in df.columns:
            logger.info(f"Unique blooms_level: {df['blooms_level'].nunique()}")

            # # drop any with blooms_source of "gpt-4o-mini-finetune"
            # df = df[df["blooms_source"] != "gpt-4o-mini-finetune"]

            # sample min_samples from each blooms_level
            num_samples = 15  # df["blooms_level"].value_counts().min()
            min_samples = df["blooms_level"].value_counts().min()
            logger.info(f"Min samples: {min_samples}")
            df = (
                df.groupby("blooms_level")
                .apply(
                    lambda x: x.sample(
                        min(num_samples, len(x)),
                        replace=False,
                        random_state=random_seed,
                    )
                )
                .reset_index(drop=True)
            )

        # fillna with empty string for col blooms_rationale
        if "blooms_rationale" in df.columns and df["blooms_rationale"].isnull().any():
            df["blooms_rationale"] = df["blooms_rationale"].fillna("")

        # fillna correct_answer if all nan
        if "correct_answer" in df.columns and df["correct_answer"].isnull().any():
            df["correct_answer"] = df["correct_answer"].fillna("")

        # HACK create new column revised question-answer
        if "microchat" in Path(filepath).stem:
            # strip ending newline
            df["description"] = (
                df["question"].copy().apply(lambda x: x.split(r"Question:")[0].strip())
            )
            df["original_answer"] = "Answer:\n" + df[
                "question_and_answer"
            ].copy().apply(lambda x: x.split(r"Answer:")[1].strip())
            df["original_question_answer"] = (
                df["question"] + "\n\nAnswer:\n```" + df["answer_correct"] + "```"
            )
            df["revised_question_answer"] = (
                "Revised Question:\n```"
                + df["revised_question"]
                + "```\n\nRevised Answer:\n```"
                + df["answer_correct"]
                + "```"
            )
            df["revised_question_answer_mc"] = (
                "Revised Question:\n```"
                + df["revised_question"]
                + "\n\nRevised Answer:\n```"
                + df["answer_correct"]
                + "\n\nOptions:\n```"
                + df["multiple_choice"]
            )
        elif Path(filepath).stem in {
            "mol_bio_cell",
            "microbench",
            "blooms",
            "other_blooms",
            "nbme_blooms",
        }:
            df["question_answer"] = (
                "Question:\n```"
                + df["question_stem"]
                + "```\n\nAnswer:\n```"
                + df["correct_answer"]
                + "```"
            )
        else:
            logger.error(f"File {filepath} not found.")

        # strip ending newline or whitespace
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        if col_set_diff := {question_key, answer_key} - set(df.columns):
            raise ValueError(
                f"DataFrame {filepath} does not have columns {col_set_diff}."
            )

        # create splits based on "split" column
        if "split" not in df.columns:
            df["split"] = "train"

        # create output dict
        df_dataset: dict = {}

        # find if split in dataset
        available_splits = df["split"].unique()
        for split in available_splits:
            try:
                df_dataset[split] = self.convert_df_to_dspy(
                    df=df,
                    subset=subset,
                    question_key=question_key,
                    answer_key=answer_key,
                )
            except Exception as e:
                logger.error(f"Error loading dataset {self.dataset_name} split {split}")
                logger.error(e)

        # create unofficial dev and validation if not present
        if (
            df_dataset.get("train")
            and not df_dataset.get("validation")
            and not df_dataset.get("dev")
        ):
            logger.info("Creating unofficial dev splits.")
            official_train = df_dataset.pop("train")
            rng = random.Random(0)
            rng.shuffle(official_train)
            df_dataset["train"] = official_train[: len(official_train) * 75 // 100]
            df_dataset["dev"] = official_train[len(official_train) * 75 // 100 :]
        elif (
            not df_dataset.get("train")
            and df_dataset.get("validation")
            or df_dataset.get("dev")
        ):
            logger.info("Creating unofficial train split from official dev.")
            official_dev = df_dataset.pop("validation") or df_dataset.pop("dev")
            rng = random.Random(0)
            rng.shuffle(official_dev)
            df_dataset["train"] = official_dev[: len(official_dev) * 75 // 100]
            df_dataset["dev"] = official_dev[len(official_dev) * 75 // 100 :]

        # assign splits
        self._train = df_dataset.get("train", [])
        self._dev = df_dataset.get("validation", []) or df_dataset.get(
            "dev", []
        )  # dspy.Dataset uses "_dev" for validation
        self._test = df_dataset.get("test", [])

    @staticmethod
    def convert_df_to_dspy(
        df: pd.DataFrame,
        question_key: str = "question",
        answer_key: str = "answer",
        keep_details: bool = False,
        subset: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert a HuggingFace dataset to a DSPy dataset."""
        # check keys "question" and "answer" are present
        if question_key not in df.columns or answer_key not in df.columns:
            raise ValueError("Dataset does not have 'question' and 'answer' fields.")

        # filter to remove rows with missing values
        if subset:
            # get num missing in subset
            num_missing = df.loc[:, subset].isnull().sum(axis=1)
            df = df.dropna(subset=subset)
            if num_missing.sum() > 0:
                logger.info(f"Removed {num_missing.sum()} rows with missing values.")
                logger.info(f"Remaining rows: {len(df)}")

        # initialize the dspy dataset
        dataset: List[Dict[str, Any]] = []
        keys: List[str] = list(
            {question_key, answer_key, "key_image", "key_question", *subset}
        )

        # iterate over HuggingFace dataset and convert to DSPy dataset
        for idx, raw_example in tqdm(
            df.iterrows(), desc="Converting df to DSPy format"
        ):
            # convert example to DSPy format
            if keep_details:
                # extend keys with additional fields, keep set of keys unique
                keys = list(set(keys + list(raw_example.keys())))

            # convert pandas Series to dict
            raw_example = raw_example.to_dict()
            example = {}
            for k in keys:
                if k == question_key:
                    # remove key from example and assign to question
                    question = raw_example.pop(k)
                    example["question"] = question.strip()
                elif k == answer_key:
                    answer = raw_example.pop(k)
                    example["answer"] = answer.strip()
                elif k in {
                    "key_image",
                    "key_question",
                }:
                    example[k] = raw_example[k]
                elif k in {"organism", "specimen", "research_subject"}:
                    example[k] = raw_example[k].strip()
                else:
                    continue
                    # example[k] = example[k].strip()

            dataset.append(example)

        return dataset
