#!/usr/bin/env python3
"""convert_hf_to_csv.py in src/microchat."""
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from dotenv import find_dotenv
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from microchat import DATA_ROOT, PROJECT_ROOT

import datasets  # HF datasets


def extract_questions(row, idx: Optional[int] = None):
    """Extract questions from row for microbench dataset.

    Will need to be customized for other HF datasets"""

    all_question = []
    for q_type in row["questions"].keys():
        elem = row["questions"][q_type]
        if elem is None:
            continue

        question_id = elem["id"]  # 'question_id' is col 'id' for ubench (UUID)
        question_stem = elem["question"]  # 'question_stem' is col 'question' for ubench
        correct_answer = elem["answer"]  # 'correct_answer' is col 'answer' for ubench
        multiple_choice = elem["options"]  # MC col is "options" for ubench
        all_question.append(
            {
                "source": row["source"],  # dataset name
                "chapter": row["chapter"],  # dummy
                "idx": idx,  # index of row in dataset
                "question_id": question_id,
                "question_stem": question_stem,
                "correct_answer": correct_answer,
                "multiple_choice": multiple_choice,
                "question_type": q_type,
            }
        )

    return all_question


@click.command()
@click.argument("dataset", type=click.STRING)
@click.option(
    "--output-dir", type=click.Path(file_okay=False, exists=False, path_type=Path)
)
@click.option("--split", type=click.STRING, default="test")
@click.option("--dry-run", is_flag=True, help="Perform a trial run with no changes.")
@click.version_option()
def main(
    dataset: str,
    output_dir: Optional[Path] = None,
    split: str = "test",
    dry_run: bool = False,
) -> None:
    """Extract questions from HF dataset and save to CSV."""
    dataset_name = dataset.split("/")[-1].lower()
    output_dir = output_dir or Path(DATA_ROOT).joinpath(dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir.joinpath(f"{dataset.replace('/', '-')}_{split}.csv")

    logger.add(
        PROJECT_ROOT.joinpath("logs", f"{Path(__file__).stem}.log"),
        rotation="10 MB",
        level="INFO",
    )

    logger.info(f"Dataset: {dataset}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output file: {output_file}")

    if output_file.exists():
        logger.warning(f"Output file already exists: {output_file}")
        click.confirm("Do you want to overwrite?", abort=True)

    # load hf dataset
    try:
        hf_dataset = datasets.load_dataset(dataset, split=split)
        logger.info(f"Dataset loaded: {hf_dataset}")
    except Exception as e:
        logger.error(f"Error loading dataset: {dataset}")
        logger.error(e)
        raise

    # convert to df
    df = hf_dataset.to_pandas()
    df["source"] = dataset_name  # dataset name
    df["chapter"] = None  # dummy

    #
    if dataset.split("/")[-1].lower() != "ubench":
        logger.warning(
            f"Please customize 'extract_questions' function for dataset: {dataset}"
        )

    # for each example, save question_stem, correct_answer to CSV
    output_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        output_list.extend(extract_questions(row, idx=idx))
        if dry_run and idx > 10:
            break

    # combine all data
    output_df = pd.DataFrame(output_list)
    logger.info(f"Output df: {output_df}")
    if dry_run:
        logger.info("Dry run: no changes will be made.")
    else:
        output_df.to_csv(output_file, index=False)
        logger.info(f"Questions saved to {output_file}")


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()
