#!/usr/bin/env python3
"""finetune_blooms_classifier.py in benchmark/finetune_blooms."""
from pathlib import Path
import os
from typing import Optional

import click
from dotenv import find_dotenv
from dotenv import load_dotenv
import logging

from openai import OpenAI


@click.command()
@click.option("--train-file", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--test-file", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--model-name", default="gpt-4o-mini-2024-07-18", type=str, help="The name of the model to fine-tune.")
@click.option("--suffix", default="blooms", type=str, help="Suffix to append to the model name.")
@click.option("--seed", default=8675309, type=int, help="Random seed for reproducibility.")
@click.option("--upload", is_flag=True, help="Upload files to OpenAI.")
@click.option("--dry-run", is_flag=True, help="Perform a trial run with no changes.")
@click.version_option()
def main(
    train_file: Optional[Path] = None,
    test_file: Optional[Path] = None,
    model_name: Optional[str] = None,
    suffix: str = "blooms",
    seed: int = 8675309,
    upload: bool = False,
    dry_run: bool = False,
) -> None:
    """Fine-tune a model to perform Bloom's Taxonomy classification for multimodal biomedical multiple-choice questions."""
    # create logger
    logging.basicConfig(
        format="{asctime} - {name} - {levelname} - {message}",
        style="{",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    train_file = train_file or Path(__file__).parent.joinpath("blooms_classification_finetuning_train.jsonl")
    test_file = test_file or Path(__file__).parent.joinpath("blooms_classification_finetuning_test.jsonl")
    if not train_file.exists():
        logger.error(f"File not found: {train_file}")
        raise FileNotFoundError(f"File not found: {train_file}")

    if not test_file.exists():
        logger.error(f"File not found: {test_file}")
        raise FileNotFoundError(f"File not found: {test_file}")

    logger.info(f"Train file: {train_file} ({train_file.stat().st_size} examples)")
    logger.info(f"Test file: {test_file} ({test_file.stat().st_size} examples)")

    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OPENAI_API_KEY not found in environment.")
        raise ValueError("OPENAI_API_KEY not found in environment")

    client = OpenAI()

    # set model name
    if not model_name:
        model_name = "gpt-4o-mini-2024-07-18"

    logger.info(f"Model name: {model_name}")

    if dry_run:
        logger.info("Dry run complete, no files uploaded.")
        return

    # upload files to OpenAI
    if upload:
        response = client.files.create(
            file=open(train_file, "rb"), purpose="fine-tune"
        )
        logger.info(f"Uploaded {train_file} to OpenAI: {response}")
        logger.info(f"Train file: {response.filename} (id={response.id})")

        response = client.files.create(
            file=open(test_file, "rb"), purpose="fine-tune"
        )
        logger.info(f"Uploaded {test_file} to OpenAI: {response}")
        logger.info(f"Test file: {response.filename} (id={response.id})")

    # get files
    openai_files = client.files.list()
    file_dict = openai_files.to_dict()
    train_file_id = [
        file["id"] for file in file_dict["data"] if file["filename"] == train_file.name
    ]
    if not train_file_id:
        logger.error(f"File not found: {train_file}")
        raise FileNotFoundError(f"File not found: {train_file}")
    elif len(train_file_id) > 1:
        logger.error(f"Multiple files found: {train_file}")
        raise ValueError(f"Multiple files found: {train_file}")

    test_file_id = [
        file["id"] for file in file_dict["data"] if file["filename"] == test_file.name
    ]
    if not test_file_id:
        logger.error(f"File not found: {test_file}")
        raise FileNotFoundError(f"File not found: {test_file}")
    elif len(test_file_id) > 1:
        logger.error(f"Multiple files found: {test_file}")
        raise ValueError(f"Multiple files found: {test_file}")

    train_file_id = train_file_id[0]
    test_file_id = test_file_id[0]
    if train_file_id == test_file_id:
        logger.error("Train and test files are the same.")
        raise ValueError("Train and test files are the same.")
    elif not train_file_id.startswith("file-") or not test_file_id.startswith("file-"):
        logger.error(f"Expected 'file-' prefix in file id. Got: {train_file_id}, {test_file_id}")
        raise ValueError(f"Expected 'file-' prefix in file id. Got: {train_file_id}, {test_file_id}")

    # fine-tune model
    job = client.fine_tuning.jobs.create(
        training_file=train_file_id,
        validation_file=test_file_id,
        model=model_name,
        seed=seed,
        suffix=suffix,
    )
    logger.info(f"Fine-tuning job (job id: {job.id}) created.")
    logger.info(f"Model: {model_name}")
    logger.info(f"Suffix: {suffix}")
    logger.info(f"Seed: {seed}")


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()
