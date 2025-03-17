#!/usr/bin/env python3
"""run_dspy.py in src/microchat."""
import os
import pprint
import re
from pathlib import Path

from typing import Optional
import click
import pandas as pd

from dotenv import find_dotenv
from dotenv import load_dotenv

from loguru import logger


import dspy
from dspy.evaluate.evaluate import Evaluate
from tqdm import tqdm

from microchat import PROJECT_ROOT, DATA_ROOT, MODULE_ROOT
from microchat.custom_datasets.dataset_factory import create_dataset
from microchat.fileio.dataframe.readers import df_loader
from microchat.fileio.text.readers import yaml_loader
from microchat.fileio.text.writers import yaml_writer
from microchat.metrics.mcq_metric import (
    validate_blooms,
    validate_nbme,
)
from microchat.models.dspy_modules import CoTSelfCorrectRAG, CoTRAG, context
from microchat.models.model_factory import create_model
from microchat.mc_questions.mcq import MCQ, Blooms
from microchat.teleprompters.teleprompter_factory import create_optimizer
from microchat.utils.process_model_history import history_to_jsonl
from microchat.utils.process_text import process_blooms

# try:
#     import datasets
#
#     if datasets.__version__ != "3.0.1":
#         raise ImportError(
#             f"Dataset may not be compatible with DSPy. Please install datasets==3.0.1."
#         )
# except ImportError as e:
#     logger.error("Please install datasets==3.0.1.")
#     logger.error(e)
#     raise e


@click.command()
@click.argument(
    "input-file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--model", type=click.STRING, default="o1-mini")  # gpt-4o-mini")
@click.option("--teacher-model", type=click.STRING, default="o1-mini")
@click.option(
    "--output-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option("--random-seed", type=click.INT, default=8675309)
@click.option("--dry-run", is_flag=True, help="Perform a trial run with no changes.")
@click.version_option()
def main(
    input_file: Path,
    model: Optional[str] = "o1-mini",  # "gpt-4o-mini",
    teacher_model: Optional[str] = "o1-mini",
    output_dir: Optional[Path] = None,
    random_seed: int = 8675309,
    dry_run: bool = False,
) -> None:
    """Docstring."""
    if not output_dir:
        output_dir = Path(input_file).parent

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir.joinpath(f"{input_file.stem}_llm-qc{input_file.suffix}")

    project_dir = Path(__file__).parents[2]
    logger.add(
        project_dir.joinpath("logs", f"{Path(__file__).stem}.log"),
        rotation="10 MB",
        level="INFO",
    )

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Model: {model}")
    logger.info(f"Teacher model: {teacher_model}") if teacher_model else None
    logger.info(f"Random seed: {random_seed}")

    if "mini" not in model:
        logger.warning(f"Model {model} may require increased costs.")
        click.confirm(
            f"Are you sure you want to continue with {model} and increased costs?",
            abort=True,
        )

    if "mini" not in teacher_model:
        logger.warning(f"Teacher model {teacher_model} may require increased costs.")
        click.confirm(
            f"Are you sure you want to continue with {teacher_model} and increased costs?",
            abort=True,
        )

    if dry_run:
        logger.info("Dry run: no changes will be made.")
        return

    # instantiate model LLM/VLM model
    model = create_model(model)

    # configure DSPy settings
    dspy.settings.configure(lm=model.lm, rm=None)

    # create module
    task = "nbme"
    retrieve_k = 5
    module = CoTRAG(context=task, num_passages=retrieve_k)
    module.name = module.__class__.__name__

    # set trained module
    model_filepath = PROJECT_ROOT.joinpath("models", "dspy_compiled", task).joinpath(
        f"{model.model_name}_{module.name}_{module.signature_name}.json",
    )
    if not model_filepath.exists():
        logger.error(f"Error loading trained model from {model_filepath}")
        raise FileNotFoundError(f"Error loading trained model from {model_filepath}")
    else:
        logger.info(f"Loading trained model {model_filepath.name}")
        module.load(model_filepath)

    # read df
    if input_file.suffix == ".csv":
        df = pd.read_csv(input_file, index_col=False)
    elif input_file.suffix == ".xlsx":
        df = pd.read_excel(input_file, index_col=False)
    else:
        logger.error(f"File type not supported: {input_file.suffix}")
        return
    logger.info(f"Processing {len(df)} examples")

    # ##### Microchat inference #####
    if not model_filepath.exists():
        logger.error(f"Error saving compiled RAG to {model_filepath}")
        raise FileNotFoundError(f"Error saving compiled RAG to {model_filepath}")

    logger.info(f"Loading compiled RAG from {model_filepath}")
    trained_module = CoTRAG(context=task, num_passages=retrieve_k)
    trained_module.load(model_filepath)
    trained_module.name = trained_module.__class__.__name__

    # read df
    df = df_loader(Path(DATA_ROOT).joinpath("20250206_questions.csv"))
    # drop nan
    df = df.dropna(subset=["question", "answer"])
    logger.info(f"Processing {len(df)} examples")

    df["original_answer"] = df["answer"].copy().apply(lambda x: x.strip())

    df["original_question_answer"] = df["description_question_answer"]
    logger.info(f"Example input:\t{df['original_question_answer'].iloc[0]}")

    # create examples
    nbme_context = context["nbme"]
    nbme_formatted = trained_module._format_context(nbme_context)
    examples = []
    start_idx = 0
    import random

    for idx, row in df.iterrows():
        if row["key_question"] < start_idx:
            continue

        # take retrieve_k random contexts index
        random.shuffle(nbme_formatted)

        # skip if empty
        if not row["original_question_answer"]:
            logger.warning(f"Empty question answer at index {idx}")
            continue
        elif pd.isna(row["original_question_answer"]) or pd.isnull(
            row["original_question_answer"]
        ):
            logger.warning(f"Empty question answer at index {idx}")
            continue

        # create example
        examples.append(
            dspy.Example(
                context=nbme_formatted[:retrieve_k],
                question=row["original_question_answer"],
                key_question=row["key_question"],
                key_image=row["key_image"],
            )
        )

    # perform inference and save to df
    output_list = []
    for idx, example in tqdm(enumerate(examples), total=len(examples)):
        try:
            # perform inference to get stage 1 output (revised question answer)
            output = trained_module(example)

            if "answer" not in output.answer.lower():
                logger.info(f"Error in output: {output}")

            try:
              mcq = MCQ(example=example, prediction=output)
            except Exception:
                logger.error(f"Error creating MCQ")

            # append to output list
            output_list.append(
                {
                    "key_image": example.key_image,
                    "key_question": example.key_question,
                    "description_question_answer": example.question,
                    "revised_question_answer": output.answer,
                }
            )

            # temp save
            if idx % 100 == 0:
                logger.info(f"Processed {idx} examples")
                df = pd.DataFrame(output_list)
                df.to_csv(
                    output_dir.joinpath(f"{model_filepath.stem}_output.csv"),
                    index=False,
                )

        except Exception:
            # convert to df and save
            logger.error(f"Error processing example at index {idx}")
            df = pd.DataFrame(output_list)
            df.to_csv(
                output_dir.joinpath(f"{model_filepath.stem}_output.csv"), index=False
            )

    df.to_csv(
        output_dir.joinpath(f"{model_filepath.stem}_output_final.csv"), index=False
    )
    return

    #####


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()