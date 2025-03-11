#!/usr/bin/env python3
"""create_wordcloud.py in src/microchat."""
import os
from pathlib import Path

from typing import Optional, Dict, Any
import click
import pandas as pd

from dotenv import find_dotenv
from dotenv import load_dotenv

from loguru import logger

import textstat
import numpy as np


@click.command()
@click.argument("input-file", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--output-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--column",
    default="revised_question_postbot",
    type=str,
    help="Column to use for wordcloud.",
)
@click.option(
    "--filter_col",
    default=None,
    type=dict,
    help="Filter to apply to the dataframe. {column_name: value}",
)
@click.option("--dry-run", is_flag=True, help="Perform a trial run with no changes.")
@click.version_option()
def main(
    input_file: Path,
    output_dir: Optional[Path] = None,
    column: str = "revised_question_postbot",
    filter_col: Optional[Dict[Any, Any]] = None,
    lang: str = "en",
    metric: str = "flesch_kincaid_grade",
    dry_run: bool = False,
) -> None:
    """Docstring."""
    data_root = Path(os.getenv("DATA_ROOT", default=""))
    if not input_file.exists() and data_root.joinpath(input_file).exists():
        input_file = data_root.joinpath(input_file)
    else:
        logger.error(f"Input file {input_file} not found.")
        return

    if not output_dir:
        output_dir = Path(input_file).parent

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir.joinpath(f"{input_file.stem}_difficulty.csv")

    project_dir = Path(__file__).parents[2]
    logger.add(
        project_dir.joinpath("logs", f"{Path(__file__).stem}.log"),
        rotation="10 MB",
        level="INFO",
    )

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Column: {column}")
    logger.info(f"Metric: {metric}")

    if dry_run:
        logger.info("Dry run: no changes will be made.")
        return

    # Read the input file
    df = pd.read_csv(input_file)
    #
    # # temp: create col `multiple_choice` from `choices_2` with A) B) C) D) ... format
    # df["choices_2_formatted"] = df["choices_2"].apply(
    #     lambda x: "".join([f"{chr(65+i)}) {choice}\n" for i, choice in enumerate(eval(x))])
    # )
    # # get element from choices_2 using correct_index_2
    # df["answer_2_formatted"] = df.apply(lambda x: eval(x["choices_2"])[int(x["correct_index_2"])], axis=1)
    #
    # # temp: create col question_answer_2_formatted from `question_2`, `choices_2_formatted`, `correct_index_2`
    # df["question_answer_2_formatted"] = (
    #     "Question:\n```"
    #     + df["question_2"]
    #     + "\n\n"
    #     + df["choices_2_formatted"]
    #     + "\nCorrect Answer: "
    #     + df["correct_index_2"].apply(lambda x: f"{chr(65 + x)}) ")
    #     + df["answer_2_formatted"]
    #     + "```"
    # )
    #
    # # find any none in "_formatted" columns
    # if df[["choices_2_formatted", "answer_2_formatted", "question_answer_2_formatted"]].isnull().values.any():
    #     logger.error(f"Error: NaN values found in the dataframe.")
    #     raise ValueError(f"Error: NaN values found in the dataframe.")
    #
    # df.to_excel(input_file, index=False)

    # filter df
    if filter_col and isinstance(filter_col, dict):
        logger.info(f"Filtering dataframe with {filter_col}")
        for key, value in filter_col.items():
            if key in df.columns:
                df = df[df[key] == value]
            else:
                logger.warning(f"Column {key} not found in the dataframe.")

    # set language
    textstat.set_lang(lang)

    # create text from the column
    if column not in df.columns:
        logger.error(f"Column {column} not found in the dataframe.")
        raise ValueError(f"Column {column} not found in the dataframe.")

    # get metric
    logger.info(f"Computing text difficulty using metric: {metric}")

    # compute reading difficulty for column
    df["consensus_difficulty"] = df[column].apply(
        textstat.text_standard, float_output=True
    )
    df["fk_difficulty"] = df[column].apply(textstat.flesch_kincaid_grade)
    df["fk_reading_ease"] = df[column].apply(textstat.flesch_reading_ease)

    # raise error if any of the values are NaN, inf or -inf or Zero
    new_cols = ["consensus_difficulty", "fk_difficulty", "fk_reading_ease"]
    if df[new_cols].isin([np.nan, np.inf, -np.inf, 0]).any().any():
        logger.error("Error: NaN, inf, -inf or Zero values found in the dataframe.")
        raise ValueError("Error: NaN, inf, -inf or Zero values found in the dataframe.")

    logger.info(
        f"{df[['consensus_difficulty', 'fk_difficulty', 'fk_reading_ease']].describe()}"
    )

    # save image
    if output_file.exists():
        click.confirm(
            f"Output file {output_file.name} already exists! Do you want to overwrite?",
            abort=True,
        )
        logger.warning(f"Overwriting {output_file}...")

    # save csv with difficulty values
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()
