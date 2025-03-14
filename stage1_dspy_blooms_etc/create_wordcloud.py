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

import matplotlib.pyplot as plt

from wordcloud import WordCloud


@click.command()
@click.option("--input-file", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--output-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--column", default="question", type=str, help="Column to use for wordcloud."
)
@click.option(
    "--filter_col",
    default=None,
    type=dict,
    help="Filter to apply to the dataframe. {column_name: value}",
)
@click.option("--height", default=800, type=int, help="Height of the wordcloud image.")
@click.option("--width", default=1600, type=int, help="Width of the wordcloud image.")
@click.option(
    "--random-seed", default=8675309, type=int, help="Random seed for reproducibility."
)
@click.option("--plot", is_flag=True, help="Plot the wordcloud.")
@click.option("--dry-run", is_flag=True, help="Perform a trial run with no changes.")
@click.version_option()
def main(
    input_file: Path,
    output_dir: Optional[Path] = None,
    column: str = "question_stem",
    filter_col: Optional[Dict[Any, Any]] = None,
    height: int = 800,
    width: int = 1600,
    random_seed: int = 8675309,
    plot: bool = True,
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
    output_file = output_dir.joinpath(f"{input_file.stem}_wordcloud.png")

    project_dir = Path(__file__).parents[2]
    logger.add(
        project_dir.joinpath("logs", f"{Path(__file__).stem}.log"),
        rotation="10 MB",
        level="INFO",
    )

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output file: {output_file}")

    if dry_run:
        logger.info("Dry run: no changes will be made.")
        return

    # Read the input file
    df = pd.read_csv(input_file)

    # filter df
    if filter_col:
        logger.info(f"Filtering dataframe with {filter_col}")
        for key, value in filter_col.items():
            if key in df.columns:
                df = df[df[key] == value]
            else:
                logger.warning(f"Column {key} not found in the dataframe.")

    # create text from the column
    if column not in df.columns:
        logger.error(f"Column {column} not found in the dataframe.")
        raise ValueError(f"Column {column} not found in the dataframe.")

    text = " ".join(df[column].tolist())

    # instantiate
    wordcloud = WordCloud(
        width=width, height=height, background_color="white", random_state=random_seed
    )

    # generate
    wordcloud.generate(text)

    # save image
    if output_file.exists():
        click.confirm(
            f"Output file {output_file.name} already exists! Do you want to overwrite?",
            abort=True,
        )
        logger.warning(f"Overwriting {output_file}...")

    wordcloud.to_file(output_file)

    # plot
    if plot:
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()
