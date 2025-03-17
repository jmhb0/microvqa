#!/usr/bin/env python3
"""convert_jsonl_to_csv.py in src/microchat."""
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from dotenv import find_dotenv
from dotenv import load_dotenv
from loguru import logger

from microchat import DATA_ROOT, PROJECT_ROOT

import re
from microchat.fileio.text.readers import yaml_loader
from microchat import MODULE_ROOT

# regex parse response into "Bloom's: Recall\nLevel: 1\nReasoning: <reasoning>
context = yaml_loader(Path(MODULE_ROOT, "conf", "question_context.yaml"))
blooms_dict = yaml_loader(Path(MODULE_ROOT, "conf", "blooms.yaml"))["taxonomy"].get(
    "revised"
)
blooms_list = [item for sublist in blooms_dict.values() for item in sublist]
re_blooms_compiled = re.compile(
    r"^Bloom's:" + "\s" + "|".join(blooms_list), re.IGNORECASE
)
re_level_compiled = re.compile(r"Level:" + "\s" + "(\d)", re.IGNORECASE)


@click.command()
@click.argument("input-file", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--output-dir", type=click.Path(file_okay=False, exists=False, path_type=Path)
)
@click.option("--dry-run", is_flag=True, help="Perform a trial run with no changes.")
@click.version_option()
def main(
    input_file: Path,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    """Docstring."""
    data_root = Path(DATA_ROOT)
    if not input_file.exists() and not data_root.joinpath(input_file).exists():
        logger.error(f"File not found: {input_file}")
        raise FileNotFoundError(f"File not found: {input_file}")
    elif not input_file.exists() and data_root.joinpath(input_file).exists():
        input_file = data_root.joinpath(input_file)

    if input_file.suffix != ".jsonl":
        logger.error(f"File must be a jsonl file: {input_file}")
        raise ValueError(f"File must be a jsonl file: {input_file}")

    # set output directory
    output_dir = output_dir or input_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        PROJECT_ROOT.joinpath("logs", f"{Path(__file__).stem}.log"),
        rotation="10 MB",
        level="INFO",
    )

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")

    # load jsonl
    image_json_df = pd.read_json(input_file, orient="records", lines=True)
    # extract field ["response"] with the real dict I need as df
    image_df = pd.DataFrame(image_json_df["response"].tolist())
    example_df = pd.DataFrame(image_df["body"].tolist())

    # response in example_df.loc[0]["choices"][0]["message"].get("content")
    response_df = pd.DataFrame(example_df["choices"].tolist())
    # get content from all rows
    response_df = response_df.map(lambda x: x["message"].get("content"))
    # rename col name from 0 to "response"
    response_df = response_df.rename(columns={0: "response"})

    # add "blooms_question_type", "blooms_level" columns
    response_df["blooms_question_type"] = None
    response_df["blooms_level"] = None
    response_df["blooms_reasoning"] = None

    # parse response
    response_df["response_list"] = response_df["response"].apply(
        lambda x: x.split("\n")
    )
    # response_df["blooms_question_type"] = response_df["response_list"].apply(lambda x: x.strip("Bloom's: "))

    for idx, row in response_df.iterrows():
        response = row["response"]
        if blooms := re_blooms_compiled.search(row["response"]):
            row["blooms_question_type"] = blooms.group(1)
            row["blooms_question_type"] = (
                row["blooms_question_type"].replace("Bloom's: ", "").strip()
            )

        if level := re_level_compiled.search(row["response"]):
            row["blooms_level"] = int(level.group(1))
            row["blooms_level"] = row["blooms_level"].replace("Level: ", "").strip()

        for item in response:
            if "Level" in item:
                row["blooms_level"] = item.split(": ")[1]
            if "Reasoning" in item:
                row["blooms_reasoning"] = item.split(": ")[1]

    response_df["blooms_qtype_re"] = response_df["response"].apply(
        lambda x: re_blooms_compiled.findall(x)
    )

    if dry_run:
        logger.info("Dry run: no changes will be made.")
        return


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()
