#!/usr/bin/env python3
"""run_dspy.py in src/microchat."""

from pathlib import Path
from typing import Optional
import click
import pandas as pd
from dotenv import find_dotenv
from dotenv import load_dotenv

from loguru import logger

import dspy
from tqdm import tqdm

from microchat import PROJECT_ROOT
from microchat.custom_datasets.dataset_factory import create_dataset
from microchat.fileio.dataframe.readers import df_loader
from microchat.models.dspy_modules import CoTRAG
from microchat.models.model_factory import create_model
from microchat.mc_questions.mcq import MCQ, Blooms
from microchat.teleprompters.teleprompter_factory import create_optimizer

try:
    import datasets

    if datasets.__version__ != "3.0.1":
        raise ImportError(
            f"Dataset may not be compatible with DSPy. Please install datasets==3.0.1."
        )
except ImportError as e:
    logger.error("Please install datasets==3.0.1.")
    logger.error(e)
    raise e


@click.command()
@click.argument("dataset_name", type=click.STRING)
@click.option("--model", type=click.STRING, default="gpt-4o-mini")
@click.option("--teacher-model", type=click.STRING, default="o1-mini")
@click.option("--retrieval-model", type=click.STRING, default="wiki17_abstracts")
@click.option("--optimizer", type=click.STRING, default="bootstrap_random")
@click.option(
    "--output-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option("--task", type=click.Choice(["nbme", "blooms"]), default="blooms")
@click.option("--random-seed", type=click.INT, default=8675309)
@click.option("--retrieve-k", type=click.IntRange(3, 10), default=5)
@click.option("--dry-run", is_flag=True, help="Perform a trial run with no changes.")
@click.version_option()
def main(
    dataset_name: str,
    model: Optional[str] = "gpt-4o-mini",
    teacher_model: Optional[str] = "o1-mini",
    retrieval_model: Optional[str] = "wiki17_abstracts",
    optimizer: Optional[str] = "bootstrap",
    output_dir: Optional[Path] = None,
    task: Optional[str] = "blooms",
    random_seed: int = 8675309,
    retrieve_k: int = 5,
    dry_run: bool = False,
) -> None:
    """Docstring."""
    if not output_dir:
        output_dir = Path(PROJECT_ROOT).joinpath("outputs", dataset_name.strip(".csv"))
    output_dir.mkdir(parents=True, exist_ok=True)

    project_dir = Path(__file__).parents[2]
    logger.add(
        project_dir.joinpath("logs", f"{Path(__file__).stem}.log"),
        rotation="10 MB",
        level="INFO",
    )

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Model: {model}")
    logger.info(f"Teacher model: {teacher_model}") if teacher_model else None
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Output directory: {output_dir}")

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

    # define retrieval model
    colbertv2_model = None

    # configure DSPy settings
    dspy.settings.configure(lm=model.lm, rm=colbertv2_model)

    # set task with question_key and answer_key
    # TODO: load config from yaml
    if task == "blooms" or "blooms" in dataset_name:
        question_key = "question_answer"  # "revised_question_answer"  # "question" #"original_question_answer"
        answer_key = "blooms_question_category"  # "blooms_question_category"  # "revised_question" #"revised_question_answer"
    elif task == "nbme":
        question_key = "original_question_answer"
        answer_key = "revised_question_answer"
    else:
        logger.error(f"Task {task} not implemented.")
        raise NotImplementedError(f"Task {task} not implemented.")

    # instantiate dataset
    subset = [question_key, answer_key]
    dataset = create_dataset(
        dataset_name, subset=subset, question_key=question_key, answer_key=answer_key
    )

    # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
    trainset = [x.with_inputs("question") for x in dataset.train]
    devset = [x.with_inputs("question") for x in dataset.dev]

    print(f"{len(trainset)}, {len(devset)}")

    train_example = trainset[0]
    dev_example = devset[0]
    logger.info(f"Train question: {train_example.question}")
    logger.info(f"Train answer: {train_example.answer}")

    # create module
    model_dir = Path(PROJECT_ROOT).joinpath("models/dspy_compiled/blooms/")
    module = CoTRAG(context=task)
    module.name = module.__class__.__name__
    model_filepath = model_dir.joinpath(
        f"{model.model_name}_{module.name}_{module.signature_name}.json"
    )
    if model_filepath.exists():
        logger.info(f"Loading trained model {model_filepath.stem}")
        module.load(model_filepath)
    else:
        logger.error(f"Model {model_filepath} not found.")

    ## hack:  temp code to loop over all examples to get consensus Bloom's category
    # loop over trainset and save response.answer to output_list, which will be new
    # consensus label
    start_idx = 0
    teacher_model = create_model(teacher_model).lm
    output_list = []
    for idx, example in tqdm(
        enumerate(trainset + devset), total=len(trainset + devset)
    ):
        if idx < start_idx:
            continue
        # get consensus blooms
        # the initial label was from MicroChat-MC
        # this loop will use predict using o1-mini and self-assess
        # or correct with gpt-4o. any examples without 100% agreement
        # among MicroChat-MC, o1-mini, and, gpt-4o will get human review
        try:
            if task == "blooms":
                response = Blooms(
                    example=example, module=module, teacher_model=teacher_model
                )
            elif task == "nbme":
                response = MCQ(
                    example=example, module=module, teacher_model=teacher_model
                )

            output_list.append(
                {
                    "key_image": response.example.key_image,
                    "key_question": response.example.key_question,
                    "revised_question_answer": response.example.question,
                    "context": response.context,
                    "self_check_question": response.self_check_question,
                    "blooms_question_category": response.blooms_name.capitalize(),
                    "blooms_confidence": response.blooms_confidence,
                    "blooms_level": response.blooms_level,
                    "blooms_source": response.blooms_source,
                    "blooms_reasoning": response.blooms_reasoning,
                }
            )
            if idx % 5 == 0:
                logger.info(
                    f"Example {idx}:\nGT\t{response.gt_level}\nPred\t{response.blooms_level} ({response.blooms_confidence:.2f})"
                )
                output_file = output_dir.joinpath(
                    "blooms_classification_finetuning_v2_temp.csv"
                )
                temp_output_df = pd.DataFrame(output_list)
                temp_output_df.to_csv(
                    output_file,
                    index=False,
                    mode="a",
                )
        except Exception as e:
            logger.error(
                f"Error with example {example.key_image} {example.key_question}: {e}"
            )
            output_file = output_dir.joinpath(
                "blooms_classification_finetuning_v2_temp.csv"
            )
            temp_output_df = pd.DataFrame(output_list)
            temp_output_df.to_csv(
                output_file,
                index=False,
                mode="a",
            )

    # convert to df
    output_df = pd.DataFrame(output_list)

    # load orig df
    orig_df = df_loader(Path(PROJECT_ROOT).joinpath("data", "processed", dataset_name))

    # merge on key_image and key_question
    output_df = pd.merge(
        orig_df,
        output_df,
        on=["key_image", "key_question"],
        how="left",
        suffixes=("", "_pred"),
    )

    # print rows with non null for both blooms_question_category and blooms_question_category_pred
    idx = output_df.loc[
        (output_df["blooms_question_category"].notnull())
        & (output_df["blooms_question_category_pred"].notnull())
    ].index
    print(
        output_df.loc[
            idx,
            [
                "blooms_question_category",
                "blooms_question_category_pred",
                "blooms_source",
            ],
        ]
    )

    # rename values in blooms_question_category_pred
    rename_map = {
        "Recall": "Recall",
        "Remember": "Recall",
        "Memorize": "Recall",
        "Knowledge": "Recall",
        "Comprehension": "Comprehension",
        "Comprehend": "Comprehension",
        "Understand": "Comprehension",
        "Understanding": "Comprehension",
        "Application": "Application",
        "Apply": "Application",
        "Applying": "Application",
        "Analysis": "Analysis",
        "Analyze": "Analysis",
        "Analyzing": "Analysis",
        "Evaluation": "Evaluation",
        "Evaluate": "Evaluation",
        "Evaluating": "Evaluation",
        "Synthesis": "Synthesis",
        "Synthesizing": "Synthesis",
    }
    output_df["blooms_question_category_pred"] = output_df[
        "blooms_question_category_pred"
    ].map(rename_map)

    # save to csv
    output_df.to_csv(
        output_dir.joinpath(f"{model.model_name}_update_blooms.csv"), index=False
    )

    # # convert col blooms_source_pred to set
    # output_df["blooms_source_pred"] = output_df["blooms_source_pred"].apply(
    #     lambda x: ",".join(set(x.split(" & ")))
    # )

    output_df_2 = pd.read_csv(
        output_dir.joinpath(f"{model.model_name}_update_blooms.csv"), index_col=None
    )
    # drop dups
    total_len = len(output_df_2)
    output_df_2 = output_df_2.drop_duplicates(
        subset=["key_question", "key_image", "revised_question_answer"], keep="first"
    )

    # groupby key_question, and find most common blooms_question_category
    temp = (
        output_df_2.copy()
        .groupby("key_question")[
            ["blooms_question_category_pred", "blooms_source", "blooms_level"]
        ]
        .agg(
            blooms_question_category=(
                "blooms_question_category_pred",
                lambda x: x.mode(),
            ),
            blooms_level=("blooms_level", lambda x: x.mode()),
            blooms_source=("blooms_source", lambda x: x.mode()),
            # fraction of blooms_level that are in the majority class
            blooms_confidence=(
                "blooms_level",
                lambda x: x.value_counts().max() / len(x),
            ),
        )
        .reset_index()
    )

    # get rows with non null and non empty list
    idx_temp = temp.loc[
        temp["blooms_question_category_pred"].notnull()
        & temp["blooms_question_category_pred"].apply(lambda x: len(x) > 0)
    ].index
    temp = temp.loc[idx_temp]

    #
    temp.to_csv(
        output_dir.joinpath(f"{model.model_name}_update_blooms_agg.csv"), index=False
    )

    #
    module.save(output_dir.joinpath(f"{model.model_name}_demos.json"))


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()
