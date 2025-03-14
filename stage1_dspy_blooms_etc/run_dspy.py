#!/usr/bin/env python3
"""run_dspy.py in src/microchat."""
import pprint
from pathlib import Path
from typing import Optional
import click
import pandas as pd

from dotenv import find_dotenv
from dotenv import load_dotenv

from loguru import logger
from tqdm import tqdm

import dspy
from dspy.evaluate.evaluate import Evaluate

from microchat import PROJECT_ROOT
from microchat.custom_datasets.dataset_factory import create_dataset
from microchat.fileio.dataframe.readers import df_loader
from microchat.fileio.text.readers import yaml_loader
from microchat.fileio.text.writers import yaml_writer
from microchat.metrics.mcq_metric import (
    validate_blooms,
    validate_nbme,
    validate_tagging,
)
from microchat.models.dspy_modules import CoTSelfCorrectRAG, CoTRAG, context
from microchat.models.model_factory import create_model
from microchat.mc_questions.mcq import MCQ, Blooms
from microchat.teleprompters.teleprompter_factory import create_optimizer
from microchat.utils.process_model_history import history_to_jsonl

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

try:
    from langtrace_python_sdk import langtrace

    # langtrace.init(api_key=os.getenv("LANGTRACE_API_KEY"))
except ImportError as e:
    logger.warning("Langtrace not installed.")


@click.command()
@click.argument("dataset_name", type=click.STRING)  # blooms.csv
@click.option("--model", type=click.STRING, default="o1-mini")
@click.option("--teacher-model", type=click.STRING, default="o1-mini")
@click.option("--retrieval-model", type=click.STRING, default=None)
@click.option("--optimizer", type=click.STRING, default="miprov2")
@click.option(
    "--output-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--task",
    type=click.Choice(["nbme", "blooms", "hotpotqa", "organism_research"]),
    default="blooms",
)
@click.option("--random-seed", type=click.INT, default=8675309)
@click.option("--retrieve-k", type=click.IntRange(1, 10), default=5)
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
    logger.info(f"Task: {task}")
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
    teacher_lm = None  # create_model(teacher_model)

    # define retrieval model
    colbertv2_model = None
    if retrieval_model or task == "hotpotqa":
        logger.info(f"Retrieval model: {retrieval_model}")
        logger.info(f"Retrieve k: {retrieve_k}")
        colbertv2_model = dspy.ColBERTv2(
            url="http://20.102.90.50:2017/wiki17_abstracts"
        )

    # configure DSPy settings
    dspy.settings.configure(lm=model.lm, rm=colbertv2_model)

    ### parse MCQ options
    input_file = Path("./data/processed/").joinpath(
        "llm-qc_eval_naive.csv"
    )
    df = df_loader(input_file)

    df["self_assess_llm"] = df["self_assess_llm"].fillna("")

    # apply MCQ to col "question_answer" in df
    # create MCQ object
    start_idx = -1
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if idx < start_idx:
            continue

        # strip ending newline
        text = row["naive_question_answer_formatted"]

        if row["self_assess_llm"] and not pd.isna(row["similarity_metric"]):
            logger.info(f"Skipping index {idx}, which was already reviewed")
            continue

        # format example and prediction
        example = dspy.Example(
            question=row["description_question_answer"],
            key_question=row["key_question"],
            key_image=row["key_image"],
        )
        pred = dspy.Prediction(answer=text)
        try:
            mcq = MCQ(example=example, prediction=pred)

            # info
            if df.loc[idx, "description"] != mcq.example_dict.get("description"):
                logger.warning(f"Description mismatch at index {idx}")

            if df.loc[idx, "original_question"] != mcq.example_dict.get("question"):
                logger.warning(f"Original question mismatch at index {idx}")

            if (
                df.loc[idx, "original_answer"]
                != mcq.example_dict.get("correct_answer").strip()
            ):
                logger.warning(f"Original answer mismatch at index {idx}")

            # update df
            df.loc[idx, "similarity_metric"] = mcq.metrics.get("similarity")
            df.loc[idx, "formatted_metric"] = mcq.metrics.get("formatted")
            df.loc[idx, "extraneous_metric"] = mcq.metrics.get("extraneous")
            df.loc[idx, "option_token_ratio_metric"] = mcq.metrics.get(
                "option_token_ratio"
            )
            df.loc[idx, "reasoning"] = mcq.metrics.get("reasoning", None)
            df.loc[idx, "self_assess_llm"] = model.model_name

            df.loc[idx, "description"] = mcq.example_dict.get("description")
            df.loc[idx, "additional_info"] = mcq.example_dict.get("additional_info")
            df.loc[idx, "original_question"] = mcq.example_dict.get("question")
            df.loc[idx, "original_question_tokens"] = mcq.get_tokens(
                original=True, field="question"
            )
            df.loc[idx, "original_answer"] = mcq.example_dict.get(
                "correct_answer"
            ).strip()
            df.loc[idx, "original_answer_tokens"] = mcq.get_tokens(
                original=True, field="correct_answer"
            )
            df.loc[idx, "revised_question"] = mcq.prediction_dict.get("question")
            df.loc[idx, "revised_question_tokens"] = mcq.get_tokens(
                original=False, field="question"
            )
            df.loc[idx, "revised_answer"] = mcq.prediction_dict.get(
                "correct_answer"
            ).strip()
            df.loc[idx, "revised_answer_tokens"] = mcq.get_tokens(
                original=False, field="correct_answer"
            )
            df.loc[idx, "options"] = pprint.pformat(mcq.prediction_dict["options"])
            df.loc[idx, "correct_index"] = mcq.prediction_dict["correct_index"]
            df.to_csv(input_file, index=False)
        except Exception as e:
            logger.error(f"Error creating MCQ for index {idx}: {e}")
            df.to_csv(input_file, index=False)

    # # save
    df.to_csv(input_file, index=False)

    # set task with question_key and answer_key
    # TODO: load config from yaml
    if task == "blooms":
        question_key = "question_answer"  # "revised_question_answer"  # "question" #"original_question_answer"
        answer_key = "blooms_question_category"  # "blooms_question_category"  # "revised_question" #"revised_question_answer"
        metric = validate_blooms
        eval_metric = dspy.evaluate.answer_exact_match
        subset = [question_key, answer_key]
    elif task == "nbme":
        question_key = "question"  # "original_question_answer"
        answer_key = "answer"  # "revised_question_answer"
        metric = validate_nbme
        eval_metric = validate_nbme
        subset = [question_key, answer_key]
    elif task == "hotpotqa":
        question_key = "question"
        answer_key = "answer"
        metric = dspy.evaluate.answer_exact_match
        eval_metric = dspy.evaluate.answer_exact_match
        subset = [question_key, answer_key]
    elif task == "organism_research":
        question_key = "description_question_answer"
        answer_key = "original_answer"
        metric = validate_tagging
        eval_metric = validate_tagging
        subset = [question_key, answer_key, "organism", "specimen", "research_subject"]
    else:
        logger.error(f"Task {task} not implemented.")
        raise NotImplementedError(f"Task {task} not implemented.")

    # instantiate dataset
    dataset = create_dataset(
        dataset_name, subset=subset, question_key=question_key, answer_key=answer_key
    )

    # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
    trainset = [x.with_inputs("context", "question") for x in dataset.train]
    devset = [x.with_inputs("context", "question") for x in dataset.dev]
    if not trainset or not devset:
        logger.error(f"Empty dataset: {dataset_name}")
        raise ValueError(f"Empty dataset: {dataset_name}")

    print(f"{len(trainset)}, {len(devset)}")

    # Set up a teleprompter/optimizer, which will compile our RAG program.
    optimizer, metric = create_optimizer(
        optimizer, teacher_model=teacher_model, metric=metric
    )

    # create module
    module = CoTRAG(context=task, num_passages=retrieve_k)
    module.name = module.__class__.__name__

    # Set up the `evaluator` function. We'll use this many times below.
    evaluator = Evaluate(
        devset=devset,
        num_threads=1,
        display_progress=True,
        provide_traceback=True,
        metric=eval_metric,
        return_outputs=True,
    )

    # # results yaml
    results_file = output_dir.joinpath("results.yaml")
    model_filepath = output_dir.joinpath(
        f"{model.model_name}_{module.name}_{module.signature_name}.json"
    )

    # evalute zero shot
    zs_score, zs_results = evaluator(module, metric=eval_metric)
    logger.info(f"Zero-shot score: {zs_score}")
    if results_file.exists():
        results = yaml_loader(results_file)
        if model.model_name not in results:
            results[model.model_name] = {}
        results[model.model_name][module.signature_name]["zero_shot_score"] = float(
            zs_score
        )
        yaml_writer(results, results_file)
    else:
        yaml_writer(
            {
                model.model_name: {
                    module.signature_name: {"zero_shot_score": float(zs_score)}
                }
            },
            results_file,
        )

    # compile rag
    logger.info(
        f"Compiling {module.name} with optimizer {optimizer.name} and model {model.model_name}"
    )
    compiled_rag = optimizer.compile(
        module, trainset=trainset, minibatch_size=len(devset)
    )

    # save compiled rag
    compiled_rag.save(output_dir.joinpath(model_filepath))
    # save history for the last 5 examples
    history_to_jsonl(
        model.lm, output_dir, output_file=f"{model.model_name}_history.jsonl", n=5
    )

    # Evaluate the `compiled_rag` program with the specified metric.
    if not model_filepath.exists():
        logger.error(f"Error saving compiled RAG to {model_filepath}")
        raise FileNotFoundError(f"Error saving compiled RAG to {model_filepath}")

    logger.info(f"Loading compiled RAG from {model_filepath}")
    trained_module = CoTRAG(context=task, num_passages=retrieve_k)
    trained_module.load(model_filepath)
    trained_module.name = trained_module.__class__.__name__

    # evaluate trained rag
    score, results = evaluator(trained_module, metric=eval_metric)
    results_df = pd.DataFrame(results, columns=["example", "prediction", "score"])
    logger.info(f"Compiled RAG score: {score}")
    # save score to yaml
    if results_file.exists():
        results = yaml_loader(results_file)
        if model.model_name not in results:
            results[model.model_name] = {}
        results[model.model_name][module.signature_name][optimizer.name] = float(score)
        yaml_writer(results, results_file)
    else:
        yaml_writer(
            {model.model_name: {module.signature_name: {optimizer.name: float(score)}}},
            results_file,
        )

    # Save the results
    results_df.to_csv(
        output_dir.joinpath(f"{model_filepath.stem}_results.csv"), index=False
    )


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()

