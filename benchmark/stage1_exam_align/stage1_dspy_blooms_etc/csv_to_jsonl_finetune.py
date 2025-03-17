#!/usr/bin/env python3
"""csv_to_jsonl_finetune.py in src/microchat."""
from pathlib import Path
import random
from typing import Optional

import click
from dotenv import find_dotenv
from dotenv import load_dotenv
from loguru import logger

from sklearn.model_selection import train_test_split

from microchat import MODULE_ROOT, DATA_ROOT
from microchat.fileio.dataframe.readers import df_loader
from openai import OpenAI

from microchat.fileio.text.readers import yaml_loader, save_jsonl
from microchat.fileio.text.writers import json_writer

# TODO: move PROMPT, CONTEXT and SUFFIX to a conf.yaml file
PREFIX = [
    "You are an expert in Biomedical AI with deep knowledge of Bloom's taxonomy and training from the National Board of Medical Examiners. Your role is to assist in designing multiple-choice benchmarks that test vision-language models' perception and reasoning capabilities by classifying questions or question-answer pairs into the most appropriate Bloom's taxonomy level according to the Revised Bloom's taxonomy and NBME guidelines.",
    "You have expertise in biomedical teaching and education with a deep understanding of Bloom's taxonomy.",
    "You are an expert in biomedical teaching and education with a deep knowledge of Bloom's taxonomy."
    "You are a Biology and Biomedical AI expert assisting in designing multiple-choice questions to test vision-language models' perception and reasoning. Your task is to assign the most appropriate level in Bloom's Revised Taxonomy to user-submitted questions or question-answer pairs. Trained by the National Board of Medical Examiners, you are deeply familiar with Bloom's taxonomy and assessing cognitive levels. You always state any uncertainties and strive to improve the accuracy of your assessments.",
    "You are an expert in Biology and BioMedical AI assisting in designing multiple-choice questions to test vision-language models' perception and reasoning. Your task is to take user-submitted questions or question-answer pairs and assign the most appropriate level in Bloom's Revised Taxonomy to each pair. You are deeply familiar with Bloom's taxonomy and trained by the National Board of Medical Examiners on assessing the cognitive levels of multiple-choice questions. You always state if you are uncertain about the classification and continually seek to improve the accuracy of your assessments.",
    "You are a factual chatbot with expertise in Bloom's taxonomy and biomedical education. Your task is to classify user-submitted questions or question-answer pairs into the most appropriate Bloom's taxonomy level according to the Revised Bloom's taxonomy and NBME guidelines. You always strive to improve the accuracy of your assessments and are transparent about any uncertainties in your classifications.",
    "As an AI assistant with expertise in Bloom's taxonomy and biomedical education, your role is to classify user-submitted questions or question-answer pairs into the most appropriate Bloom's taxonomy level according to the Revised Bloom's taxonomy and NBME guidelines. You always strive to improve the accuracy of your assessments and are transparent about any uncertainties in your classifications.",
    "You know Bloom's taxonomy and biomedical education well. Your task is to classify user-submitted questions or question-answer pairs into the most appropriate Bloom's taxonomy level according to the Revised Bloom's taxonomy and NBME guidelines. You always strive to improve the accuracy of your assessments and are transparent about any uncertainties in your classifications.",
]
CONTEXT = [
    """Consider the definition of the revised Bloom's Taxonomy levels below:
    A group of educational researchers and cognitive psychologists developed the new and revised Bloom’s Taxonomy framework in 2001 to be more action-oriented. This way, students work their way through a series of verbs to meet learning objectives. Below are descriptions of each of the levels in revised Bloom’s Taxonomy:
      - Remember/Recall: To bring an awareness of the concept to learners’ minds.
      - Understand/Comprehend: To summarize or restate the information in a particular way.
      - Apply: The ability to use learned material in new and concrete situations.
      - Analyze: Understanding the underlying structure of knowledge to be able to distinguish between fact and opinion.
      - Synthesis/Evaluate: Making judgments about the value of ideas, theories, items and materials.
      - Create: Reorganizing concepts into new structures or patterns through generating, producing or planning.""",
    """Consider the definition of the revised Bloom's Taxonomy levels for biomedical image-based reasoning below:
    # Revised Bloom's classification applied to biology or histology multiple-choice image questions:
      Level 1: Recall - Basic definitions, facts, and terms as well as basic image classification or object identification. Recall facts and basic concepts (ex. recall, define, memorize) (Krathwohl 2002)
        - Skills assessed: Recall, memorization
        - Recall MC questions: These questions only require recall. Students may memorize the answer without understanding the concepts of process. Recall questions test whether students know the "what" but does not test if they understand the "why".
      Level 2: Comprehension (aka understand) - Basic understanding of architectural and subcellular organization of cells and tissues, and concepts (organelles, tissue types, etc). Interpretation of subcellular organization, cell types, and organs from novel images, often limited to a single cell type or structure. Explain ideas or concepts, without relating to anything else (ex. classify, identify, locate) (Krathwohl 2002). "Requires recall and comprehension of facts. Image questions asking to identify a structure/cell type without requiring a full understanding of the relationship of all parts" (Zaidi 2017).
        - Skills assessed: Explain, identify, classify, locate
        - Comprehension MC questions:  These questions require recall and comprehension of facts. Image questions asking to identify a structure/cell type without requiring a full understanding of all parts. The process of identification requires students to evaluate internal or external contextual clues without requiring knowledge of functional aspects.
      Level 3: Application - Visual identification in new situations by applying acquired knowledge. Additional functional or structural knwoledge about the cell/tissue is also required. Use information in new situations (ex. apply, implement, use) (Krathwohl,2002). "Two-step questions that require image-based identification as well as the application of knowledge (e.g., identify structure and know function/ purpose)" (Zaidi 2017).
        - Skills assessed: Apply, connect
        - Application MC questions:  Two-step questions that require image-based identification as well as the application of knowledge (e.g., identify structure and explain/demonstrate knowledge of function/purpose).
      Level 4: Analysis - Visual identification and analysis of *comprehensive* additional knowledge. Connection between structure and function confined to single cell type/structure. Draw connections among ideas (ex. organize, analyze, calculate, compare, contrast, attribute) (Krathwohl 2002) "Students must call upon multiple independent facts and properly join them together." (Zaidi 2017).
        - Skills assessed: Analyze, classify
        - Analysis MC questions: Students must call upon multiple independent facts and properly join them together. May be required to correctly analyze accuracy of multiple statements in order to elucidate the correct answer. The student must also evaluate all options and understand all steps and can't rely on simple recall.
      Level 5: Synthesis/Evaluation - Interactions between different cell types/tissues to predict relationships; judge and critique knowledge of multiple cell types/tissues at the same time in new situations. Potential to use scientific or clinical judgement to make decisions. Justify a decision (ex. critique, judge, predict, appraise) (Krathwohl 2002).
        - Skills assessed: Predict, judge, critique, decide, evaluate
        - Synthesis/evaluation MC questions: Use information in a *new* context with the possibility for a scientific or clinical judgement. Students are required to go through multiple steps and apply those connections to a situation (e.g., predicting an outcome, scientific result, or diagnosis or critiquing a suggested experimental or clinical plan.)""",
    """# Revised Bloom's classification applied to biology or histology multiple-choice image questions:
    Recall
        Skills assessed: Recall
        Description: Basic definitions, facts, and terms, as well as basic image classification or object identification.
        Recall MC questions: Require only memorization. Students may know the "what" but not the "why." These questions do not test understanding of concepts or processes.
    Comprehension
        Skills assessed: Explain, identify
        Description: Basic understanding of the architectural and subcellular organization of cells and tissues, and concepts like organelles and tissue types. Involves interpretation of subcellular organization, cell types, and organs from novel images, often limited to a single cell type or structure.
        Comprehension MC questions: Require recall and comprehension of facts. Students identify structures or cell types without needing a full understanding of all parts. Identification relies on evaluating contextual clues without requiring knowledge of functional aspects.
    Application
        Skills assessed: Apply, connect
        Description: Visual identification in new situations by applying acquired knowledge. Requires additional functional or structural knowledge about the cell or tissue.
        Application MC questions: Two-step questions that involve image-based identification and the application of knowledge (e.g., identifying a structure and explaining its function or purpose).
    Analysis
        Skills assessed: Analyze, classify
        Description: Visual identification and analysis of comprehensive additional knowledge. Connects structure and function confined to a single cell type or structure.
        Analysis MC questions: Students must integrate multiple independent facts. They may need to analyze the accuracy of several statements to find the correct answer, requiring evaluation of all options and a deep understanding beyond simple recall.
    Synthesis/Evaluation
        Skills assessed: Predict, judge, critique, decide
        Description: Involves interactions between different cell types or tissues to predict relationships. Requires judging and critiquing knowledge of multiple cell types or tissues simultaneously in new situations, potentially using scientific or clinical judgment to make decisions.
        Synthesis/Evaluation MC questions: Students use information in a new context with the possibility of making scientific or clinical judgments. They must go through multiple steps and apply connections to situations like predicting outcomes, scientific results, diagnoses, or critiquing experimental or clinical plans.""",
]
SUFFIX = [
    "Provide an assessment of the most appropriate Bloom's Taxonomy level for the provided question.",
    "Review the question and assign the most appropriate Bloom's Taxonomy level.",
    "What is the most appropriate Bloom's Taxonomy level for the provided question?",
    "Assign the most appropriate Bloom's Taxonomy level for the provided question. Double-check your classification and make adjustments if necessary to ensure the question stem accurately reflects the appropriate level of cognitive skills according to Bloom's taxonomy.",
    "What is the most appropriate Bloom's Taxonomy level for the provided question? After the initial evaluation, ask yourself: 'Are you sure about the Bloom's taxonomy category?' Double-check your classification and make adjustments if necessary to ensure the question stem accurately reflects the appropriate level of cognitive skills according to Bloom's taxonomy."
    "Review the question and assign the most appropriate Bloom's Taxonomy level. After making an initial assessment of the Bloom's classification, ask yourself: 'Are you sure about the Bloom's taxonomy category?' Double-check your classification and make adjustments if necessary to ensure the question stem accurately reflects the appropriate level of cognitive skills according to Bloom's taxonomy.",
]


def format_data_inference(
    row,
    url: str = "/v1/chat/completions",
    model: str = "ft:gpt-4o-mini-2024-07-18:marvl-lab:gpt-4o-mini-2024-07-18-blooms:ANR07kPO",
    system_context: str = PREFIX[0],
):
    custom_id = row["question_id"]
    user = row["user"]
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": url,
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_context},
                {"role": "user", "content": user},
            ],
            "max_tokens": 1000,
        },
    }


def format_data_train(row):
    return {
        "messages": [
            {"role": "system", "content": row["system_context"]},
            {"role": "user", "content": row["user"]},
            {"role": "assistant", "content": row["assistant"]},
        ]
    }


def create_system_context(row, PREFIX, CONTEXT):
    prefix = random.choice(PREFIX)
    context = random.choice(CONTEXT)

    # randomly include more detailed context in system_context
    return f"{prefix}" if random.randint(0, 1) else f"{prefix}\n{context}\n"


def create_user(row, user_columns: list, suffix=SUFFIX):
    user = random.choice(suffix) + "\n"
    for col in user_columns:
        user += f"{col.replace('_',' ').capitalize()}: {row[col]}\n"

    user = user.strip()
    return user


def create_assistant(row, assistant_columns: list):
    assistant = ""
    for col in assistant_columns:
        if col == "blooms_question_category":
            assistant += f"Bloom's: {row[col]}\n"
        elif col == "blooms_level":
            assistant += f"Level: {row[col]}\n"
        elif col == "blooms_reasoning":
            assistant += f"Reasoning: {row[col]}\n"
        else:
            logger.error(f"Column not found: {col}")
            raise ValueError(f"Column not found: {col}")

    assistant = assistant.strip()
    return assistant


@click.command()
@click.argument("input-file", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--output-file", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--train", is_flag=True, help="Format data for openai fine-tuning.")
@click.option(
    "--n-sample", type=int, help="Number of samples to extract from the input file."
)
@click.option(
    "--random-seed", type=int, default=8675309, help="Random seed for shuffling."
)
@click.option(
    "--system-context",
    type=str,
    help="Column name for system context or static text to use for all rows.",
)
@click.option(
    "--user",
    type=click.STRING,
    default="question_stem",
    help="Column name for user input.",
)
@click.option(
    "--assistant",
    type=click.STRING,
    default="correct_answer",
    help="Column name for assistant output.",
)
@click.option("--upload", is_flag=True, help="Upload the output file to openai.")
@click.option("--dry-run", is_flag=True, help="Perform a trial run with no changes.")
@click.version_option()
def main(
    input_file: Path,
    output_file: Optional[Path] = None,
    system_context: Optional[str] = None,
    user: Optional[str] = "question_stem",
    assistant: Optional[str] = "correct_answer",
    upload: bool = False,
    random_seed: int = 8675309,
    n_sample: Optional[int] = None,
    train: bool = False,
    dry_run: bool = False,
) -> None:
    """Docstring."""
    data_root = Path(DATA_ROOT)
    if not input_file.exists() and not data_root.joinpath(input_file).exists():
        logger.error(f"File not found: {input_file}")
        raise FileNotFoundError(f"File not found: {input_file}")
    elif not input_file.exists() and data_root.joinpath(input_file).exists():
        input_file = data_root.joinpath(input_file)

    # set vars
    output_file = output_file or input_file.with_suffix(".jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    required_cols = ["question_stem", "correct_answer"]
    if train:
        logger.info("Training mode enabled. Formatting data for openai fine-tuning.")
        required_cols = [
            "question_stem",
            "correct_answer",
            "blooms_question_category",
            "blooms_level",
            "blooms_reasoning",
        ]

    logger.add(
        MODULE_ROOT.joinpath("logs", "csv_to_openai_finetune.log"),
        rotation="10 MB",
        level="INFO",
    )

    client = OpenAI()
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.warning(
        "The system currently only supports Bloom's taxonomy classification."
    )

    # load df
    df = df_loader(input_file)
    logger.info(f"Loaded {len(df)} rows from {input_file}")

    # drop duplicates
    df_size = len(df)
    df.drop_duplicates(inplace=True)
    if df_size != len(df):
        logger.info(f"Dropped {df_size - len(df)} duplicate rows.")

    # drop rows with missing values
    df_size = len(df)
    df.dropna(subset=required_cols, inplace=True)
    if df_size != len(df):
        logger.info(f"Dropped {df_size - len(df)} rows with missing values.")

    if df.empty or len(df) < 2:
        logger.error("No data to process.")
        raise ValueError("No data to process.")

    # format
    if system_context is None:
        df["system_context"] = df.apply(
            create_system_context, axis=1, args=(PREFIX, CONTEXT)
        )
    elif system_context not in df.columns:
        logger.error(f"Column not found: {system_context}")
        raise ValueError(f"Column not found: {system_context}")

    # create user input
    if user is None or user not in df.columns:
        logger.error(f"Column not found: {user}")
        raise ValueError(f"Column not found: {user}")

    df["user"] = df.apply(
        create_user, axis=1, user_columns=["question_stem", "correct_answer"]
    )

    if train:
        train_file = output_file.parent.joinpath(f"{output_file.stem}_train.jsonl")
        test_file = output_file.parent.joinpath(f"{output_file.stem}_test.jsonl")

        if train_file.exists() or test_file.exists():
            logger.warning(f"Output file already exists: {train_file} or {test_file}")
            click.confirm("Do you want to overwrite?", abort=True)

        # create assistant response
        if assistant is None or assistant not in df.columns:
            logger.error(f"Column not found: {assistant}")
            raise ValueError(f"Column not found: {assistant}")

        df["assistant"] = df.apply(
            create_assistant,
            axis=1,
            assistant_columns=[
                "blooms_question_category",
                "blooms_level",
                "blooms_reasoning",
            ],
        )

        # format data in the dataframe
        df["template"] = df.apply(format_data_train, axis=1)

        # shuffle
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # train test split, stratified
        idx = range(len(df))
        labels = df["blooms_question_category"].astype("category").cat.codes.to_numpy()
        # df["blooms_question_category"].value_counts()
        train, test = train_test_split(
            idx, stratify=labels, test_size=0.2, random_state=random_seed
        )

        # save jsonl train and test
        train_df = df.loc[train]
        test_df = df.loc[test]

        # save jsonl
        if not dry_run:
            train_list = train_df["template"].tolist()
            test_list = test_df["template"].tolist()
            save_jsonl(train_list, train_file)
            save_jsonl(test_list, test_file)

            if upload and train_file.exists():
                response = client.files.create(
                    file=open(train_file, "rb"), purpose="fine-tune"
                )
                logger.info(f"Uploaded {train_file} to OpenAI: {response}")

            if upload and test_file.exists():
                response = client.files.create(
                    file=open(test_file, "rb"), purpose="fine-tune"
                )
                logger.info(f"Uploaded {test_file} to OpenAI: {response}")
    else:
        temp_output = output_file.parent.joinpath(f"{output_file.stem}_*.jsonl")
        output_file_list = list(temp_output.parent.glob(f"{temp_output.stem}"))
        if any(f.exists() for f in output_file_list):
            logger.warning(f"Output file already exists: {temp_output}")
            click.confirm("Do you want to overwrite?", abort=True)

        # get random sample of 10k
        if n_sample is not None:
            df = df.sample(n=10000, random_state=random_seed).reset_index(drop=True)

        # format data for inference
        df["template"] = df.apply(format_data_inference, axis=1)
        if df["template"].isna().sum() > 0:
            logger.error("Error formatting data.")
            raise ValueError("Error formatting data.")

        # save jsonl
        if not dry_run:
            output_list = df["template"].tolist()

            # save in batches of <50k (max for openai batch processing)
            output_file_list = []
            for i in range(0, len(output_list), 1000):
                idx_start = i
                idx_end = min(i + 1000, len(output_list))
                temp_output_file = output_file.parent.joinpath(
                    f"{output_file.stem}_{idx_start}-{idx_end}_gpt4o-mini-base.jsonl"
                )
                output_file_list.append(temp_output_file)
                save_jsonl(output_list[idx_start:idx_end], temp_output_file)

            for temp_output_file in output_file_list:
                if upload and temp_output_file.exists():
                    batch_input_file = client.files.create(
                        file=open(temp_output_file, "rb"), purpose="batch"
                    )
                    logger.info(
                        f"Uploaded {temp_output_file} to OpenAI: {batch_input_file}"
                    )

                    client.batches.create(
                        input_file_id=batch_input_file.id,
                        endpoint="/v1/chat/completions",
                        completion_window="24h",
                        metadata={
                            "description": "Bloom's taxonomy classification",
                        },
                    )


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    main()
