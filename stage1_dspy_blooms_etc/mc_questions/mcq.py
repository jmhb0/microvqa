#!/usr/bin/env python3
"""mcq.py in src/microchat/mc_questions.

This module defines the MCQ model using Pydantic to interact with DSPy examples and modules.
It provides methods for prediction, cleaning answers, and extracting multiple-choice options.
"""

import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any, Tuple
from loguru import logger
import dspy
import tiktoken as tk

from microchat import PROJECT_ROOT
from microchat.metrics.token_metrics import compute_token_metric

from microchat.models.base_signatures import SelfAssessBlooms, CheckSimilar, CheckFlaws
from microchat.models.dspy_modules import re_blooms_compiled, blooms_dict, CoTRAG
from microchat.models.model_factory import create_model
from microchat.utils.process_text import process_blooms, compute_tokens, compute_chars

re_clean_text = re.compile(r"`{2,3}|'{2,3}", re.IGNORECASE)
re_correct_ans = re.compile(r"\(Correct\)$", re.IGNORECASE)
re_true = re.compile(r"(True|Yes|Correct|Right)", re.IGNORECASE)
re_false = re.compile(r"(False|No|Incorrect|Wrong)", re.IGNORECASE)
# re_correct_incorrect = re.compile(r"\((Correct)\)|\((Incorrect)\)", re.IGNORECASE)
re_clean_opt = re.compile(
    r"^\d+\.\s|^\w{1}\.\s|\(Correct\)$|\(Incorrect\)$", re.IGNORECASE
)
re_remove_letter = re.compile(r"^[A-G]\)\s", re.IGNORECASE)
re_parse_example = re.compile(
    r"Description of image preparation:\n(?P<quote1>['`]{2,3})(?P<description>.*?)(?P=quote1)\n+"
    r"(Additional information:\n(?P<quote2>['`]{2,3})(?P<additional_info>.*?)(?P=quote2)\n+)?"
    r"Question:\n(?P<quote3>['`]{2,3})(?P<question>.*?)(?P=quote3)\n+"
    r"Answer:\n(?P<quote4>['`]{2,3})(?P<correct_answer>.*?)(?P=quote4)",
    re.IGNORECASE | re.DOTALL,
)

re_parse_options = re.compile(
    r"\n?\*?\*?(?P<option_label>[A-G])\)\s?\*?\*?\s?(?P<option_text>.*?)(?:\s{2,}|\n+)"
)

re_parse_question = re.compile(
    r"(?:\*\*?(?:Revised\s+)?Question:\*\*\n+)"  # Matches "Question" label with optional "Revised"
    r"['`]{0,3}(?P<question>.*?)['`]{0,3}"  # Captures the question within triple backticks
    r"\n+\n?\*\*Answer Choices:\*\*\n"  # Matches "Answer Choices" label
    r"(?P<choices>(?:[A-E]\) .+?(?:\s+\n|$))+)",  # Captures choices A-D with double spaces at line end
    # r"\n+\*?\*?Correct answer:\*?\*?\s?(?P<correct>[A-D]\) .+)",  # Matches "Correct answer" label and text
    re.IGNORECASE | re.DOTALL,  # Enables multi-line matching for `?P<question>`
)

re_parse_answer = re.compile(
    r".*\n+\s*\**(?:Correct\s+)?Answer:\**\s*(?P<correct_option>\(?[A-E]\)?[.)]?)?\s*(?P<correct_answer>.*)",
    re.IGNORECASE | re.DOTALL,
)

re_parse_prediction = re.compile(
    r"(?=Question:\n+\n?)?['`]{0,3}(?P<question>.*?)(?=\n\nA\))"  # Capture the question up to 'A)' marking the first option
    r"\n+A\)\s?(?P<option_a>.*?)(?:\s{2,}|\n+)"  # Capture option A with flexible whitespace handling
    r"B\)\s?(?P<option_b>.*?)(?:\s{2,}|\n+)"  # Capture option B
    r"C\)\s?(?P<option_c>.*?)(?:\s{2,}|\n+)"  # Capture option C
    r"D\)\s?(?P<option_d>.*?)(?:\s{2,}|\n+)"  # Capture option D
    r"\*?E\)\*?\s?(?P<option_e>.*?)(?:\s{2,}|\n+)"  # Capture option E
    r"\*?F\)\*?\s?(?P<option_f>.*?)(?:\s{2,}|\n+)"  # Capture option F
    r"\*?G\)\*?\s?(?P<option_g>.*?)(?:\s{2,}|\n+)"  # Capture option G
    r"(?:\*?H\)\*?\s?(?P<option_h>.*?)(?:\s{2,}|\n+))?"  # Optional capture for option H
    r"\n+\**([Cc]orrect\s)?[Aa]nswer:\**\s?(?P<correct_option>\(?[A-Ga-g])\)\s?(?P<correct_answer>.*)['`]{0,3}?",  # Capture correct answer with flexible capitalization
    re.IGNORECASE | re.DOTALL,  # Allows matching across multiple lines
)
re_parse_prediction_2 = re.compile(
    r"(?:\*\*Question:\*\*\n)?(?P<question>.*?)(?=\n\n\*?\**A\))"  # Capture question up to 'A)'
    r"\n+\*?\**A\)\*?\**\s?(?P<option_a>.*?)(?:\s{2,}|\n+)"  # Capture option A with flexible whitespace
    r"\*?\**B\)\*?\**\s?(?P<option_b>.*?)(?:\s{2,}|\n+)"  # Capture option B
    r"\*?\**C\)\*?\**\s?(?P<option_c>.*?)(?:\s{2,}|\n+)"  # Capture option C
    r"\*?\**D\)\*?\**\s?(?P<option_d>.*?)(?:\s{2,}|\n+)"  # Capture option D
    r"\*?\**E\)\*?\**\s?(?P<option_e>.*?)(?:\s{2,}|\n+)"  # Capture option E
    r"\*?\**F\)\*?\**\s?(?P<option_f>.*?)(?:\s{2,}|\n+)"  # Capture option F
    r"\*?\**G\)\*?\**\s?(?P<option_g>.*?)(?:\s{2,}|\n+)"  # Capture option G
    r"\n+\*?\**([Cc]orrect\s)?[Aa]nswer:\*?\**\s?(?P<correct_option>[A-Ga-g])\)?\s?(?P<correct_answer>.*)",  # Capture "Correct answer" and answer option
    re.IGNORECASE | re.DOTALL,  # Allows multi-line matching and case insensitivity
)
re_parse_prediction_3 = re.compile(
    r"(?:\*\*)?(?P<question>.*?)(?=\n\n\*?\**A\))"  # Capture question up to 'A)'
    r"\n+\*?\**A\)\*?\**\s?(?P<option_a>.*?)(?:\s{2,}|\n+)"  # Capture option A with flexible whitespace
    r"\*?\**B\)\*?\**\s?(?P<option_b>.*?)(?:\s{2,}|\n+)"  # Capture option B
    r"\*?\**C\)\*?\**\s?(?P<option_c>.*?)(?:\s{2,}|\n+)"  # Capture option C
    r"\*?\**D\)\*?\**\s?(?P<option_d>.*?)(?:\s{2,}|\n+)"  # Capture option D
    r"\*?\**E\)\*?\**\s?(?P<option_e>.*?)(?:\s{2,}|\n+)"  # Capture option E
    r"\*?\**F\)\*?\**\s?(?P<option_f>.*?)(?:\s{2,}|\n+)"  # Capture option F
    r"\*?\**G\)\*?\**\s?(?P<option_g>.*?)(?:\s{2,}|\n+)"  # Capture option G
    r"\n+\*?\**([Cc]orrect\s)?[Aa]nswer:\*?\**\s?(?P<correct_option>[A-Ga-g])\)?\s?(?P<correct_answer>.*)",  # Capture "Correct answer" and answer option
    re.IGNORECASE | re.DOTALL,  # Allows multi-line matching and case insensitivity
)
re_parse_prediction_4 = re.compile(
    r"(?:\*?\*?Revised\s+)?Question(?:\s?\d?)?:\s*['`]{0,3}(?P<question>.*?)(?=\n\n\*?A\))"
    r"\n+\*?A\)\*?\s?(?P<option_a>.*?)(?:\s{2,}|\n+)"  # Capture option A
    r"\*?B\)\*?\s?(?P<option_b>.*?)(?:\s{2,}|\n+)"  # Capture option B
    r"\*?C\)\*?\s?(?P<option_c>.*?)(?:\s{2,}|\n+)"  # Capture option C
    r"\*?D\)\*?\s?(?P<option_d>.*?)(?:\s{2,}|\n+)"  # Capture option D
    r"\*?E\)\*?\s?(?P<option_e>.*?)(?:\s{2,}|\n+)"  # Capture option E
    r"\*?F\)\*?\s?(?P<option_f>.*?)(?:\s{2,}|\n+)"  # Capture option F
    r"\*?G\)\*?\s?(?P<option_g>.*?)(?:\s{2,}|\n+)"  # Capture option G
    r"(?:\*?H\)\*?\s?(?P<option_h>.*?)(?:\s{2,}|\n+))?"  # Optional capture for option H
    r"\n+\*?(?:Revised\s)?[Cc]orrect\s[Aa]nswer:\*?\s?(?P<correct_option>[A-E])\)\s?(?P<correct_answer>.*)['`]{0,3}",
    re.IGNORECASE | re.DOTALL,
)
re_parse_prediction_5 = re.compile(
    r"(?:\*?\*?Revised\s+)?Question(?:\s?\d?)?:(?:\*?\*?)?\s*\n+\n*['`]{0,3}(?P<question>.*?)['`]{0,3}"  # Capture "Revised Question" with optional asterisks and backticks
    r"\n+\n*(?:\*?\*?Revised\s+)?Answer(?:\s?\d?)?:(?:\*?\*?)?\s*\n+\n*['`]{0,3}(?P<correct_answer>.*?)['`]{0,3}",  # Capture "Revised Answer" with optional asterisks and backticks
    re.IGNORECASE | re.DOTALL,  # Allows matching across multiple lines
)
re_parse_prediction_6 = re.compile(
    r"(?:Question:\s*)?(?P<question>.*?)(?=\n\n(Revised|Correct)\s?Answer:)"  # Optional "Question:" prefix and captures question up to "Answer" prefix
    r"\n\n(?:Revised|Correct)\s?Answer:\s*\n?(?P<correct_answer>.*)",  # Matches "Correct Answer:" or "Revised Answer:" followed by answer text
    re.IGNORECASE
    | re.DOTALL,  # Allows case-insensitive matching and multi-line capture
)
re_parse_prediction_7 = re.compile(
    r"(?<=\*\*Question:\*\*\n\n)?(?P<question>.*?)(?=\n\n(?:[Nn]o|[Yy]es|[Aa]\)))"  # Capture question up to an answer indicator (No, Yes, or A))
    r"\n\n(?:[Nn]o|[Yy]es|A\))\s?(?P<correct_answer>.*)",  # Match answer prefix (No, Yes, or A)) and capture answer text
    re.IGNORECASE | re.DOTALL,  # Allows case-insensitive and multi-line matching
)


DEFAULT_TEACHER = create_model("o1-mini")  # "o1-preview"


class MCQ(BaseModel):
    """
    Pydantic model for handling multiple-choice question processing.
    It accepts an example and a DSPy module to make predictions.
    """

    example: dspy.Example
    prediction: dspy.Example
    tokenizer: Optional[tk.Encoding] = None

    # variables set from processing the example
    tokenizer_name: Optional[str] = Field(None, alias="tokenizer_name")  # set by model
    example_dict: Optional[Dict[str, str]] = {}
    prediction_dict: Optional[Dict[str, Any]] = {}
    metrics: Optional[Dict[str, Any]] = {}
    errors: Optional[int] = 0

    class Config:
        arbitrary_types_allowed = True

    @field_validator("example")
    def validate_example(cls, value):
        if not isinstance(value, dspy.Example):
            raise ValueError("example must be a DSPy Example instance.")
        return value

    @field_validator("tokenizer")
    def validate_tokenizer(cls, value):
        if not isinstance(value, tk.Encoding):
            raise ValueError("tokenizer must be a TikToken Encoding instance.")
        return value

    @staticmethod
    def compute_chars(prompt: str) -> int:
        """Compute the number of characters in a prompt."""
        return compute_chars(prompt)

    @staticmethod
    def compute_tokens(prompt: str, tokenizer: tk.Encoding) -> int:
        """Compute the number of tokens in a prompt."""
        return compute_tokens(prompt, tokenizer)

    def __repr__(self) -> str:
        """An unambiguous string representation of the class instance."""
        return f"MCQ(example={self.example}, prediction={self.prediction})"

    def __str__(self) -> str:
        """An easy-to-read string representation of the dataset class."""
        question = self.example_dict.get("question", "")
        answer = self.example_dict.get("correct_answer", "")
        revised_question = self.prediction_dict.get("question", "")
        revised_answer = self.prediction_dict.get("correct_answer", "")
        output_str = [
            f"Original question: {question}\nOriginal answer: {answer}\n\n",
            f"----------------------------------------\n",
            f"Revised question: {revised_question}Revised answer: {revised_answer}\n",
            f"----------------------------------------\n",
            f"Metrics: {self.metrics}",
        ]

        return "".join(output_str)

    def get_tokens(self, original: bool, field: str) -> int:
        if original:
            return self.compute_tokens(self.example_dict.get(field, ""), self.tokenizer)
        else:
            return self.compute_tokens(
                self.prediction_dict.get(field, ""), self.tokenizer
            )

    def compute_metrics(
        self, question_key: str = "question", answer_key: str = "correct_answer"
    ) -> Dict[str, float]:
        temp_example = dspy.Example(
            question=self.example_dict.get(question_key),
            answer=self.example_dict.get(answer_key),
        )
        temp_pred = dspy.Example(
            question=self.prediction_dict.get(question_key),
            answer=self.prediction_dict.get(answer_key),
        )

        if match := False:
            return {
                "similarity": float(match),
                "formatted": float(match),
                "extraneous": 1 - float(match),
                "reasoning": None,
            }

        # check for semantic similarity of the question
        context = self.example_dict.get("description")
        if addtl_info := self.example_dict.get("additional_info"):
            context += f"\n\nAdditional information: {addtl_info}"

        question_str = (
            f"Original question: {temp_example.question}\nOriginal answer: {temp_example.answer}\n\n"
            f"----------------------------------------\n"
            f"Revised question: {temp_pred.question}Revised answer: {temp_pred.answer}\n"
        )
        self_assess_str = (
            f"{self.example.question}\n"
            "----------------------------------------\n"
            "Revised:\n"
            f"{self.prediction.answer}"
        )
        #### Original process check similarity
        # with dspy.settings.context(lm=DEFAULT_TEACHER.lm):
        result = dspy.ChainOfThought(CheckSimilar)(
            context=context, question=self_assess_str  # question_str
        )

        # clean text outputs
        similarity = re_clean_text.sub("", result.similarity).strip()
        similarity = re_true.sub("1", similarity)
        formatted = re_clean_text.sub("", result.formatted).strip()
        formatted = re_true.sub("1", formatted)
        extraneous = re_clean_text.sub("", result.extraneous).strip()
        extraneous = re_true.sub("1", extraneous)

        try:
            similarity = float(eval(similarity))
            formatted = float(eval(formatted))
            extraneous = float(eval(extraneous))
        except ValueError as e:
            logger.error(f"Error in converting metrics to float: {e}")
            similarity = 0
            formatted = 0
            extraneous = 0

        return {
            "similarity": float(similarity),
            "formatted": float(formatted),
            "extraneous": 1 - float(extraneous),
            "reasoning": result.reasoning,
        }

        ### Separate processing check flaws
        # # check flaws
        # check_flaws = dspy.ChainOfThought(CheckFlaws)(
        #     context=context, question=f"Revised question: {temp_pred.question}Revised answer: {temp_pred.answer}\n"
        # )
        # # clean text outputs
        # nbme_formatted = re_clean_text.sub("", check_flaws.nbme_formatted).strip()
        # nbme_formatted = re_true.sub("1", nbme_formatted)
        # question_flaws = re_clean_text.sub("", check_flaws.question_flaws).strip()
        # question_flaws = re_true.sub("1", question_flaws)
        # answer_flaws = re_clean_text.sub("", check_flaws.answer_flaws).strip()
        # answer_flaws = re_true.sub("1", answer_flaws)
        # distractor_flaws = re_clean_text.sub("", check_flaws.distractor_flaws).strip()
        # distractor_flaws = re_true.sub("1", distractor_flaws)
        #
        # try:
        #     nbme_formatted = float(eval(nbme_formatted))
        #     question_flaws = float(eval(question_flaws))
        #     answer_flaws = float(eval(answer_flaws))
        #     distractor_flaws = float(eval(distractor_flaws))
        # except ValueError as e:
        #     logger.error(f"Error in converting metrics to float: {e}")
        #     nbme_formatted = 0
        #     question_flaws = 0
        #     answer_flaws = 0
        #     distractor_flaws = 0
        #
        # return {
        #     "nbme_formatted": float(nbme_formatted),
        #     "question_flaws": float(question_flaws),
        #     "answer_flaws": float(answer_flaws),
        #     "distractor_flaws": 1 - float(distractor_flaws),
        #     "reasoning": check_flaws.reasoning,
        # }

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook for the MCQ model.
        """
        lm_name = getattr(dspy.settings.lm, "model_name", None)
        lm_prefix = getattr(dspy.settings.lm, "model_prefix", None)
        if self.tokenizer is None:
            # set tokenizer name
            if "o1" in lm_prefix:
                self.tokenizer_name: str = "o200k_base"
            else:
                self.tokenizer_name: str = tk.encoding_name_for_model(lm_prefix)

            self.tokenizer = tk.get_encoding(self.tokenizer_name)

        # process the example
        # temp hack
        example_match = re_parse_example.search(self.example.question)
        # example_match = re_parse_prediction.search(self.example.question)
        if example_match is None:
            self.errors += 1
            logger.warning("Example does not match the expected format.")
        else:
            self.example_dict = example_match.groupdict()

        # process the prediction
        pred_match = None
        for idx, compiled_regex in enumerate(
            [
                re_parse_prediction_4,
                re_parse_prediction,
                re_parse_prediction_2,
                re_parse_prediction_3,
                re_parse_prediction_5,
                re_parse_prediction_6,
                re_parse_prediction_7,
            ]
        ):
            pred_match = compiled_regex.search(self.prediction.answer)
            if pred_match:
                break
            pred_match = compiled_regex.search(self.prediction.answer.strip("*"))

        if pred_match is None:
            if question := re_parse_question.search(self.prediction.answer):
                self.prediction_dict["question"] = question.group("question")
                options_1 = re_parse_options.findall(question.group("choices"))

            if answer := re_parse_answer.search(self.prediction.answer):
                self.prediction_dict["correct_answer"] = answer.group("correct_answer")
                self.prediction_dict["correct_option"] = answer.group("correct_option")

            self.errors += 1
            logger.warning("Prediction does not match the expected format.")
        else:
            prediction_dict = pred_match.groupdict()
            # strip leading \* from the question
            for key in ["question", "correct_answer"]:
                if prediction_dict.get(key):
                    prediction_dict[key] = prediction_dict[key].strip("*").strip()
            # clean_text
            for key in ["question", "correct_answer"]:
                if prediction_dict.get(key):
                    prediction_dict[key] = re_clean_text.sub("", prediction_dict[key])
            self.prediction_dict = prediction_dict

        # extract the question, options, and correct answer from the prediction
        pred_question = self.prediction_dict.get("question")
        pred_options = [
            self.prediction_dict.get(f"option_{char}")
            for char in "abcdefg"
            if self.prediction_dict.get(f"option_{char}")
        ]
        if not pred_options:
            # logger.warning("No options found in the prediction.")
            pred_options = re_parse_options.findall(self.prediction.answer)
            pred_options = [elem[1] for elem in pred_options]
            pred_answer = re_parse_answer.match(self.prediction.answer)
            if pred_answer is not None:
                pred_answer = re_clean_text.sub(
                    "", pred_answer.group("correct_answer")
                ).strip()
            else:
                logger.warning("Correct answer does not match the expected format.")
                self.errors += 1

            if self.prediction_dict.get("correct_answer").strip() != pred_answer:
                if (
                    re_remove_letter.sub(
                        "", self.prediction_dict.get("correct_answer").strip()
                    )
                    == pred_answer
                ):
                    self.prediction_dict["correct_answer"] = pred_answer
                else:
                    logger.warning("Correct answer does not match the expected format.")
                    self.errors += 1

        if not pred_options:
            logger.warning("No options found in the prediction.")
            self.errors += 1

        self.prediction_dict["options"] = pred_options
        pred_answer = self.prediction_dict.get("correct_answer")
        if pred_answer:
            # pred_answer = re_clean_text.sub("", pred_answer).strip()
            pred_answer = re_clean_opt.sub("", pred_answer)
            pred_answer = re_remove_letter.sub("", pred_answer)

        # if not hasattr(self.prediction_dict, "correct_option"):
        #     self.prediction_dict.get("correct_option")

        pred_option_correct = self.prediction_dict.get("correct_option")
        pred_correct_index = -1
        if pred_answer in pred_options:
            pred_correct_index = pred_options.index(pred_answer)
            self.prediction_dict["correct_index"] = pred_correct_index
        elif pred_option_correct:
            pred_correct_index = ord(pred_option_correct.lower()) - ord("a")
            pred_answer = pred_options[pred_correct_index]
            self.prediction_dict["correct_answer"] = pred_answer
            self.prediction_dict["correct_index"] = pred_correct_index
        else:
            logger.warning(f"Correct answer not found in options: {pred_answer}")
            raise ValueError("Correct answer not found in options.")

        # check if pred_answer tokens are longer than the original answer
        ans_token_metric = 1
        if self.example_dict.get("correct_answer") and pred_answer:
            orig_tokens = self.compute_tokens(
                self.example_dict.get("correct_answer"), self.tokenizer
            )
            pred_tokens = self.compute_tokens(pred_answer, self.tokenizer)
            ans_token_ratio = pred_tokens / orig_tokens
            ans_token_metric = compute_token_metric(orig_tokens, pred_tokens, k=0.5)

            if ans_token_ratio >= 4:
                logger.error(
                    f"Predicted answer too long: {orig_tokens} vs. {pred_tokens}"
                )
                logger.error(
                    f"Original answer: {self.example_dict.get('correct_answer')}"
                )
                logger.error(f"Predicted answer: {pred_answer}")
            elif ans_token_ratio >= 1.5:
                logger.warning(
                    f"Predicted answer longer: {orig_tokens} vs. {pred_tokens}"
                )
                logger.warning(
                    f"Original answer: {self.example_dict.get('correct_answer')}"
                )
                logger.warning(f"Predicted answer: {pred_answer}")

        # compare token difference example and prediction
        token_diff = {}
        for key in ["question", "correct_answer"]:
            if key in self.example_dict and key in self.prediction_dict:
                example_tokens = self.compute_tokens(
                    self.example_dict.get(key, ""), self.tokenizer
                )
                pred_tokens = self.compute_tokens(
                    self.prediction_dict.get(key, ""), self.tokenizer
                )
                token_diff[key] = abs(example_tokens - pred_tokens)

        # compute the number of tokens in the options and correct answer
        option_token_ratio = 1  # if no pred_options, don't penalize the model
        if pred_options:
            option_tokens = [
                self.compute_tokens(option, self.tokenizer) for option in pred_options
            ]
            correct_tokens = option_tokens[pred_correct_index] or 1
            incorrect_tokens = [
                tokens
                for i, tokens in enumerate(option_tokens)
                if i != pred_correct_index
            ]
            token_ratio = np.divide(np.array(incorrect_tokens), correct_tokens)

            # compute metric for token ratio, want to have ratio near 1
            option_token_ratio = np.mean(token_ratio)

        try:
            self.metrics = self.compute_metrics(
                question_key="question", answer_key="correct_answer"
            )
        except Exception:
            logger.warning("Error in computing metrics:")
            logger.warning(f"Example: {self.example}")
            logger.warning(f"Prediction: {self.prediction}")
            self.metrics = {
                "similarity": 0,
                "formatted": 0,
                "extraneous": 0,
                "reasoning": None,
            }

        self.metrics["option_token_ratio"] = option_token_ratio
        self.metrics["answer_token_metric"] = ans_token_metric
        self.metrics["errors"] = 1 - (self.errors / 2)
        self.metrics["reasoning"] = self.metrics.get("reasoning")


class Blooms(BaseModel):
    """
    Pydantic model for handling multiple-choice question processing.
    It accepts an example and a DSPy module to make predictions.
    """

    example: dspy.Example
    module: Any
    tokenizer: Optional[tk.Encoding] = None
    teacher_model: Optional[dspy.LM] = None

    #
    gt_level: Optional[int] = None
    gt_bloom: Optional[str] = None
    blooms_level: Optional[int] = None
    blooms_name: Optional[str] = None
    blooms_verb: Optional[str] = None
    blooms_confidence: Optional[float] = None
    blooms_source: Optional[str] = None
    blooms_reasoning: Optional[str] = None
    context: Optional[List[str]] = None
    self_check_question: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @field_validator("example")
    def validate_example(cls, value):
        if not isinstance(value, dspy.Example):
            raise ValueError("example must be a DSPy Example instance.")
        return value

    @field_validator("module")
    def validate_module(cls, value):
        if not isinstance(value, dspy.Module):
            raise ValueError("module must be a DSPy Module instance.")
        return value

    @field_validator("tokenizer")
    def validate_tokenizer(cls, value):
        if not isinstance(value, tk.Encoding):
            raise ValueError("tokenizer must be a TikToken Encoding instance.")
        return value

    @staticmethod
    def compute_chars(prompt: str) -> int:
        """Compute the number of characters in a prompt."""
        return compute_chars(prompt)

    @staticmethod
    def compute_tokens(prompt: str, tokenizer: tk.Encoding) -> int:
        """Compute the number of tokens in a prompt."""
        return compute_tokens(prompt, tokenizer)

    @staticmethod
    def _process_answer(
        answer: str, reference_dict: Optional[dict] = blooms_dict
    ) -> Tuple[int, str, str]:
        return process_blooms(answer, reference_dict)

    def predict(self, return_reasoning: bool = False) -> dspy.Prediction:
        """
        Predict the answer using the DSPy module.
        """
        # if os.getenv("DEBUG", False):
        #     logger.debug(f"Predicting answer for question: {self.example.question}")

        try:
            response = self.module(
                self.example.question, return_reasoning=return_reasoning
            )
            if response is None:
                logger.error(f"Prediction failed for question: {self.example.question}")

            if getattr(response, "reasoning", None) is None:
                response.reasoning = "No reasoning provided."
            else:
                response.reasoning = response.reasoning

            return response
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise e

    def __repr__(self) -> str:
        """An unambiguous string representation of the class instance."""
        return f"Blooms(\n\texample={self.example},\n\tmodule={self.module}\n)"

    def __str__(self) -> str:
        """An easy-to-read string representation of the dataset class."""
        return f"{self.example.question}\nBloom's: {self.blooms_name.capitalize()} (level {self.blooms_level})"

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook for the MCQ model.
        """
        # process GT example to get the ground truth blooms name and level
        if isinstance(self.example.blooms_source, str):
            gt_model = self.example.blooms_source.split("&")  # "CustomGPT"
        else:
            gt_model = ["CustomGPT"]

        gt_model = [elem.strip() for elem in gt_model]
        gt_level, gt_bloom, gt_bloom_verb = self._process_answer(
            self.example.answer, blooms_dict
        )
        gt_level = gt_level if gt_level > 0 else self.example.blooms_level
        gt_bloom = gt_bloom or self.example.answer

        self.gt_level = gt_level
        self.gt_bloom = gt_bloom
        # predict blooms the example
        response = self.predict(return_reasoning=True)

        # set the context for the response
        # note that the context is manually set to be a curated list of Bloom's
        # or NBME reference information.
        if getattr(response, "context", None) is None:
            self.context = getattr(dspy.settings, "context", None)

        # process the response
        init_model = dspy.settings.lm.model_name
        init_level, init_bloom, init_verb = self._process_answer(
            response.answer, blooms_dict
        )
        # correct for null
        init_level = init_level or "Unknown"
        init_bloom = init_bloom or response.answer

        # set the self-check question
        self.self_check_question = (
            "Multiple LLM models evaluated the Bloom's Taxonomy level of the following  multiple choice question:\n"
            f"{self.example.question}\n"
            f"One model predicted '{gt_bloom.capitalize()}' (Level {gt_level}) with the following reasoning: {self.example.blooms_reasoning}\n"
            f"A second model predicted '{init_bloom.capitalize()}' (Level {init_level}) with the following reasoning: {response.reasoning}\n\n"
            "# Independent Assessment of Bloom's Taxonomy Level\n"
            "Provide an independent assessment of the most appropriate Bloom's Taxonomy level for the question below. Explain whether you agree or disagree with previous predictions, and why."
            "When evaluating between Comprehension (Level 2) and Application (Level 3) or higher levels, consider:"
            "  - Does the question involve straightforward identification without broader context?"
            "  - Does it require advanced or non-obvious identification?"
            "  - Does it involve application or connection to a broader context?"
            "  - Is it applied to a new or challenging setting?"
            "  - Does it require deep understanding or drawing conclusions that are not obvious?\n"
            f"{self.example.question}\n"
            f"Bloom's:"
        )
        if self.teacher_model is None:
            # model is LLModel class with dspy.LM in model.lm
            self.teacher_model = create_model("o1-mini").lm

        # change context manager to allow self-assessment by a teacher model
        # logger.debug(f"Original model: {dspy.settings.lm.model_name}")
        model_dir = Path(PROJECT_ROOT).joinpath("models/dspy_compiled/blooms/")
        with dspy.settings.context(lm=self.teacher_model):
            # logger.debug(f"Model: {dspy.settings.lm.model_name}")
            rev_model = dspy.settings.lm.model_name
            teacher_module = CoTRAG(
                context="blooms", return_reasoning=True
            )  # context sets signature
            teacher_module.name = teacher_module.__class__.__name__
            model_filepath = model_dir.joinpath(
                f"{self.teacher_model.model_name}_{teacher_module.name}_{teacher_module.signature_name}.json"
            )
            if model_filepath.exists():
                teacher_module.load(model_filepath)
                teacher_module.name = teacher_module.__class__.__name__
            else:
                logger.error(f"Model file not found: {model_filepath}")

            assess_response = teacher_module(
                question=self.self_check_question  # , context=self.context
            )

        # process
        rev_level, rev_bloom, rev_verb = self._process_answer(
            assess_response.answer, blooms_dict
        )
        # correct for null
        rev_level = rev_level or "Unknown"
        rev_bloom = rev_bloom or assess_response.answer

        #
        if gt_level == init_level and gt_level == rev_level:
            # best case scenario, all cases match, use the ground truth level
            self.blooms_level = gt_level
            self.blooms_name = gt_bloom
            self.blooms_confidence = 1.0
            self.blooms_source = " & ".join(gt_model + [init_model, rev_model])
            self.blooms_reasoning = assess_response.reasoning
            self.blooms_verb = gt_bloom_verb
        elif gt_level == init_level:
            # if the ground truth and initial levels match, use the ground truth level
            self.blooms_level = gt_level
            self.blooms_name = gt_bloom
            self.blooms_confidence = 2 / 3
            self.blooms_source = " & ".join(gt_model + [init_model])
            self.blooms_reasoning = response.reasoning
            self.blooms_verb = gt_bloom_verb
        elif gt_level == rev_level:
            # if the ground truth and self-assessment levels match, use the ground truth level
            self.blooms_level = gt_level
            self.blooms_name = gt_bloom
            self.blooms_confidence = 2 / 3
            self.blooms_source = " & ".join(gt_model + [rev_model])
            self.blooms_reasoning = assess_response.reasoning
            self.blooms_verb = gt_bloom_verb
        elif rev_level == init_level and abs(gt_level - rev_level) == 1:
            # if the self-assessment and initial levels match, use the self-assessment level
            # if the ground truth level is one level higher or lower
            self.blooms_level = init_level
            self.blooms_name = init_bloom
            self.blooms_confidence = 2 / 3
            self.blooms_source = " & ".join([init_model, rev_model])
            self.blooms_reasoning = assess_response.reasoning
            self.blooms_verb = init_verb
        else:
            tiebreak_question = (
                "Multiple LLM models evaluated the Bloom's Taxonomy level of the following  multiple choice question:\n"
                f"{self.example.question}\n"
                f"One model predicted '{gt_bloom.capitalize()}' (Level {gt_level}) with the following reasoning: {self.example.blooms_reasoning}\n"
                f"A second model predicted '{init_bloom.capitalize()}' (Level {init_level}) with the following reasoning: {response.reasoning}\n\n"
                f"A third model predicted '{rev_bloom.capitalize()}' (Level {rev_level}) with the following reasoning: {assess_response.reasoning}\n\n"
                "# Independent Assessment of Bloom's Taxonomy Level\n"
                "Provide an independent assessment of the most appropriate Bloom's Taxonomy level for the question below. Explain whether you agree or disagree with previous predictions, and why."
                "When evaluating between Comprehension (Level 2) and Application (Level 3) or higher levels, consider:"
                "  - Does the question involve straightforward identification without broader context?"
                "  - Does it require advanced or non-obvious identification?"
                "  - Does it involve application or connection to a broader context?"
                "  - Is it applied to a new or challenging setting?"
                "  - Does it require deep understanding or drawing conclusions that are not obvious?\n"
                f"{self.example.question}\n"
                f"Bloom's:"
            )
            # o1-preview judge
            oracle_model = create_model("o1-preview").lm

            # change context manager to allow self-assessment by a teacher model
            logger.info(
                f"Mismatched Bloom's predictions:\nGT:\t\t{gt_level}\n{dspy.settings.lm.model_name}:\t{init_level}\n{rev_model}:\t{rev_level}"
            )
            logger.info(f"Tiebreaker model: {oracle_model.model_name}")
            with dspy.settings.context(lm=oracle_model):
                # logger.debug(f"Model: {dspy.settings.lm.model_name}")
                oracle_module = CoTRAG(context="blooms", return_reasoning=True)
                oracle_module.name = teacher_module.__class__.__name__

                final_response = oracle_module(question=tiebreak_question)

            # process
            final_level, final_bloom, final_verb = self._process_answer(
                final_response.answer, blooms_dict
            )

            if final_level == gt_level:
                self.blooms_level = final_level
                self.blooms_name = final_bloom
                self.blooms_confidence = 2 / 4
                self.blooms_source = " & ".join(gt_model + [oracle_model.model_name])
                self.blooms_reasoning = final_response.reasoning
                self.blooms_verb = final_verb
            elif final_level == init_level:
                self.blooms_level = final_level
                self.blooms_name = final_bloom
                self.blooms_confidence = 2 / 4
                self.blooms_source = " & ".join([init_model, oracle_model.model_name])
                self.blooms_reasoning = final_response.reasoning
                self.blooms_verb = final_verb
            elif final_level == rev_level:
                self.blooms_level = final_level
                self.blooms_name = final_bloom
                self.blooms_confidence = 2 / 4
                self.blooms_source = " & ".join([rev_model, oracle_model.model_name])
                self.blooms_reasoning = final_response.reasoning
                self.blooms_verb = final_verb
            else:
                logger.warning(
                    f"No agreement between models: {gt_model}, {init_model}, {rev_model}, {oracle_model.model_name}"
                )
                logger.warning(
                    f"{oracle_model.model_name} reasoning: {final_response.reasoning}"
                )
                self.blooms_level = final_level
                self.blooms_name = final_bloom
                self.blooms_confidence = 1 / 4
                self.blooms_source = oracle_model.model_name
                self.blooms_reasoning = f"No agreement between models: {gt_model}, {init_model}, {rev_model}, {oracle_model.model_name}"
                self.blooms_reasoning += f"\n\n{oracle_model.model_name} reasoning: {final_response.reasoning}"
                logger.info(self.blooms_reasoning)

        # rename self.blooms_name to a standard name
        rename_dict = {
            "Recall": "Recall",
            "Remember": "Recall",
            "Memorize": "Recall",
            "Knowledge": "Recall",
            "Comprehension": "Comprehension",
            "Comprehend": "Comprehension",
            "Understand": "Comprehension",
            "Apply": "Application",
            "Applying": "Application",
            "Analyze": "Analysis",
            "Analyzing": "Analysis",
            "Evaluate": "Evaluation",
            "Evaluating": "Evaluation",
            "Synthesis": "Synthesis",
            "Synthesizing": "Synthesis",
        }
        self.blooms_name = rename_dict.get(self.blooms_name, self.blooms_name)


class MCQTopic(BaseModel):
    """
    Pydantic model for handling multiple-choice question processing.
    It accepts an example and a DSPy module to make predictions.
    """

    example: dspy.Example
    module: Any
    tokenizer: Optional[tk.Encoding] = None
    teacher_model: Optional[dspy.LM] = None

    #
    topic_name: Optional[str] = None
    topic_confidence: Optional[float] = None
    context: Optional[List[str]] = None
    self_check_question: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @field_validator("example")
    def validate_example(cls, value):
        if not isinstance(value, dspy.Example):
            raise ValueError("example must be a DSPy Example instance.")
        return value

    @field_validator("module")
    def validate_module(cls, value):
        if not isinstance(value, dspy.Module):
            raise ValueError("module must be a DSPy Module instance.")
        return value

    @field_validator("tokenizer")
    def validate_tokenizer(cls, value):
        if not isinstance(value, tk.Encoding):
            raise ValueError("tokenizer must be a TikToken Encoding instance.")
        return value

    @staticmethod
    def compute_chars(prompt: str) -> int:
        """Compute the number of characters in a prompt."""
        return len(prompt)

    @staticmethod
    def compute_tokens(prompt: str, tokenizer: tk.Encoding) -> int:
        """Compute the number of tokens in a prompt."""
        return len(tokenizer.encode(prompt))

    # @staticmethod
    # def _process_answer(answer: str, reference_dict: dict) -> Tuple[int, str]:
    #     # extract the blooms level from the response
    #     topic_name = None
    #
    #     if match := re_blooms_compiled.search(answer):
    #         blooms_name = match.group().lower()
    #         # find the level of the blooms taxonomy from blooms_dict
    #         blooms_level = next(
    #             level for level, names in reference_dict.items() if blooms_name in names
    #         )
    #     else:
    #         logger.warning(f"Bloom's taxonomy level found in answer: {answer}")
    #
    #     return blooms_level, blooms_name

    def predict(self) -> dspy.Prediction:
        """
        Predict the answer using the DSPy module.
        """
        # if os.getenv("DEBUG", False):
        #     logger.debug(f"Predicting answer for question: {self.example.question}")

        try:
            return self.module(self.example.question)
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise e

    def __repr__(self) -> str:
        """An unambiguous string representation of the class instance."""
        return f"MCQTopic(\n\texample={self.example},\n\tmodule={self.module}\n)"

    def __str__(self) -> str:
        """An easy-to-read string representation of the dataset class."""
        return f"{self.example.question}\nTopic: {self.topic_name.capitalize()}"

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook for the MCQ model.
        """
        # process GT example to get the ground truth blooms name and level
        gt_model = self.example.blooms_source.split("&")  # "CustomGPT"
        gt_model = [elem.strip() for elem in gt_model]
        gt_level, gt_bloom = self._process_answer(self.example.answer, blooms_dict)
        # predict blooms the example
        response = self.predict()
        if response is None:
            logger.error(f"Prediction failed for question: {self.example.question}")

        # set the context for the response
        # note that the context is manually set to be a curated list of Bloom's
        # or NBME reference information.
        self.context = response.context

        # process the response
        init_model = dspy.settings.lm.model_name
        init_level, init_bloom = self._process_answer(response.answer, blooms_dict)
        #
        self.self_check_question = (
            "Multiple LLM models evaluated the Bloom's Taxonomy level of the following  multiple choice question:\n"
            f"{self.example.question}\n"
            f"One model predicted '{gt_bloom.capitalize()}' (Level {gt_level}) with the following reasoning: {self.example.blooms_reasoning}\n"
            f"A second model predicted '{init_bloom.capitalize()}' (Level {init_level}) with the following reasoning: {response.reasoning}\n\n"
            "# Independent Assessment of Bloom's Taxonomy Level\n"
            "Provide an independent assessment of the most appropriate Bloom's Taxonomy level for the question below. Explain whether you agree or disagree with previous predictions, and why."
            "When evaluating between Comprehension (Level 2) and Application (Level 3) or higher levels, consider:"
            "  - Does the question involve straightforward identification without broader context?"
            "  - Does it require advanced or non-obvious identification?"
            "  - Does it involve application or connection to a broader context?"
            "  - Is it applied to a new or challenging setting?"
            "  - Does it require deep understanding or drawing conclusions that are not obvious?\n"
            f"{self.example.question}\n"
            f"Bloom's:"
        )
        if self.teacher_model is None:
            # model is LLModel class with dspy.LM in model.lm
            self.teacher_model = create_model("o1-mini").lm

        # change context manager to allow self-assessment by a teacher model
        # logger.debug(f"Original model: {dspy.settings.lm.model_name}")
        with dspy.settings.context(lm=self.teacher_model):
            # logger.debug(f"Model: {dspy.settings.lm.model_name}")
            rev_model = dspy.settings.lm.model_name
            assess_response = dspy.ChainOfThought(SelfAssessBlooms)(
                question=self.self_check_question, context=response.context
            )

        # logger.debug(f"Restored model: {dspy.settings.lm.model_name}")
        rev_level, rev_bloom = self._process_answer(assess_response.answer, blooms_dict)

        #
        if gt_level == init_level and gt_level == rev_level:
            # best case scenario, all cases match, use the ground truth level
            self.blooms_level = gt_level
            self.blooms_name = gt_bloom
            self.blooms_confidence = 1.0
            self.blooms_source = " & ".join(gt_model + [init_model, rev_model])
            self.blooms_reasoning = assess_response.reasoning
        elif gt_level == init_level:
            # if the ground truth and initial levels match, use the ground truth level
            self.blooms_level = gt_level
            self.blooms_name = gt_bloom
            self.blooms_confidence = 2 / 3
            self.blooms_source = " & ".join(gt_model + [init_model])
            self.blooms_reasoning = response.reasoning
        elif gt_level == rev_level:
            # if the ground truth and self-assessment levels match, use the ground truth level
            self.blooms_level = gt_level
            self.blooms_name = gt_bloom
            self.blooms_confidence = 2 / 3
            self.blooms_source = " & ".join(gt_model + [rev_model])
            self.blooms_reasoning = assess_response.reasoning
        elif rev_level == init_level:
            # if the self-assessment and initial levels match, use the self-assessment level
            self.blooms_level = init_level
            self.blooms_name = init_bloom
            self.blooms_confidence = 2 / 3
            self.blooms_source = " & ".join([init_model, rev_model])
            self.blooms_reasoning = assess_response.reasoning
        else:
            # if none of the levels match, use the ground truth level
            logger.error(
                f"Ground truth, initial prediction, and self-assessment levels do not match. {gt_level} != {init_level} != {rev_level}"
            )
            self.blooms_confidence = 0

        # rename self.blooms_name to a standard name
        rename_dict = {
            "Recall": "Recall",
            "Remember": "Recall",
            "Memorize": "Recall",
            "Knowledge": "Recall",
            "Comprehension": "Comprehension",
            "Comprehend": "Comprehension",
            "Understand": "Comprehension",
            "Apply": "Application",
            "Applying": "Application",
            "Analyze": "Analysis",
            "Analyzing": "Analysis",
            "Evaluate": "Evaluation",
            "Evaluating": "Evaluation",
            "Synthesis": "Synthesis",
            "Synthesizing": "Synthesis",
        }
        self.blooms_name = rename_dict.get(self.blooms_name, self.blooms_name)
