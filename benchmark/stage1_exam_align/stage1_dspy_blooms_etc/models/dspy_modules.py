#!/usr/bin/env python3
"""dspy_modules.py in src/microchat/models."""

from pathlib import Path
from random import shuffle
from typing import List, Optional, Any

from loguru import logger

import dspy
import re
from pprint import pprint

from microchat import MODULE_ROOT

from microchat.fileio.text.readers import yaml_loader
from microchat.models.base_signatures import (
    ReviseInput,
    ClassifyBlooms,
    GenerateSearchQuery,
    SelfAssessBlooms,
    DefaultQA,
    SelfAssessRevisedInput,
    TagDataset,
)

context = yaml_loader(Path(MODULE_ROOT, "conf", "question_context.yaml"))
blooms_dict = yaml_loader(Path(MODULE_ROOT, "conf", "blooms.yaml"))["taxonomy"].get(
    "revised"
)
blooms_list = [item for sublist in blooms_dict.values() for item in sublist]
re_blooms_compiled = re.compile(r"|".join(blooms_list), re.IGNORECASE)


# Base class for RAG Modules
class BaseRAG(dspy.Module):
    def __init__(self, num_passages: int = 3, **kwargs):
        """Initialize shared components for RAG modules."""
        super().__init__()
        self.num_passages = num_passages
        self.context = None
        self.retrieve = None
        self.signature: Optional[dspy.Signature] = None
        self.generate_answer: Optional[Any] = None
        self.kwargs = kwargs
        self._set_context_and_signature()

    def _set_context_and_signature(self):
        """Set context and signature based on specified context type."""
        if self.kwargs.get("context") == "nbme":
            self.signature = SelfAssessRevisedInput
            temp_context = self._format_context(context["nbme"])
            shuffle(temp_context)
            self.context = temp_context
        elif self.kwargs.get("context") == "blooms" or "blooms" in self.kwargs.get(
            "context", []
        ):
            # gpt-4o-mini with SelfAssessBlooms is better default
            self.signature = SelfAssessBlooms
            temp_context = self._format_context(context["blooms"])
            shuffle(temp_context)
            self.context = temp_context
        elif self.kwargs.get("context") == "organism_research":
            self.signature = TagDataset
            temp_context = self._format_context(context["organism_research"])
            shuffle(temp_context)
            self.context = temp_context
        else:
            self.signature = DefaultQA
            self.retrieve = dspy.Retrieve(k=self.num_passages)

        self.signature_name = self.signature.__name__

    @staticmethod
    def _format_context(raw_context: dict) -> List[str]:
        """Format context into list of strings with capitalized keys and stripped values."""
        return [
            f"{k.strip().replace('_', ' ').capitalize()}| {v.strip()}"
            for k, v in raw_context.items()
        ]


# Define the module
# Specific RAG module implementations
class CoTRAG(BaseRAG):
    def __init__(self, num_passages: int = 3, **kwargs):
        """Initialize the CoTRAG module with specified context and passages."""
        super().__init__(num_passages=num_passages, **kwargs)
        self.generate_answer = dspy.ChainOfThought(self.signature)
        self.return_reasoning = kwargs.get("return_reasoning", False)

    def forward(self, question: str, return_reasoning: bool = False):
        """Forward method for processing the question through the RAG pipeline."""
        if self.retrieve:
            self.context = self.retrieve(question).passages
        else:
            # Shuffle the context for variety
            context = self.context.copy()
            shuffle(context)
            context = context[: self.num_passages]

        prediction = self.generate_answer(context=context, question=question)

        if return_reasoning or self.return_reasoning:
            return dspy.Prediction(
                context=context,
                answer=prediction.answer,
                reasoning=prediction.reasoning,
            )
        elif self.signature == TagDataset:
            # dump all prediction attr into the prediction object
            # filter "_" prefixed attributes
            data_dict = {
                k: v
                for k, v in prediction.__dict__["_store"].items()
                if not k.startswith("_")
            }
            return dspy.Prediction(context=context, **data_dict)
        else:
            return dspy.Prediction(context=context, answer=prediction.answer)


class CoTMultiHopRAG(BaseRAG):
    """Module for multi-hop reasoning with multiple query hops."""

    def __init__(
        self,
        num_passages: int = 5,
        passages_per_hop: int = 3,
        max_hops: int = 3,
        **kwargs,
    ):
        super().__init__(num_passages=num_passages, **kwargs)
        self.passages_per_hop = passages_per_hop
        self.max_hops = max_hops
        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)
        ]
        if not self.signature:
            self.signature = ReviseInput  # Default signature if none is specified

    def forward(self, question: str):
        """Multi-hop forward method for iterative retrieval and answering."""
        if self.context is None:
            for hop in range(self.max_hops):
                query = self.generate_query[hop](
                    context=self.context, question=question
                ).query
                passages = self.retrieve(query).passages
                self.context = dspy.deduplicate(self.context + passages)

        return self.generate_answer(question)


class CoTSelfCorrectRAG(BaseRAG):
    """Module for classifying Bloom's Taxonomy level with self-assessment."""

    def __init__(self, num_passages: int = 5, **kwargs):
        super().__init__(num_passages=num_passages, **kwargs)

        if not self.signature:
            self.signature = ClassifyBlooms  # Default signature if none is specified

    def forward(self, question: str):
        """Forward method for processing the question through the RAG pipeline."""

        initial_prediction = self.generate_answer(question, context=self.context)
        # format for self assessment
        self_check_question = (
            f"An LLM model evaluated the Bloom's Taxonomy level of the following  multiple choice question:\n"
            f"{question}\n"
            f" The model predicted the question is at the Bloom's level of:\n"
            f"Bloom's:```{initial_prediction.answer}```\n"
            f"Please evaluate the Bloom's Taxonomy level of the question and provide your independent assessment:"
        )

        # Perform self-assessment on the initial prediction
        if teacher_model := getattr(dspy.settings, "tm", None):
            with dspy.settings.context(lm=teacher_model):
                assess_response = dspy.ChainOfThought(SelfAssessBlooms)(
                    question=self_check_question, context=self.context
                )
        else:
            logger.error("Teacher model not found in settings.")

        return assess_response


class CoTSelfCorrectRAG(BaseRAG):
    """Module for classifying Bloom's Taxonomy level with self-assessment."""

    def __init__(self, num_passages: int = 5, **kwargs):
        super().__init__(num_passages=num_passages, **kwargs)

        if not self.signature:
            self.signature = ClassifyBlooms  # Default signature if none is specified

    def forward(self, question: str):
        """Forward method for processing the question through the RAG pipeline."""

        initial_prediction = self.generate_answer(question, context=self.context)
        # format for self assessment
        self_check_question = (
            f"An LLM model evaluated the Bloom's Taxonomy level of the following  multiple choice question:\n"
            f"{question}\n"
            f" The model predicted the question is at the Bloom's level of:\n"
            f"Bloom's:```{initial_prediction.answer}```\n"
            f"Please evaluate the Bloom's Taxonomy level of the question and provide your independent assessment:"
        )

        # Perform self-assessment on the initial prediction
        if teacher_model := getattr(dspy.settings, "tm", None):
            with dspy.settings.context(lm=teacher_model):
                assess_response = dspy.ChainOfThought(SelfAssessBlooms)(
                    question=self_check_question, context=self.context
                )
        else:
            logger.error("Teacher model not found in settings.")

        # compare the initial prediction with the self-assessment
        return assess_response
        # return dspy.Prediction(context=self.context, answer=assess_response.answer)

    # def generate_answer(self, question: str, context: Optional[List[str]] = None):
    #     """Generate an answer using the given context and question."""
    #     if context is None:
    #         context = self.retrieve(question).passages if self.retrieve else self.context
    #     prediction = dspy.ChainOfThought(self.signature)(context=context, question=question)
    #     return dspy.Prediction(context=context, answer=prediction.answer)
