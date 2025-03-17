#!/usr/bin/env python3
"""base_llmodel.py in src/microchat/models."""
import os
import re

from pydantic import (
    NonNegativeInt,
    NonNegativeFloat,
    ValidationInfo,
    field_validator,
    ConfigDict,
)

from pydantic import Field
from typing import Any
from pydantic import BaseModel
from typing import List, Optional
from loguru import logger

import dspy
import tiktoken as tk


# Base class for all backend LLM models
class LLModel(BaseModel):
    """Model configuration for loading a model and its associated tokenizer."""

    # set pydantic model config
    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    # args
    lm: dspy.LM  # arbitrary type
    model_name: Optional[str] = Field(None, alias="model_name")  # set by model
    model_prefix: Optional[str] = Field(None, alias="model_prefix")  # set by model
    temperature: NonNegativeFloat = 1.0
    max_tokens: NonNegativeInt = 2048
    top_p: NonNegativeFloat = 1.0
    frequency_penalty: NonNegativeFloat = 0.0
    presence_penalty: NonNegativeFloat = 0.0
    seed: NonNegativeInt = 8675309
    tokenizer: Optional[tk.core.Encoding] = Field(None)
    tokenizer_name: Optional[str] = Field(None, alias="tokenizer_name")  # set by model
    kwargs: Optional[dict] = Field(None, alias="kwargs")

    @field_validator("lm")
    def validate_model(cls, v: dspy.LM, info: ValidationInfo):
        # assign
        if not isinstance(v, dspy.LM):
            raise ValueError(f"Model {v} is not an instance of dspy.LM.")

        return v

    def model_post_init(self, __context: Any) -> None:
        # set model name and prefix  a
        self.model_name: str = (
            getattr(self.lm, "model_name", None) or self.lm.model.split("/")[-1]
        )
        if self.model_name:
            self.model_prefix: str = re.sub(
                r"\d{8}$|mini$|latest$|mini_\d{8}$", "", self.model_name
            )

            # set tokenizer name
            if "o1" in self.model_prefix:
                self.tokenizer_name: str = "o200k_base"
            else:
                self.tokenizer_name: str = tk.encoding_name_for_model(self.model_prefix)

            self.tokenizer = tk.get_encoding(self.tokenizer_name)

            logger.debug(f"Model: {self.model_name}") if os.getenv("DEBUG") else None
            logger.debug(f"Tokenizer: {self.tokenizer_name}")
