#!/usr/bin/env python3
"""model_factory.py in src/microchat/models."""

from typing import Tuple, Optional
import re

from dotenv import find_dotenv
from dotenv import load_dotenv
from loguru import logger

from microchat import MODULE_ROOT
from microchat.fileio.text.readers import yaml_loader
from microchat.models.base_llmodel import LLModel
from microchat.models.model_registry import ModelType
import dspy
import tiktoken as tk
import pprint as pp

# set ENV vars from .env
load_dotenv(find_dotenv())


def create_model(    model_name: str, config: Optional[dict] = None, kwargs: Optional[dict] = None) -> LLModel:
    """Create a model and its associated transformation given a model type and cache directory.

    Args:
        model_name (str): The type of model to create, specified as an enum value.

    Returns:
        Tuple[dspy.LM, tk.core.Encoding]: A tuple containing the model and its associated tokenizer.
    """
    #
    config_file = MODULE_ROOT.joinpath("conf", "llm_config.yaml")
    config = yaml_loader(config_file).get("llm_settings")
    if config and kwargs:
        raise ValueError("Cannot provide both config and kwargs.")
    elif not config and not kwargs:
        logger.info(f"Loading config from {config_file}, default settings")
        config = yaml_loader(config_file)
        config = config['default']
    elif isinstance(config, 'str'):
        logger.info(f"Loading config from {config_file}, {config} settings")
        config = yaml_loader(config_file)
        config = config[config]
    elif kwargs:
        config = kwargs

    # get model prefix (remove ending date or _mini)
    model_prefix = re.sub(r"\d{8}$|mini$|latest$|mini_\d{8}$", "", model_name)
    model_name = model_name.lower().replace("-", "_")

    # convert model_type to ModelType enum
    model_type = ModelType[model_name]
    if not isinstance(model_type, ModelType):
        raise ValueError(f"Model {model_name} not found in ModelType enum.")

    # load model
    logger.info(f"Loading model: {model_type.name}")
    temperature = config.pop("temperature", 1.0)
    max_tokens = config.pop("max_tokens", 2048)
    match model_type:
        case ModelType.gpt_4o: # ModelType.gpt_4_turbo
            dspy_model = dspy.LM("/".join(model_type.value),
                                 temperature=temperature,
                                 max_tokens=max_tokens,
                                 **config)
        case ModelType.gpt_4o_mini:
            dspy_model = dspy.LM("/".join(model_type.value),
                                 temperature=temperature,
                                 max_tokens=max_tokens,
                                 **config)
        case ModelType.o1_mini | ModelType.o1_preview:
            temperature = 1.0 # required for o1
            max_tokens = 5000 # required for o1
            dspy_model = dspy.LM("/".join(model_type.value),
                                 temperature=temperature,
                                 max_tokens=max_tokens,
                                 **config)
        case ModelType.claude_3_opus:
            dspy_model = dspy.LM("/".join(model_type.value),
                                 temperature=temperature,
                                 max_tokens=max_tokens,
                                 **config)
        case _: # no match
            logger.error("Model not found.")
            raise NotImplementedError(f"Model {model_type} is not yet supported.")

    #
    dspy_model.model_name = model_type.value[-1]
    dspy_model.model_prefix = model_prefix
    dspy_model.config = dict(temperature=temperature, max_tokens=max_tokens, **config)
    logger.info(f"Model settings: {pp.pformat(dspy_model.config, indent=4)}")

    return LLModel(lm = dspy_model)