#!/usr/bin/env python3
"""process_text.py in src/microchat/utils."""
from pathlib import Path
from typing import Tuple


import re
from microchat import MODULE_ROOT
from typing import Optional
from loguru import logger


import tiktoken as tk
from microchat.fileio.text.readers import yaml_loader


context = yaml_loader(Path(MODULE_ROOT, "conf", "question_context.yaml"))
blooms_dict = yaml_loader(Path(MODULE_ROOT, "conf", "blooms.yaml"))["taxonomy"].get(
    "revised"
)
blooms_list = [item for sublist in blooms_dict.values() for item in sublist]
re_blooms_compiled = re.compile(r"|".join(blooms_list), re.IGNORECASE)


def process_blooms(
    answer: str, reference_dict: Optional[dict] = blooms_dict
) -> Tuple[int, str, str]:
    # extract the blooms level from the response
    blooms_name = None
    blooms_verb = answer  # default to answer if not found
    blooms_level = -1
    if match := re_blooms_compiled.search(blooms_verb):
        # find the blooms verb from the match
        blooms_verb = match.group().lower()
        # find the level of the blooms taxonomy from blooms_dict
        blooms_level = next(
            level for level, names in reference_dict.items() if blooms_verb in names
        )
        # process reference dict to get 1st key from each value
        # this is to standardize the name of the blooms level
        # although the blooms verb is more descriptive
        blooms_name = blooms_dict[blooms_level][0]
    else:
        logger.warning(f"Bloom's taxonomy level not found: {blooms_verb}")
    return blooms_level, blooms_name, blooms_verb


def compute_chars(prompt: str) -> int:
    """Compute the number of characters in a prompt."""
    return len(prompt)


def compute_tokens(prompt: str, tokenizer: tk.Encoding) -> int:
    """Compute the number of tokens in a prompt."""
    return len(tokenizer.encode(prompt))
