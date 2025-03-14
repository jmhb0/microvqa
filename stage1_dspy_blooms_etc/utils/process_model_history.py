#!/usr/bin/env python3
"""process_model_history.py in src/microchat/utils."""


import json
import pprint
from pathlib import Path
from typing import Union, Optional

from loguru import logger


def history_to_jsonl(
    lm, output_dir: Union[str, Path], n: int = 1, output_file: Optional[str] = None
):
    """
    Writes the last n prompts and their completions to a JSONL file.

    Args:
        lm: The language model with a `history` attribute.
        output_dir: The directory to write the JSONL file to.
        n: The number of recent history items to include.
        output_file: Path to the JSONL file to write.
    """
    for idx, data in enumerate(lm.history[-n:]):
        output_file = output_file or f"{lm.model_name}_history_{idx:02d}.txt"
        output_path = Path(output_dir).joinpath(output_file)

        # Initialize an empty list to hold the formatted conversation
        formatted_output = []

        # Extract messages from the data
        messages = data["messages"]

        # Process the system message separately
        system_message = messages[0]
        if system_message["role"] == "system":
            system_message_str = system_message["content"].strip()
            formatted_output.append("System message:\n" + system_message_str)

        # Iterate over the messages starting from the second message
        for idx, message in enumerate(messages[1:]):
            role = message["role"]
            content = message["content"].strip()

            if role == "assistant":
                formatted_output.append("\nAssistant message:\n" + content)

            elif role == "user":
                formatted_output.append("\n") if idx == 0 else None
                formatted_output.append(
                    f"----- Example {idx:} -----\nUser message:\n{content}"
                )
        # If there's a 'response' in data, include it as the final assistant message
        if (
            "response" in data
            and "choices" in data["response"]
            and data["response"]["choices"]
        ):
            assistant_response = data["response"]["choices"][0]["message"][
                "content"
            ].strip()
            formatted_output.append("Response:\n" + assistant_response)

        # Combine the formatted output into a single string
        formatted_output_str = "\n".join(formatted_output)

        # save_formatted string
        with open(output_path, "w") as f:
            f.write(formatted_output_str)
