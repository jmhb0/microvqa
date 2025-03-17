#!/usr/bin/env python3
"""model_registry.py in src/microchat/models.

Adapted from:
https://github.com/sanketx/AL-foundation-models/blob/main/ALFM/src/models/registry.py
"""

from enum import Enum

MODELS = [
    "claude-3-opus-20240229",
    "gpt-4o-mini",
    "command-nightly",
    "databricks-meta-llama-3-1-70b-instruct",
]


class ModelType(Enum):
    """Enum of supported Models."""

    # openai
    gpt_3_5_turbo = ("openai", "gpt-3.5-turbo")  # points to latest snapshot

    gpt_4_turbo = ("openai", "gpt-4-turbo")  # points to latest snapshot
    gpt_4_turbo_20240409 = ("openai", "gpt-4-turbo-2024-07-18")
    gpt_4_turbo_preview = ("openai", "gpt-4-turbo-preview")  # points to latest snapshot

    gpt_4o_mini = ("openai", "gpt-4o-mini")  # points to latest snapshot
    gpt_4o_mini_20240718 = ("openai", "gpt-4o-mini-2024-07-18")

    gpt_4o = ("openai", "gpt-4o")  # points to latest snapshot
    gpt_4o_20240806 = ("openai", "gpt-4o-2024-08-06")
    gpt_4o_20240513 = ("openai", "gpt-4o-2024-05-13")
    gpt_4o_latest = ("openai", "gpt-4o-latest")  # dynamic model continuously updates

    o1_mini = ("openai", "o1-mini")  # points to latest snapshot
    o1_mini_20240912 = ("openai", "o1-mini-2024-09-12")

    o1_preview = ("openai", "o1-preview")  # points to latest snapshot
    o1_preview_20240912 = ("openai", "o1-preview-2024-09-12")

    # claude
    claude_3_opus = ("anthropic", "claude-3-opus-20240229")

    # cohere
    cohere_latest = ("cohere", "command-nightly")

    # databricks
    databricks_llama_3_1 = ("databricks", "databricks-meta-llama-3-1-70b-instruct")

    def __repr__(self):
        return self.name
