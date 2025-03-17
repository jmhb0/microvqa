#!/usr/bin/env python3
"""teleprompter_registry.py in src/microchat/teleprompters."""

from enum import Enum


class OptimizerType(Enum):
    """Enum of supported Models."""

    # bootstrap
    bootstrap = ("bootstrap", "bootstrap_few_shot")
    bootstrap_random = ("bootstrap", "bootstrap_few_shot_with_random_search")
    bootstrap_optuna = ("bootstrap", "bootstrap_few_shot_with_optuna")

    bootstrap_fine_tune = ("bootstrap", "bootstrap_fine_tune")

    # miprov2
    miprov2 = ("miprov2", "miprov2")

    def __repr__(self):
        return self.name
