#!/usr/bin/env python3
"""token_metrics.py in src/microchat/metrics."""


import math


def compute_token_metric(orig_tokens, pred_tokens, k=0.5):
    """
    Calculate the token difference score using an exponential decay function.

    Parameters:
    - orig_tokens (float or int): The original number of tokens.
    - pred_tokens (float or int): The predicted number of tokens.
    - k (float): The scaling factor controlling the rate of decay (default is 0.5).
        Smaller k (e.g., k=0.3): The score drops more rapidly with increasing difference.
        Larger k (e.g., k=0.7): The score decreases more slowly, allowing for a wider tolerance.

    Returns:
    - float: The token difference score between 0 and 1.
    """
    # Ensure that orig_tokens is not zero to avoid division by zero
    if orig_tokens == 0:
        raise ValueError("orig_tokens must be greater than zero.")

    # Calculate the absolute difference
    diff = abs(orig_tokens - pred_tokens)

    # Calculate the exponent term
    exponent = -(diff / (k * orig_tokens))

    return math.exp(exponent)
