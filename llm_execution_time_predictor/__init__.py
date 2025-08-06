"""
LLM Execution Time Predictor

A CLI tool for profiling and predicting LLM batch inference latency.
"""

__version__ = "0.1.2"
__author__ = "Vikranth Srivatsa"

from .llm_forward_predictor_cli import main

__all__ = ["main"]