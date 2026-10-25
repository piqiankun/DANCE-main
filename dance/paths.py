"""Shared project resource locations, resolved relative to this checkout."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(PROJECT_ROOT / "data")
PROMPT_DIR = str(PROJECT_ROOT / "prompts")
RESULTS_DIR = str(PROJECT_ROOT / "results") + os.sep
BENCHMARK_RESULTS_DIR = str(PROJECT_ROOT / "results" / "benchmarks")
CACHE_DIR = os.getenv("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
