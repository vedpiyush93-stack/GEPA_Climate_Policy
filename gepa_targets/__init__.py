"""GEPA for climate-target detection in national climate policy documents.

Every LLM call is cached inside the repo (cache/dspy), so re-running the notebook replays the same model answers and
reproduces every number exactly. Delete cache/dspy (or set GEPA_CACHE_DIR) to make fresh calls."""
import os
import pathlib

import dspy

CACHE_DIR = os.environ.get("GEPA_CACHE_DIR", str(pathlib.Path(__file__).resolve().parent.parent / "cache" / "dspy"))
dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=CACHE_DIR)
