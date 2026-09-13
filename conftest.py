"""
Pytest automatically discovers and imports conftest.py files, and adds
their directory to sys.path. Having this at the project root (not inside
tests/) guarantees `from src import ...` works whether pytest is invoked
as `pytest tests/`, `python -m pytest`, from a different working
directory, or from an IDE's test runner.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
