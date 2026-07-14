"""Make the project root importable in tests (config, src.*)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
