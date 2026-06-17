"""Project paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
LOG_DIR = OUTPUT_DIR / "logs"
SAMPLE_CODE_DIR = PROJECT_ROOT / "sample_code"

# Local cg-lib path (place extracted dataset here)
CG_LIB_DIR = INPUT_DIR / "cg-lib"

# Kaggle paths
KAGGLE_INPUT_GLOB = "/kaggle/input/**/cg-lib"
KAGGLE_WORKING_DIR = Path("/kaggle/working")

# Default deck used in official sample (card IDs)
SAMPLE_DECK = [
    721, 721, 722, 722, 722, 722, 723, 723, 723, 723,
    1092, 1121, 1121, 1145, 1145, 1163, 1163, 1219, 1219, 1219, 1219,
    1227, 1227, 1227, 1227, 1262, 1262,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
]
