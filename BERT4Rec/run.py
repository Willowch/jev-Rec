"""Run RecBole BERT4Rec in the OpenJev comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comparison.recbole_runner import main


if __name__ == "__main__":
    main("BERT4Rec")
