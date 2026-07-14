"""Entrypoint: synthetic data -> model leaderboard -> export winner to ONNX.

Reads like a table of contents; every stage lives in src/.
"""
import logging

from src.export_onnx import export
from src.model import leaderboard
from src.synth import make_synthetic

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    df = make_synthetic()
    log.info("synthetic dock events: %d rows", len(df))

    board, fitted = leaderboard(df)
    print("\n=== Leaderboard (held-out test split) ===")
    print(board.round(2).to_string(index=False))

    parity = export(fitted["XGBoost"], board)
    log.info("exported ONNX + manifest to docs/ — parity fixture predicts %.2f min", parity)


if __name__ == "__main__":
    main()
