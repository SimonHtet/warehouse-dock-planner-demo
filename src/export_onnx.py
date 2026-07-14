"""Export the fitted XGBoost regressor to ONNX + a feature manifest.

Only the regressor is converted (FloatTensorType input). The one-hot encoding
is replicated by the consumer — encode_row() here, and the same logic in JS
from model_meta.json — because string-tensor pipelines are flaky in
onnxruntime-web, while a plain float input runs everywhere. The parity check
proves sklearn and ONNX agree before anything ships.
"""
import json
from datetime import date

import numpy as np
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline

from config import CAT_COLS, META_PATH, NUM_COLS, ONNX_PATH, PARITY_FIXTURE, SEED, N_EVENTS


def categories(pipe: Pipeline) -> dict[str, list[str]]:
    """Fitted category lists per cat column, in OHE order."""
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    return {col: [str(c) for c in cats] for col, cats in zip(CAT_COLS, ohe.categories_)}


def encode_row(row: dict, cats: dict[str, list[str]]) -> np.ndarray:
    """One-hot the categoricals + append numerics — the manifest contract.

    Must stay in lockstep with encodeRow() in docs/index.html.
    """
    vec = []
    for col in CAT_COLS:
        vec.extend(1.0 if str(row[col]) == c else 0.0 for c in cats[col])
    vec.extend(float(row[col]) for col in NUM_COLS)
    return np.array([vec], dtype=np.float32)


def export(pipe: Pipeline, board) -> float:
    """Write model.onnx + model_meta.json; return the parity prediction."""
    cats = categories(pipe)
    n_features = sum(len(v) for v in cats.values()) + len(NUM_COLS)

    onnx_model = convert_xgboost(
        pipe.named_steps["model"],
        initial_types=[("input", FloatTensorType([None, n_features]))],
    )
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    ONNX_PATH.write_bytes(onnx_model.SerializeToString())

    # Parity: sklearn pipeline vs ONNX runtime on the shared fixture.
    import onnxruntime as ort
    import pandas as pd

    sk_pred = float(pipe.predict(pd.DataFrame([PARITY_FIXTURE]))[0])
    sess = ort.InferenceSession(str(ONNX_PATH))
    onnx_pred = float(sess.run(None, {"input": encode_row(PARITY_FIXTURE, cats)})[0].ravel()[0])
    if abs(sk_pred - onnx_pred) > 0.1:
        raise ValueError(f"ONNX parity broken: sklearn {sk_pred:.3f} vs onnx {onnx_pred:.3f}")

    META_PATH.write_text(json.dumps({
        "winner": "XGBoost",
        "cat_cols": [{"name": c, "categories": cats[c]} for c in CAT_COLS],
        "num_cols": NUM_COLS,
        "n_features": n_features,
        "metrics": board.round(3).to_dict(orient="records"),
        "parity": {"input": PARITY_FIXTURE, "expected_min": round(onnx_pred, 2)},
        "trained_at": date.today().isoformat(),
        "seed": SEED,
        "n_events": N_EVENTS,
    }, indent=2))
    return onnx_pred
