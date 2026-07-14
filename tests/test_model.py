"""Tests for the fragile parts: reproducibility, model ordering, ONNX parity."""
import numpy as np
import pandas as pd
import pytest

from config import CAT_COLS, NUM_COLS, PARITY_FIXTURE
from src.export_onnx import categories, encode_row, export
from src.model import leaderboard
from src.synth import make_synthetic


@pytest.fixture(scope="module")
def df():
    return make_synthetic()


@pytest.fixture(scope="module")
def fitted(df):
    board, models = leaderboard(df)
    return board, models


def test_synth_is_deterministic(df):
    again = make_synthetic()
    pd.testing.assert_frame_equal(df, again)


def test_synth_shape_and_ranges(df):
    from config import N_EVENTS, TRUCK_TYPES
    assert len(df) == N_EVENTS
    assert (df["duration_min"] >= 15).all()
    assert set(df["truck_type"]) == set(TRUCK_TYPES)


def test_hand_arrange_is_slower(df):
    """Domain truth from Simon: เรียงมือ (hand-arrange) is the slow path."""
    by_method = df.groupby("arrange_method")["duration_min"].mean()
    assert by_method["hand"] > by_method["pallet"]


def test_tree_models_beat_linear(fitted):
    board, _ = fitted
    mae = board.set_index("model")["MAE"]
    assert mae["RandomForest"] < mae["Linear"]
    assert mae["XGBoost"] < mae["Linear"]


def test_encode_row_matches_pipeline_width(fitted):
    _, models = fitted
    pipe = models["XGBoost"]
    cats = categories(pipe)
    vec = encode_row(PARITY_FIXTURE, cats)
    n_pipeline = pipe.named_steps["pre"].transform(
        pd.DataFrame([PARITY_FIXTURE])
    ).shape[1]
    assert vec.shape == (1, n_pipeline)


def test_onnx_parity_on_random_rows(fitted, df, tmp_path):
    """ONNX output must match the sklearn pipeline on many rows, not just one."""
    board, models = fitted
    pipe = models["XGBoost"]
    export(pipe, board)  # writes docs/model.onnx + manifest, raises if fixture parity breaks

    import onnxruntime as ort
    from config import ONNX_PATH

    sess = ort.InferenceSession(str(ONNX_PATH))
    cats = categories(pipe)
    sample = df.sample(50, random_state=7)[CAT_COLS + NUM_COLS]
    sk = pipe.predict(sample)
    onnx = np.array([
        sess.run(None, {"input": encode_row(row._asdict(), cats)})[0].ravel()[0]
        for row in sample.itertuples(index=False)
    ])
    assert np.allclose(sk, onnx, atol=0.1)
