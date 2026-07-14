"""Central config: everything the model and demo agree on lives here."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
ONNX_PATH = DOCS_DIR / "model.onnx"
META_PATH = DOCS_DIR / "model_meta.json"

SEED = 42
N_EVENTS = 3000

# Generic warehouse — deliberately NOT any real dock layout.
DOCKS = [f"D{d}" for d in range(1, 9)]
TRUCK_TYPES = ["4W", "6W", "10W"]
PRODUCT_CATEGORIES = ["ambient", "chilled", "frozen"]

CAT_COLS = ["truck_size", "product_category", "dock"]
NUM_COLS = ["carton_count", "hour_of_day", "day_of_week"]
TARGET = "duration_min"

# Fixture row used by the parity test AND the browser self-check —
# the demo page must predict the same minutes for this truck.
PARITY_FIXTURE = {
    "truck_size": "10W",
    "product_category": "frozen",
    "dock": "D3",
    "carton_count": 400,
    "hour_of_day": 9,
    "day_of_week": 2,
}
