"""Central config: everything the trainer and the demo agree on lives here."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
ONNX_PATH = DOCS_DIR / "model.onnx"
META_PATH = DOCS_DIR / "model_meta.json"

SEED = 42
N_EVENTS = 4000

# Thai fleet classes. Trailer (18W-T) occupies 2 adjacent docks; 18W-EXP is the
# export pool. Pallet capacities drive how many cartons an order can carry.
TRUCK_TYPES = ["4W", "6W", "10W", "18W", "18W-T", "18W-EXP", "22W"]
PALLET_CAP = {"4W": 3, "6W": 6, "10W": 12, "18W": 20, "18W-T": 26, "18W-EXP": 20, "22W": 32}

# Single product (UHT) — no product-category feature.
ARRANGE_METHODS = ["hand", "pallet"]  # hand-arrange is the slow path

CAT_COLS = ["truck_type", "arrange_method"]
NUM_COLS = ["carton_count", "sku_count", "hour_of_day",
            "asrs_rate", "forklift_count", "sortation_rate"]
TARGET = "duration_min"

# Fixture row used by the parity test AND the browser self-check —
# the demo page must predict the same minutes for this truck.
PARITY_FIXTURE = {
    "truck_type": "22W",
    "arrange_method": "hand",
    "carton_count": 3300,
    "sku_count": 12,
    "hour_of_day": 8,
    "asrs_rate": 100,
    "forklift_count": 2,
    "sortation_rate": 1400,
}
