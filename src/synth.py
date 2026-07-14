"""Synthetic dock-event generator.

Realistic dock service durations with nonlinearity + interactions baked in
(big frozen loads are disproportionately slow, morning congestion bump) —
which is exactly why the tree models should beat the linear baseline.

All data is synthetic. No real warehouse, customer, or operational data.
"""
import numpy as np
import pandas as pd

from config import DOCKS, N_EVENTS, PRODUCT_CATEGORIES, SEED, TRUCK_TYPES

SIZE_FACTOR = {"4W": 1.0, "6W": 1.5, "10W": 2.2}
PRODUCT_FACTOR = {"ambient": 1.0, "chilled": 1.3, "frozen": 1.6}


def make_synthetic(n: int = N_EVENTS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    truck_size = rng.choice(TRUCK_TYPES, size=n, p=[0.40, 0.35, 0.25])
    cartons = rng.integers(20, 600, size=n)
    product = rng.choice(PRODUCT_CATEGORIES, size=n, p=[0.30, 0.50, 0.20])
    dock = rng.choice(DOCKS, size=n)
    hour = rng.integers(6, 20, size=n)
    dow = rng.integers(0, 7, size=n)

    size_f = pd.Series(truck_size).map(SIZE_FACTOR).to_numpy()
    prod_f = pd.Series(product).map(PRODUCT_FACTOR).to_numpy()

    base = 8 + 0.05 * cartons * size_f * prod_f            # interaction term
    rush = np.where((hour >= 8) & (hour <= 10), 6.0, 0.0)  # morning congestion
    noise = rng.normal(0, 5, size=n)
    duration = np.clip(base + rush + noise, 5, None)

    return pd.DataFrame({
        "truck_size": truck_size,
        "carton_count": cartons,
        "product_category": product,
        "dock": dock,
        "hour_of_day": hour,
        "day_of_week": dow,
        "duration_min": duration.round(1),
    })
