"""Synthetic dock-event generator.

Duration structure (all synthetic, tuned for plausibility not realism):
- hand-arrange: cartons stacked individually — cost scales with cartons and
  worsens when the sortation line runs slow; the SLOW path
- pallet: forklift cycles + ASRS retrieval — cost scales with pallets,
  eased by more forklifts and a faster ASRS picking rate
- SKU count adds changeover/sortation overhead (worse for hand-arrange)
- before 09:00 the warehouse is still ramping up; 12:00–13:00 lunch runs
  at half rate; trailers pay a two-dock repositioning premium

Nonlinearity + interactions are deliberate — they are why the tree models
beat the linear baseline. No real warehouse data anywhere.
"""
import numpy as np
import pandas as pd

from config import ARRANGE_METHODS, N_EVENTS, PALLET_CAP, SEED, TRUCK_TYPES

CARTONS_PER_PALLET = (100, 180)

# Big trucks run denser pallets (more cartons, similar handling time): fixed
# carton ranges per class instead of the per-pallet density above.
DENSE_CTN = {"18W": (2400, 3200), "18W-T": (2400, 3600), "18W-EXP": (4500, 5300)}


def make_synthetic(n: int = N_EVENTS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    truck = rng.choice(TRUCK_TYPES, size=n, p=[0.14, 0.16, 0.22, 0.18, 0.08, 0.12, 0.10])
    cap = pd.Series(truck).map(PALLET_CAP).to_numpy()

    pallets = np.maximum(1, np.round(cap * rng.uniform(0.6, 1.0, size=n)))
    cartons = np.round(pallets * rng.integers(*CARTONS_PER_PALLET, size=n)).astype(int)
    for t, (lo, hi) in DENSE_CTN.items():
        m = truck == t
        cartons[m] = rng.integers(lo, hi + 1, size=m.sum())
    skus = 1 + rng.binomial(24, 0.25, size=n)                      # skewed low, 1–25
    method = rng.choice(ARRANGE_METHODS, size=n, p=[0.3, 0.7])
    hour = rng.integers(6, 18, size=n)
    asrs = rng.integers(60, 131, size=n)                           # pallets/hour
    forklifts = rng.integers(1, 5, size=n)
    sortation = rng.integers(800, 2001, size=n)                    # cartons/hour

    hand = method == "hand"
    load = np.where(
        hand,
        cartons * 0.020 * (1200 / sortation) ** 0.8,               # hand: carton-by-carton (denser pallets → faster per carton)
        pallets * (60 / asrs) + pallets * 1.8 / np.sqrt(forklifts) # pallet: ASRS + forklift
    )
    sku_overhead = skus * np.where(hand, 1.5, 0.8)
    ramp_up = np.where(hour < 9, 0.15 * load + 5, 0)               # pre-9am half-awake warehouse
    lunch = np.where(hour == 12, 20, 0)                            # 12:00–13:00 half rate
    trailer = np.where(truck == "18W-T", 12, 0)                    # two-dock repositioning
    export = np.where(truck == "18W-EXP",                          # customs docs + container seal
                      rng.uniform(40, 70, size=n), 0)

    duration = 12 + load + sku_overhead + ramp_up + lunch + trailer + export + rng.normal(0, 6, size=n)

    return pd.DataFrame({
        "truck_type": truck,
        "arrange_method": method,
        "carton_count": cartons,
        "sku_count": skus,
        "hour_of_day": hour,
        "asrs_rate": asrs,
        "forklift_count": forklifts,
        "sortation_rate": sortation,
        "duration_min": np.clip(duration, 15, None).round(1),
    })
