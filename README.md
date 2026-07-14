# Warehouse Dock Planner — ML Estimates, Rules Decide

**[▶ Live demo](https://simonhtet.github.io/warehouse-dock-planner-demo/)** — an XGBoost model exported to ONNX,
running entirely in your browser, feeding a deterministic dock-assignment engine.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-duration%20regression-F7931E)
![ONNX](https://img.shields.io/badge/ONNX-in--browser%20inference-005CED?logo=onnx&logoColor=white)
![Tests](https://img.shields.io/badge/pytest-parity%20verified-brightgreen)

## The architecture principle

A dock planner has two very different sub-problems, and they deserve different tools:

| Sub-problem | Nature | Tool |
|---|---|---|
| *How long will this truck occupy a dock?* | Multi-factor pattern with interactions (truck class × cartons × SKUs × arrange method × time-of-day × ASRS/forklift/sortation state) | **ML** — learned regression |
| *Which dock, which wave?* | Assignment under constraints — must be auditable, explainable, and overridable by a human planner | **Rules** — waves, zones, exits, pins |

**ML estimates, rules decide.** The ONNX model predicts each order's loading duration; everything
else is auditable logic: docks open with a **07:00 gate-open wave** (trucks enter 4 min apart —
one gate), and from then on assignment is **event-driven**: the moment a truck clears its exit, the
next queued truck docks immediately; **exports load first**, exclusively on docks D1–D7; a dock stays occupied through
loading **plus exit-to-gate time** (5 min for a 4W up to 15 min for a 22W or trailer); trailers span
2 adjacent docks. An hourly strip shows **planned cartons vs the sortation-line capacity**. And the
human stays in charge: click a truck, click a new spot — legal moves get **📌 pinned** (surviving
re-plans), illegal ones are **blocked** with the exact rule violated. When the board is packed, a
**📥 yard/staging strip** lets the planner park a truck out of the plan (freeing its dock) and
place it back later.

## Why ONNX

The model is trained in Python (scikit-learn pipeline + XGBoost), then exported to ONNX and served
by [onnxruntime-web](https://onnxruntime.ai/) — **no backend, no API, no server**. The same `.onnx`
artifact could be served from a Python worker, a C# service, or an edge device. Training and serving
are decoupled by design.

A **parity test** guarantees the deployed model is the trained model: `tests/test_model.py` checks
ONNX output ≡ sklearn pipeline output on 50 rows (±0.1 min), and the demo page re-runs a fixture
prediction on every load — the "model verified" badge in the header means the browser's answer
matches the training machine's to the hundredth of a minute.

## The model ladder

`train.py` climbs **Linear Regression → Random Forest → XGBoost** and only keeps complexity that
earns its accuracy (held-out MAE, in minutes):

```
       model  MAE  RMSE
     XGBoost 5.54  6.98
RandomForest 6.46  8.31
      Linear 8.61 11.29
```

The synthetic generator bakes in nonlinearity and interactions (hand-arrange cost scales with
cartons and worsens on a slow sortation line, pallet cost rides the ASRS rate and forklift count,
pre-9am ramp-up, lunch-break slowdown) — which is exactly why the tree models win, and the whole
"why not just linear regression" story in one table.

## Run it

```bash
pip install -r requirements.txt
python train.py     # data → leaderboard → docs/model.onnx + manifest (seeded, reproducible)
pytest              # determinism, model ordering, ONNX↔sklearn parity
```

Then open `docs/index.html` via any static server (`python -m http.server -d docs`).

## Structure

```
config.py             constants shared by trainer and demo (incl. the parity fixture)
train.py              entrypoint — data → leaderboard → export
src/synth.py          synthetic dock-event generator
src/model.py          pipeline builders + MAE/RMSE leaderboard
src/export_onnx.py    XGBoost → ONNX + model_meta.json manifest + parity check
tests/test_model.py   the fragile parts: reproducibility, ordering, parity
docs/                 GitHub Pages — index.html + model.onnx + model_meta.json
```

The ONNX graph contains only the regressor; one-hot encoding is replicated from the manifest
(`model_meta.json` fixes the exact feature order) on both sides — `encode_row()` in Python,
`encodeRow()` in JS — and the parity test is what keeps them honest.

## Scope

All data is **synthetic** — no real warehouse, customer, or operational data. This is the public
mini-version of a private production system (a SAP/WMS-fed continuous dock-timeline planner);
the architecture walkthrough of the real thing is available on request.
