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
| *Which dock should it go to?* | Assignment under constraints — must be auditable, explainable, and overridable by a human planner | **Rules** — deterministic earliest-finish |

**ML estimates, rules decide.** The demo makes the loop literal: for each truck, the ONNX model
predicts a duration *per candidate dock*, and the rule engine picks the dock with the earliest finish.
The prediction is learned; the decision is a one-line rule anyone can audit.

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
