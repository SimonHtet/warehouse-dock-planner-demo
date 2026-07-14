"""Model ladder: Linear (baseline) -> Random Forest -> XGBoost.

Climb on purpose and only keep the complexity that earns its accuracy.
`leaderboard()` returns MAE/RMSE per model on a held-out split; the caller
picks the winner. MAE = average minutes off (easy to explain to logistics).
"""
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import CAT_COLS, NUM_COLS, SEED, TARGET


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", "passthrough", NUM_COLS),
    ])


def make_pipelines() -> dict[str, Pipeline]:
    return {
        "Linear": Pipeline([("pre", make_preprocessor()), ("model", LinearRegression())]),
        "RandomForest": Pipeline([("pre", make_preprocessor()), ("model", RandomForestRegressor(
            n_estimators=300, random_state=SEED, n_jobs=-1,
        ))]),
        "XGBoost": Pipeline([("pre", make_preprocessor()), ("model", xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1,
        ))]),
    }


def split(df: pd.DataFrame):
    X = df[CAT_COLS + NUM_COLS]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.20, random_state=SEED)


def leaderboard(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Fit every pipeline, score on the held-out split, return (board, fitted)."""
    X_train, X_test, y_train, y_test = split(df)
    fitted, rows = {}, []
    for name, pipe in make_pipelines().items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        })
        fitted[name] = pipe
    board = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    return board, fitted
