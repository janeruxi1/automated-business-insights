"""Rolling-origin backtest for daily forecasts.

A backtest replays history: train on everything before a cutoff date, forecast the
next `horizon` days, score against what actually happened, then move the cutoff
forward and repeat. Every forecast is scored only on data the model never saw.

Prophet is imported inside `prophet_forecast`, so the metrics, baselines and
splitting logic can be tested without installing it.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

Forecaster = Callable[[pd.DataFrame, int], np.ndarray]

SEASON = 7  # weekly seasonality in daily data


# ---------------------------------------------------------------- metrics
def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, in percent. Days with zero actuals are skipped."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE, in percent."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def mase(y_true, y_pred, y_train, m: int = SEASON) -> float:
    """Mean absolute scaled error: forecast MAE divided by the in-sample MAE of a
    seasonal-naive forecast. Below 1 means better than repeating last week."""
    y_train = np.asarray(y_train, float)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return mae(y_true, y_pred) / scale


# ---------------------------------------------------------------- forecasters
def naive_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Repeat the last observed day."""
    return np.repeat(float(train["y"].iloc[-1]), horizon)


def seasonal_naive_forecast(train: pd.DataFrame, horizon: int, m: int = SEASON) -> np.ndarray:
    """Repeat the last observed week."""
    last_season = train["y"].iloc[-m:].to_numpy(float)
    return np.resize(last_season, horizon)


def prophet_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Prophet with weekly seasonality and Brazilian public holidays.

    Yearly seasonality is off: each training window holds at most ~1.5 years,
    too little to estimate a yearly pattern reliably.
    """
    import logging

    from prophet import Prophet

    logging.getLogger("cmdstanpy").disabled = True
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
    )
    model.add_country_holidays(country_name="BR")
    model.fit(train[["ds", "y"]])
    future = model.make_future_dataframe(periods=horizon, include_history=False)
    return model.predict(future)["yhat"].to_numpy()


# ---------------------------------------------------------------- backtest
def rolling_origin_cutoffs(n: int, initial: int, horizon: int, step: int) -> List[int]:
    """Row positions where each training window ends (exclusive).

    Fold k trains on rows [0, cutoff) and is scored on rows [cutoff, cutoff + horizon).
    """
    if initial + horizon > n:
        raise ValueError("Series too short for one fold.")
    return list(range(initial, n - horizon + 1, step))


def run_backtest(
    df: pd.DataFrame,
    forecasters: Dict[str, Forecaster],
    initial: int = 365,
    horizon: int = 30,
    step: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each forecaster on every fold.

    `df` needs columns ds (date) and y (value), sorted by date with no gaps.
    Returns (scores, forecasts): one row per fold and model, and one row per
    forecast day and model.
    """
    df = df.sort_values("ds").reset_index(drop=True)
    score_rows, forecast_rows = [], []
    for fold, cut in enumerate(rolling_origin_cutoffs(len(df), initial, horizon, step), 1):
        train, test = df.iloc[:cut], df.iloc[cut : cut + horizon]
        for name, fn in forecasters.items():
            pred = np.asarray(fn(train, horizon), float)
            score_rows.append(
                {
                    "fold": fold,
                    "train_end": train["ds"].iloc[-1].date().isoformat(),
                    "test_start": test["ds"].iloc[0].date().isoformat(),
                    "test_end": test["ds"].iloc[-1].date().isoformat(),
                    "model": name,
                    "mae": mae(test["y"], pred),
                    "rmse": rmse(test["y"], pred),
                    "mape": mape(test["y"], pred),
                    "smape": smape(test["y"], pred),
                    "mase": mase(test["y"], pred, train["y"]),
                }
            )
            forecast_rows.append(
                pd.DataFrame({"fold": fold, "model": name, "ds": test["ds"].to_numpy(),
                              "y": test["y"].to_numpy(), "yhat": pred})
            )
    return pd.DataFrame(score_rows), pd.concat(forecast_rows, ignore_index=True)


def summarize(scores: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Average each metric across folds, and count folds where a model's MAE
    beats the baseline's MAE."""
    summary = scores.groupby("model")[["mae", "rmse", "mape", "smape", "mase"]].mean()
    base = scores[scores["model"] == baseline].set_index("fold")["mae"]
    wins = (
        scores.assign(base_mae=scores["fold"].map(base))
        .assign(win=lambda d: d["mae"] < d["base_mae"])
        .groupby("model")["win"]
        .sum()
    )
    summary["folds_beating_baseline"] = wins.astype(int)
    summary["mae_vs_baseline_pct"] = (summary["mae"] / summary.loc[baseline, "mae"] - 1) * 100
    return summary.sort_values("mae")
