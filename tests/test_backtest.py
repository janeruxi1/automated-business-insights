"""Tests for the backtest logic and for the published results.

These run without Prophet: forecasters are swapped for simple stand-ins.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from backtest import (  # noqa: E402
    mae,
    mape,
    mase,
    naive_forecast,
    rmse,
    rolling_origin_cutoffs,
    run_backtest,
    seasonal_naive_forecast,
    smape,
    summarize,
)

RESULTS = ROOT / "results"


# ------------------------------------------------------------ metrics
def test_metrics_on_known_values():
    y, yhat = [10, 20, 40], [12, 18, 40]
    assert mae(y, yhat) == pytest.approx(4 / 3)
    assert rmse(y, yhat) == pytest.approx(np.sqrt(8 / 3))
    assert mape(y, yhat) == pytest.approx((20 + 10 + 0) / 3)
    assert smape(y, yhat) == pytest.approx(np.mean([2 * 2 / 22, 2 * 2 / 38, 0]) * 100)


def test_mape_skips_zero_actuals():
    assert mape([0, 10], [5, 12]) == pytest.approx(20.0)


def test_mase_is_one_for_seasonal_naive_error_scale():
    train = np.arange(28, dtype=float)  # week-over-week change is always 7
    assert mase([0, 0], [7, 7], train, m=7) == pytest.approx(1.0)


# ------------------------------------------------------------ baselines
def _frame(values):
    return pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=len(values)), "y": values})


def test_naive_repeats_last_value():
    assert naive_forecast(_frame([1, 2, 3]), 4).tolist() == [3, 3, 3, 3]


def test_seasonal_naive_repeats_last_week():
    train = _frame(list(range(1, 15)))  # last week is 8..14
    assert seasonal_naive_forecast(train, 9).tolist() == [8, 9, 10, 11, 12, 13, 14, 8, 9]


# ------------------------------------------------------------ splitting
def test_cutoffs_cover_series_without_overlap():
    cuts = rolling_origin_cutoffs(n=594, initial=365, horizon=30, step=30)
    assert cuts == [365, 395, 425, 455, 485, 515, 545]
    assert cuts[-1] + 30 <= 594


def test_cutoffs_reject_short_series():
    with pytest.raises(ValueError):
        rolling_origin_cutoffs(n=100, initial=90, horizon=30, step=30)


def test_forecasters_never_see_test_data():
    df = _frame(np.arange(60, dtype=float))
    seen = []

    def spy(train, horizon):
        seen.append(train["ds"].max())
        return np.zeros(horizon)

    scores, forecasts = run_backtest(df, {"spy": spy}, initial=30, horizon=10, step=10)
    for fold, last_train_day in enumerate(seen, 1):
        test_days = forecasts.loc[forecasts["fold"] == fold, "ds"]
        assert test_days.min() > last_train_day


def test_summary_counts_wins_against_baseline():
    scores = pd.DataFrame(
        {
            "fold": [1, 1, 2, 2],
            "model": ["a", "base", "a", "base"],
            "mae": [1.0, 2.0, 3.0, 2.0],
            "rmse": 0.0, "mape": 0.0, "smape": 0.0, "mase": 0.0,
        }
    )
    s = summarize(scores, "base")
    assert s.loc["a", "folds_beating_baseline"] == 1
    assert s.loc["a", "mae_vs_baseline_pct"] == pytest.approx(0.0)


# ------------------------------------------------------------ published figures
def test_data_window_is_complete():
    daily = pd.read_csv(ROOT / "data" / "olist_daily_orders.csv", parse_dates=["date"])
    assert len(daily) == 594
    assert daily["date"].diff().dropna().eq(pd.Timedelta(days=1)).all()
    assert daily["orders"].sum() == 98_390


def test_summary_table_matches_fold_scores():
    folds = pd.read_csv(RESULTS / "backtest_folds.csv")
    means = folds.groupby("model")["mae"].mean()
    table = (RESULTS / "backtest_summary.md").read_text()
    for model, value in means.items():
        row = next(line for line in table.splitlines() if line.startswith(f"| {model} |"))
        assert row.split("|")[2].strip() == f"{value:.1f}"


def test_readme_shows_current_results():
    readme = (ROOT / "README.md").read_text()
    block = re.search(r"<!-- BACKTEST:START -->\n(.*?)<!-- BACKTEST:END -->", readme, re.S)
    assert block, "README is missing the backtest results block"
    assert block.group(1).strip() == (RESULTS / "backtest_summary.md").read_text().strip()
