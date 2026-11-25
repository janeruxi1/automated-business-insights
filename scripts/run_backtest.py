"""Run the rolling-origin backtest on real Olist daily orders.

Usage:
    python scripts/run_backtest.py

Writes to results/:
    backtest_folds.csv     scores per fold and model
    backtest_forecasts.csv every forecast day vs. the actual
    backtest_summary.md    the summary table pasted into the README
    backtest_chart.png     actuals vs. forecasts across all test windows

Decision rule, fixed before running: Prophet counts as better than the
seasonal-naive baseline only if its average MAE is lower AND it wins on MAE
in at least 5 of the 7 folds.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from backtest import (  # noqa: E402
    naive_forecast,
    prophet_forecast,
    run_backtest,
    seasonal_naive_forecast,
    summarize,
)

INITIAL, HORIZON, STEP = 365, 30, 30
BASELINE = "Seasonal naive (last week)"
MIN_WINS = 5
RESULTS = ROOT / "results"


def summary_markdown(summary: pd.DataFrame, n_folds: int) -> str:
    lines = [
        f"| Model | MAE (orders/day) | RMSE | MAPE | sMAPE | MASE | Folds beating baseline (of {n_folds}) | MAE vs. baseline |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, r in summary.iterrows():
        vs = "—" if model == BASELINE else f"{r.mae_vs_baseline_pct:+.1f}%"
        wins = "—" if model == BASELINE else str(int(r.folds_beating_baseline))
        lines.append(
            f"| {model} | {r.mae:.1f} | {r.rmse:.1f} | {r.mape:.1f}% | {r.smape:.1f}% "
            f"| {r.mase:.2f} | {wins} | {vs} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "olist_daily_orders.csv", parse_dates=["date"])
    df = df.rename(columns={"date": "ds", "orders": "y"})

    forecasters = {
        "Prophet (weekly + BR holidays)": prophet_forecast,
        BASELINE: seasonal_naive_forecast,
        "Naive (last day)": naive_forecast,
    }
    scores, forecasts = run_backtest(df, forecasters, INITIAL, HORIZON, STEP)
    summary = summarize(scores, BASELINE)
    n_folds = scores["fold"].nunique()

    RESULTS.mkdir(exist_ok=True)
    scores.round(4).to_csv(RESULTS / "backtest_folds.csv", index=False)
    forecasts.round({"y": 4, "yhat": 4}).to_csv(RESULTS / "backtest_forecasts.csv", index=False)
    # Robustness: fold 1 starts right after New Year, so "repeat last week" copies the
    # holiday lull into January. Report the comparison without it too.
    rest = summarize(scores[scores["fold"] != 1], BASELINE)
    p, b = rest.loc["Prophet (weekly + BR holidays)", "mae"], rest.loc[BASELINE, "mae"]
    note = (
        f"\nExcluding fold 1 (whose baseline copies the post-New Year lull), Prophet's MAE is "
        f"{p:.1f} vs. {b:.1f} for seasonal naive ({(p / b - 1) * 100:+.1f}%).\n"
    )
    (RESULTS / "backtest_summary.md").write_text(summary_markdown(summary, n_folds) + note)

    prophet = summary.loc["Prophet (weekly + BR holidays)"]
    passed = prophet.mae < summary.loc[BASELINE, "mae"] and prophet.folds_beating_baseline >= MIN_WINS
    print(summary.round(2).to_string())
    print(f"\nDecision rule (lower mean MAE and >= {MIN_WINS}/{n_folds} fold wins): "
          f"{'PASSED' if passed else 'NOT PASSED'}")

    fig, ax = plt.subplots(figsize=(11, 4.2))
    actual = forecasts.drop_duplicates("ds")
    ax.plot(df["ds"], df["y"], color="#9aa5b1", lw=0.8, label="Actual (training history)")
    ax.plot(actual["ds"], actual["y"], color="#1f2933", lw=1.2, label="Actual (test windows)")
    for name, color in [("Prophet (weekly + BR holidays)", "#2f6fdd"), (BASELINE, "#e07b00")]:
        f = forecasts[forecasts["model"] == name]
        for i, (_, g) in enumerate(f.groupby("fold")):
            ax.plot(g["ds"], g["yhat"], color=color, lw=1.4, label=name if i == 0 else None)
    for cut in scores["test_start"].unique():
        ax.axvline(pd.Timestamp(cut), color="#cbd2d9", lw=0.6, zorder=0)
    ax.set_ylabel("Orders per day")
    ax.set_title(f"Olist daily orders: {n_folds} rolling 30-day forecasts vs. actuals")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "backtest_chart.png", dpi=150)


if __name__ == "__main__":
    main()
