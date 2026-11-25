"""Build the daily order series used by the forecast backtest.

Source: Olist "Brazilian E-Commerce Public Dataset" (Kaggle: olistbr/brazilian-ecommerce),
file olist_orders_dataset.csv, licensed CC BY-NC-SA 4.0.

Usage:
    python scripts/build_daily_orders.py path/to/olist_orders_dataset.csv

Writes data/olist_daily_orders.csv with one row per day: date, orders.
"""
import sys
from pathlib import Path

import pandas as pd

# Orders ramp up from early January 2017. After 2018-08-21 the counts collapse
# (collection stops, not demand), so the window ends there.
START = "2017-01-05"
END = "2018-08-21"
OUT = Path(__file__).resolve().parents[1] / "data" / "olist_daily_orders.csv"


def build(raw_path: str) -> pd.DataFrame:
    orders = pd.read_csv(raw_path, parse_dates=["order_purchase_timestamp"])
    daily = (
        orders.set_index("order_purchase_timestamp")
        .resample("D")
        .size()
        .rename("orders")
        .loc[START:END]
        .rename_axis("date")
        .reset_index()
    )
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    return daily


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    daily = build(sys.argv[1])
    daily.to_csv(OUT, index=False)
    print(f"Wrote {len(daily)} days ({daily['orders'].sum():,} orders) to {OUT}")
