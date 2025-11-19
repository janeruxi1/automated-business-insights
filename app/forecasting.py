import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import io
import base64

# silence warnings
import warnings
warnings.filterwarnings("ignore", message="Importing plotly failed")
# import logging
# logging.getLogger("cmdstanpy").disabled=True


def load_data(file_path="data/sample_sales_data.csv"):
    """Load sample revenue data."""
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def forecast_revenue(df, days_ahead=30):
    """Train Prophet and forecast future revenue."""
    prophet_df = df.rename(columns={"date": "ds", "revenue": "y"})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)
    return model, forecast

def plot_forecast(model, forecast):
    """Generate and return a base64-encoded forecast plot."""
    fig = model.plot(forecast)
    plt.title("Revenue Forecast (Next 30 Days)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

def extract_model_df(model, df):
    """Return a clean HTML block summarizing Prophet model configuration."""
    sections = []

    # --- Training Range ---
    sections.append(
        f"<b>Training Data Range:</b><br/>"
        f"&bull; {df['ds'].min().date()} → {df['ds'].max().date()}<br/><br/>"
    )

    # --- Growth ---
    sections.append(
        f"<b>Growth:</b><br/>"
        f"&bull; {model.growth}<br/><br/>"
    )

    # --- Seasonalities ---
    season_lines = []
    for name, s in model.seasonalities.items():
        season_lines.append(
            f"&bull; {name.capitalize()} (Period={s['period']}, Fourier Order={s['fourier_order']})"
        )

    sections.append(
        "<b>Seasonalities Enabled:</b><br/>"
        + "<br/>".join(season_lines)
        + "<br/><br/>"
    )

    # --- Changepoints ---
    cp_lines = []
    if len(model.changepoints) > 0:
        for cp in model.changepoints[:10]:  # limit for readability
            cp_lines.append(f"&bull; {cp.date()}")
    else:
        cp_lines.append("&bull; None detected")

    sections.append(
        f"<b>Detected Changepoints:</b><br/>"
        + "<br/>".join(cp_lines)
        + "<br/><br/>"
    )

    # --- Priors ---
    sections.append(
        "<b>Prior Scales:</b><br/>"
        f"&bull; Changepoint prior scale: {model.changepoint_prior_scale}<br/>"
        f"&bull; Seasonality prior scale: {model.seasonality_prior_scale}<br/>"
        f"&bull; Holidays prior scale: {model.holidays_prior_scale}<br/><br/>"
    )

    return "".join(sections)

