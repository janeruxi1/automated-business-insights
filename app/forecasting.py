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

def extract_model_df(model,df):
    """Return a formatted text block of prophet model configuration."""
    info = []
    
    # Training range
    info.append(f"<b>Training Data Range:</b> {df['ds'].min().date()} → {df['ds'].max().date()}\n")

    # Growth
    info.append(f"<b>Growth:</b> {model.growth}\n")

    # Seasonalities
    info.append(f"<b>Seasonalities Enabled:</b>\n")
    for name, s in model.seasonalities.items():
        info.append(f"• {name.capitalize()} (Period={s['period']}, Fourier Order={s['fourier_order']})\n")

    # Changepoints
    info.append(f"<b>Number of Changepoints:</b> {len(model.changepoints)}\n")
    info.append("<b>Detected Changepoints:</b>\n")
    if len(model.changepoints) > 0:
        for cp in model.changepoints[:10]:   # show top 10
            info.append(f"• {cp.date()}\n")
    else:
        info.append("• None detected.\n")
    
    # Priors
    info.append("<b>Prior Scales:</b>\n")
    info.append(f"• Changepoint prior scale: {model.changepoint_prior_scale}\n")
    info.append(f"• Seasonality prior scale: {model.seasonality_prior_scale}\n")
    info.append(f"• Holidays prior scale: {model.holidays_prior_scale}\n")

    return "<br/>".join(info)
