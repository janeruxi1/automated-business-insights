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
