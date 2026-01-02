import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def extract_full_model_report(model, df, forecast_df):
    """
    Unified extractor for Prophet model:
    - Model configuration
    - Performance metrics
    - Diagnostics summary
    """

    # ─────────────────────────────────────────
    # 1. MODEL CONFIGURATION
    # ─────────────────────────────────────────
    info = []

    info.append(f"<b>Training Range:</b><br/>• {df['ds'].min().date()} → {df['ds'].max().date()}<br/><br/>")
    info.append(f"<b>Growth Type:</b><br/>• {model.growth}<br/><br/>")

    # Seasonalities
    info.append("<b>Seasonalities Enabled:</b><br/>")
    for name, s in model.seasonalities.items():
        info.append(f"• {name.capitalize()} (Period={s['period']}, Fourier Order={s['fourier_order']})<br/>")
    info.append("<br/>")

    # Changepoints
    info.append(f"<b>Number of Changepoints:</b><br/>• {len(model.changepoints)}<br/><br/>")
    info.append("<b>Detected Changepoints (first 10):</b><br/>")
    if len(model.changepoints) > 0:
        for cp in model.changepoints[:10]:
            info.append(f"• {cp.date()}<br/>")
    else:
        info.append("• None<br/>")
    info.append("<br/>")

    # Priors
    info.append("<b>Prior Scales:</b><br/>")
    info.append(f"• Changepoint prior scale: {model.changepoint_prior_scale}<br/>")
    info.append(f"• Seasonality prior scale: {model.seasonality_prior_scale}<br/>")
    info.append(f"• Holidays prior scale: {model.holidays_prior_scale}<br/><br/>")

    model_info_text = "".join(info)

    # ─────────────────────────────────────────
    # 2. MODEL PERFORMANCE
    # ─────────────────────────────────────────

    # Only evaluate if actual values exist
    perf_text = ""
    if "y" in df.columns:
        merged = pd.merge(df[["ds", "y"]], forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds")

        y_true = merged["y"]
        y_pred = merged["yhat"]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Coverage: how often actuals fall inside forecast ranges
        coverage = np.mean(
            (merged["y"] >= merged["yhat_lower"]) & (merged["y"] <= merged["yhat_upper"])
        ) * 100

        perf_lines = [
            "<b>Model Performance:</b><br/>",
            f"• MAE: {mae:.2f}<br/>",
            f"• RMSE: {rmse:.2f}<br/>",
            f"• MAPE: {mape:.2f}%<br/>",
            f"• Prediction Interval Coverage: {coverage:.1f}%<br/><br/>"
        ]

        perf_text = "".join(perf_lines)

    # ─────────────────────────────────────────
    # 3. MODEL DIAGNOSTICS
    # ─────────────────────────────────────────
    diag_text = ""
    if "y" in df.columns:
        residuals = y_true - y_pred

        bias = np.mean(residuals)
        std = np.std(residuals)

        # Simple autocorrelation diagnostic (lag 1)
        autocorr = residuals.autocorr(lag=1)

        diag_lines = [
            "<b>Diagnostics:</b><br/>",
            f"• Residual Mean (Bias): {bias:.2f}<br/>",
            f"• Residual Std Dev: {std:.2f}<br/>",
            f"• Residual Autocorrelation (lag 1): {autocorr:.2f}<br/>",
        ]

        # Interpretation
        diag_lines.append("<br/><b>Interpretation:</b><br/>")
        diag_lines.append("• Low bias means stable model.<br/>" if abs(bias) < std * 0.1 else
                          "• Noticeable bias may indicate under/over prediction.<br/>")

        diag_lines.append("• Low autocorrelation → residuals behave normally.<br/>" if abs(autocorr) < 0.3 else
                          "• High autocorrelation → missing seasonality or trends.<br/>")

        diag_text = "".join(diag_lines)

    # ─────────────────────────────────────────
    # RETURN EVERYTHING CLEANLY
    # ─────────────────────────────────────────
    return {
        "model_info": model_info_text,
        "performance": perf_text,
        "diagnostics": diag_text
    }
