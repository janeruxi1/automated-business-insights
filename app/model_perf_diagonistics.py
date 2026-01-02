import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller


def extract_model_performance(model, df):
    """Compute forecast training performance metrics and return HTML text"""
    # Merge actuals with forecast
    future=model.make_future_dataframe(periods=0,freq='D')
    forecast_in_sample = model.predict(future)
    merged = df.rename(columns={"date": "ds", "revenue": "y"}).merge(
        forecast_in_sample[["ds", "yhat"]], on="ds", how="left"
    )

    # Calculate performance metrics
    merged=merged.dropna()
    y_true=merged["y"]
    y_pred=merged["yhat"]
    
    merged["error"] = y_true - y_pred
    mae = np.mean(np.abs(merged["error"]))
    mse = np.mean(merged["error"] ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(merged["error"] / merged["y"])) * 100
    smape = 100 * np.mean(
        2 * np.abs(merged["error"]) / (np.abs(y_true) + np.abs(y_pred))
    )
    
    # Create HTML summary
    performance_html = (
        f"<b>Model Performance on Training Data:</b><br/>"
        f"&bull; Mean Absolute Error (MAE): {mae:.2f}<br/>"
        f"&bull; Mean Squared Error (MSE): {mse:.2f}<br/>"
        f"&bull; Root Mean Squared Error (RMSE): {rmse:.2f}<br/>"
        f"&bull; Mean Absolute Percentage Error (MAPE): {mape:.2f}%<br/>"
        f"&bull; Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.2f}%<br/><br/>"
    )

    return performance_html


# def extract_model_df(model, forecast_df, actual_df):
#     """
#     Extract detailed model information:
#     - Model performance (RMSE, MAE, MAPE, R2...)
#     - Diagnostics (residual stats, ADF, Shapiro)
#     - Fit summary (AIC, BIC, hyperparameters)
#     """

#     # Merge actuals & forecast
#     merged = actual_df.merge(forecast_df[['ds', 'yhat']], on='ds', how='inner')
#     merged['residual'] = merged['y'] - merged['yhat']

#     y_true = merged["y"].values
#     y_pred = merged["yhat"].values
#     residuals = merged["residual"].values

#     # -------------------------
#     #  Model Performance
#     # -------------------------
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = mean_absolute_error(y_true, y_pred)
#     mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#     mse = mean_squared_error(y_true, y_pred)

#     try:
#         r2 = r2_score(y_true, y_pred)
#     except:
#         r2 = None

#     # -------------------------
#     # Diagnostics
#     # -------------------------
#     resid_mean = np.mean(residuals)
#     resid_std = np.std(residuals)

#     # lag-1 autocorrelation (residuals)
#     try:
#         resid_autocorr = pd.Series(residuals).autocorr(lag=1)
#     except:
#         resid_autocorr = None

#     # Shapiro-Wilk normality test (residuals)
#     try:
#         shapiro_stat, shapiro_p = shapiro(residuals)
#     except:
#         shapiro_stat, shapiro_p = None, None

#     # ADF stationarity test (residuals)
#     try:
#         adf_stat, adf_p, *_ = adfuller(residuals)
#     except:
#         adf_stat, adf_p = None, None

#     # -------------------------
#     # Model Fit Summary
#     # -------------------------
#     aic = getattr(model, "aic", None)
#     bic = getattr(model, "bic", None)

#     # prophet / neuralprophet parameters
#     model_params = None
#     if hasattr(model, "params"):
#         model_params = model.params
#     elif hasattr(model, "seasonality_mode"):
#         model_params = {
#             "seasonality_mode": model.seasonality_mode,
#             "changepoint_prior_scale": model.changepoint_prior_scale,
#             "seasonality_prior_scale": model.seasonality_prior_scale,
#         }

#     # -------------------------
#     # Build DataFrame Output
#     # -------------------------
#     data = {
#         "Metric": [
#             "RMSE", "MAE", "MAPE (%)", "MSE", "R² Score",
#             "Residual Mean", "Residual Std Dev", "Residual Autocorr (lag-1)",
#             "Shapiro-Wilk Stat", "Shapiro-Wilk p-value",
#             "ADF Statistic", "ADF p-value",
#             "AIC", "BIC", "Training Sample Size",
#             "Model Parameters"
#         ],
#         "Value": [
#             rmse, mae, mape, mse, r2,
#             resid_mean, resid_std, resid_autocorr,
#             shapiro_stat, shapiro_p,
#             adf_stat, adf_p,
#             aic, bic, len(actual_df),
#             str(model_params)
#         ]
#     }

#     model_df = pd.DataFrame(data)
#     return model_df
