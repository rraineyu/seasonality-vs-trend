"""
Streamlit dashboard for Holt-Winters weekly weather forecasting.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy for free at https://share.streamlit.io by pointing it at this repo + app.py.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

st.set_page_config(page_title="Weather Forecast Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Core forecasting logic (kept separate from Streamlit calls so it's testable)
# ---------------------------------------------------------------------------
def fit_ets(series: pd.Series, trend: bool, seasonal: bool, seasonal_periods: int):
    """Fit a Holt-Winters (ETS) model with trend/seasonal components toggled on/off."""
    model = ETSModel(
        series,
        error="add",
        trend="add" if trend else None,
        seasonal="add" if seasonal else None,
        seasonal_periods=seasonal_periods if seasonal else None,
    )
    return model.fit(disp=False)


def forecast_with_ci(fit_result, series_len: int, horizon: int, alpha: float) -> pd.DataFrame:
    """Return a dataframe of mean forecast + confidence interval for `horizon` steps ahead."""
    pred = fit_result.get_prediction(start=series_len, end=series_len + horizon - 1)
    return pred.summary_frame(alpha=alpha)


def load_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Build a weekly-indexed series from a dataframe column."""
    values = df[column].astype(float)
    index = pd.date_range(start="2020-01-01", periods=len(values), freq="W")
    return pd.Series(values.values, index=index, name=column)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("Weekly Weather Forecast Dashboard")
st.caption(
    "Holt-Winters (ETS) exponential smoothing — toggle the trend and seasonal "
    "components to see how each affects the fit and the out-of-sample forecast."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload data_week.csv", type="csv")

    st.header("Model")
    beta_on = st.checkbox("Trend (beta) ON", value=True)
    gamma_on = st.checkbox("Seasonality (gamma) ON", value=True)
    seasonal_periods = st.number_input(
        "Seasonal periods (52 = weekly data, annual cycle)", min_value=2, value=52
    )

    st.header("Forecast")
    horizon = st.slider("Forecast horizon (weeks)", min_value=1, max_value=52, value=26)
    confidence = st.slider("Confidence level (%)", min_value=50, max_value=99, value=95)

if uploaded is None:
    st.info(
        "Upload your `data_week.csv` file in the sidebar to get started. "
        "The dataset isn't bundled with this repo since it's course/company-provided data."
    )
    st.stop()

data = pd.read_csv(uploaded)
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found in the uploaded file.")
    st.stop()

default_col = "avg_temp" if "avg_temp" in numeric_cols else numeric_cols[0]
column = st.selectbox("Series to forecast", numeric_cols, index=numeric_cols.index(default_col))

series = load_series(data, column)

with st.spinner("Fitting model..."):
    fit = fit_ets(series, trend=beta_on, seasonal=gamma_on, seasonal_periods=int(seasonal_periods))
    forecast_df = forecast_with_ci(fit, len(series), horizon, alpha=1 - confidence / 100)

# --- Data / trend / seasonal components ------------------------------------
states = fit.states
n_plots = 1 + ("trend" in states.columns) + ("seasonal" in states.columns)
fig_components = make_subplots(rows=n_plots, cols=1, subplot_titles=(
    ["Data Series"]
    + (["Trend Component"] if "trend" in states.columns else [])
    + (["Seasonal Component"] if "seasonal" in states.columns else [])
))

fig_components.add_trace(
    go.Scatter(x=series.index, y=series.values, name=column, mode="lines"), row=1, col=1
)

row = 2
if "trend" in states.columns:
    fig_components.add_trace(
        go.Scatter(x=states.index, y=states["trend"], name="Trend", mode="lines"), row=row, col=1
    )
    row += 1
if "seasonal" in states.columns:
    fig_components.add_trace(
        go.Scatter(x=states.index, y=states["seasonal"], name="Seasonal", mode="lines"), row=row, col=1
    )

fig_components.update_layout(height=250 * n_plots, showlegend=False)
st.subheader("Decomposition")
st.plotly_chart(fig_components, use_container_width=True)

# --- Forecast plot -----------------------------------------------------------
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=series.index, y=series.values, name="Observed", mode="lines"))
fig_forecast.add_trace(
    go.Scatter(x=forecast_df.index, y=forecast_df["mean"], name="Forecast", mode="lines")
)
fig_forecast.add_trace(
    go.Scatter(
        x=list(forecast_df.index) + list(forecast_df.index[::-1]),
        y=list(forecast_df["pi_upper"]) + list(forecast_df["pi_lower"][::-1]),
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"{confidence}% CI",
    )
)
fig_forecast.update_layout(
    title=f"{horizon}-Week Forecast (Trend {'ON' if beta_on else 'OFF'}, Seasonality {'ON' if gamma_on else 'OFF'})",
    xaxis_title="Week",
    yaxis_title=column,
    height=450,
)
st.subheader("Forecast")
st.plotly_chart(fig_forecast, use_container_width=True)

with st.expander("Forecast values"):
    st.dataframe(forecast_df.rename(columns={"mean": "forecast"}))

with st.expander("Model summary"):
    st.text(fit.summary().as_text())
