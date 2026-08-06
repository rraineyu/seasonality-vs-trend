"""
Streamlit dashboard for Holt-Winters weekly weather forecasting.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy for free at https://share.streamlit.io by pointing it at this repo + app.py.

Data handling: the underlying dataset is confidential and is never committed to this
repo. On the deployed app, it's loaded from Streamlit's private "Secrets" store (set
in the app's Settings on share.streamlit.io — never in git), so visitors see the
dashboard fully populated without needing to upload anything themselves. Locally, if
no secret is configured, it falls back to a file uploader for testing.
"""

from __future__ import annotations

import io
import re
import textwrap

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


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a dataframe that may have picked up stray whitespace (e.g. from an
    editor auto-indenting a pasted multi-line TOML string): strips column names,
    strips string cell values, and coerces mostly-numeric object columns to numeric.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            stripped = df[col].astype(str).str.strip()
            numeric = pd.to_numeric(stripped, errors="coerce")
            # If the vast majority of values convert cleanly, treat the column as numeric.
            if numeric.notna().mean() >= 0.9:
                df[col] = numeric
            else:
                df[col] = stripped
    return df


def load_data() -> tuple[pd.DataFrame | None, bool]:
    """
    Load the dataset from Streamlit Secrets if configured (used on the deployed app
    so visitors don't need to upload anything), otherwise fall back to a file
    uploader (used for local development/testing).

    Returns (dataframe_or_none, using_secret).
    """
    try:
        csv_text = st.secrets.get("data_csv")
    except Exception:
        # No secrets.toml configured (e.g. running locally without one) — fall through to uploader.
        csv_text = None

    if csv_text:
        # Undo any common indentation the secrets editor may have added to a
        # pasted multi-line string, then strip a leading/trailing blank line.
        csv_text = textwrap.dedent(csv_text).strip("\n")
        df = pd.read_csv(io.StringIO(csv_text))

        if len(df) == 0 and len(df.columns) > 0:
            # If the secrets editor collapsed newlines into a single line, everything
            # (header + all rows) parses as one giant header with zero data rows.
            # Reconstruct rows: the header field count is wherever the first purely
            # numeric-looking token appears (header names are text, data starts numeric).
            numeric_re = re.compile(r"^-?\d+(\.\d+)?$")
            all_fields = [f.strip() for f in csv_text.replace("\n", ",").split(",") if f.strip() != ""]
            first_numeric_idx = next(
                (i for i, f in enumerate(all_fields) if numeric_re.match(f)), None
            )
            if first_numeric_idx and first_numeric_idx > 0:
                n_cols = first_numeric_idx
                if len(all_fields) > n_cols and (len(all_fields) - n_cols) % n_cols == 0:
                    header = all_fields[:n_cols]
                    rows = [
                        all_fields[i : i + n_cols] for i in range(n_cols, len(all_fields), n_cols)
                    ]
                    df = pd.DataFrame(rows, columns=header)

        return clean_dataframe(df), True

    uploaded = st.sidebar.file_uploader("Upload data_week.csv", type="csv")
    if uploaded is None:
        return None, False
    return clean_dataframe(pd.read_csv(uploaded)), False


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
    data, using_secret = load_data()
    if using_secret:
        st.success("Data loaded.")

    if data is not None:
        with st.expander("Debug: data diagnostics"):
            st.write(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")
            st.write("Column dtypes:")
            st.dataframe(data.dtypes.astype(str).rename("dtype"))

    st.header("Model")
    beta_on = st.checkbox("Trend (beta) ON", value=True)
    gamma_on = st.checkbox("Seasonality (gamma) ON", value=True)
    seasonal_periods = st.number_input(
        "Seasonal periods (52 = weekly data, annual cycle)", min_value=2, value=52
    )

    st.header("Forecast")
    horizon = st.slider("Forecast horizon (weeks)", min_value=1, max_value=52, value=26)
    confidence = st.slider("Confidence level (%)", min_value=50, max_value=99, value=95)

if data is None:
    st.info(
        "Upload your `data_week.csv` file in the sidebar to get started. "
        "The dataset isn't bundled with this repo since it's confidential."
    )
    st.stop()

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
