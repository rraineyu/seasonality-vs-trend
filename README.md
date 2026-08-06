# Weekly Weather Forecasting with Holt-Winters Exponential Smoothing

Forecasting a weekly weather time series 26 weeks into the future using Holt-Winters exponential smoothing, comparing how the trend (beta) and seasonal (gamma) components affect forecast quality.

**[Live dashboard →](#) (add your Streamlit Cloud link here after deploying)**

## Overview

Using average weekly temperature (`avg_temp`) from a weekly operations dataset, this project:

1. Builds a weekly-indexed time series (52 periods/year seasonality)
2. Fits four Holt-Winters models by toggling the trend and seasonal components:
   - Beta ON / Gamma ON (full model: level + trend + seasonality)
   - Beta ON / Gamma OFF (level + trend only)
   - Beta OFF / Gamma ON (level + seasonality only)
   - Beta OFF / Gamma OFF (simple exponential smoothing — level only)
3. Plots the data series, trend component, and seasonal component for each combination
4. Generates 26-week out-of-sample forecasts with 95% confidence bands for each model

## Findings

The four models diverge sharply in stability. The models with trend enabled produced increasingly wide, unstable confidence bands over the forecast horizon, suggesting the trend component was overfitting noise in the data rather than capturing a real long-term drift in temperature. The seasonal-only and level-only models produced tighter, more plausible forecasts, with the seasonal component clearly recovering the annual temperature cycle. This suggests that for a seasonal series like weekly temperature, a model with seasonality enabled and no trend term is likely the better fit, since temperature has strong seasonality but no persistent linear trend.

## Repo Contents

| File | Description |
|---|---|
| `app.py` | Streamlit dashboard — interactive version of the analysis (upload data, toggle trend/seasonality, adjust forecast horizon) |
| `holt_winters_forecast.ipynb` | Full analysis notebook: data prep, four Holt-Winters model fits, decomposition plots, 26-week forecasts with confidence bands |
| `time_series_forecasting_report.pdf` | Report with all plots and write-up |
| `requirements.txt` | Python dependencies for both the notebook and the dashboard |

**Note:** `data_week.csv` (the underlying weekly operations dataset) is confidential and is not included in this repo. See "Dashboard" below for how the live version loads it privately.

## Dashboard

The Streamlit dashboard lets visitors pick which numeric column to forecast, toggle the trend and seasonal components on/off, and adjust the forecast horizon and confidence level — all with interactive Plotly charts.

Since the dataset is confidential, it's never committed to this repo or exposed to visitors. Instead, it's loaded server-side via **Streamlit Secrets** — a private key/value store attached to the deployed app that only the app owner can edit. Visitors just see the dashboard already populated; they never see or download the raw data. If no secret is configured (e.g. running locally without one), the app falls back to a file-upload widget instead.

**Run locally (with your own data):**
```bash
pip install -r requirements.txt
streamlit run app.py
# uploads your CSV via the sidebar since no secret is configured locally
```

**Deploy for free with the data private (so you can share a live link):**
1. Push this repo to GitHub — `data_week.csv` should **not** be included (it's already excluded via `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click "New app", select this repo, branch `main`, and set the main file path to `app.py`
4. Click "Deploy"
5. Once deployed, open the app's **Settings → Secrets** and paste in:
   ```toml
   data_csv = """
   <paste the full contents of data_week.csv here>
   """
   ```
   Save. The app will restart and auto-load the data for every visitor — without it ever touching GitHub.
6. Copy the app's public URL (`https://<your-app>.streamlit.app`) and add it to the top of this README

## Methods

- **Model:** `statsmodels.tsa.exponential_smoothing.ets.ETSModel`
- **Forecasting:** `get_prediction()`, adjustable horizon (default 26 weeks), adjustable confidence interval (default 95%)
- **Language:** Python (packages: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `streamlit`, `plotly`)

## Reproducing the Notebook

```bash
pip install -r requirements.txt matplotlib jupyter
jupyter notebook holt_winters_forecast.ipynb
```
