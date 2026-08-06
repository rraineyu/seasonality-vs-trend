# Weekly Weather Forecasting with Holt-Winters Exponential Smoothing

Forecasting a weekly weather time series 26 weeks into the future using Holt-Winters exponential smoothing, comparing how the trend (beta) and seasonal (gamma) components affect forecast quality.

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
| `holt_winters_forecast.ipynb` | Full analysis: data prep, four Holt-Winters model fits, decomposition plots, 26-week forecasts with confidence bands |
| `time_series_forecasting_report.pdf` | Report with all plots and write-up |

**Note:** `data_week.csv` (the underlying weekly operations dataset) is not included, since it's course/company-provided data rather than ours to redistribute. To reproduce, place your own `data_week.csv` with an `avg_temp` column in the project root and update the file path in the notebook.

## Methods

- **Model:** `statsmodels.tsa.exponential_smoothing.ets.ETSModel`
- **Forecasting:** `get_prediction()`, h = 26, 95% confidence interval
- **Language:** Python (packages: `pandas`, `numpy`, `matplotlib`, `statsmodels`)

## Reproducing

```bash
pip install pandas numpy matplotlib statsmodels jupyter
jupyter notebook holt_winters_forecast.ipynb
```
