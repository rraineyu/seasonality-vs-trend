# Weekly Weather Forecasting with Holt-Winters Exponential Smoothing

Team project for BAX 442 (Homework 6) — forecasting a weekly weather time series 26 weeks into the future using the Holt-Winters filter, and comparing how the trend (beta) and seasonal (gamma) components affect forecast quality.

**Authors:** Anita Cheng, Raine Yu, Udita Saha, Sheryn Liao

## Overview

Using average weekly temperature (`avg_temp`) from a company's weekly operations dataset, we:

1. Converted the series into a `ts` object with weekly seasonality (frequency = 52)
2. Fit four Holt-Winters models by toggling the trend and seasonal components:
   - Beta ON / Gamma ON (full model: level + trend + seasonality)
   - Beta ON / Gamma OFF (level + trend only)
   - Beta OFF / Gamma ON (level + seasonality only)
   - Beta OFF / Gamma OFF (simple exponential smoothing — level only)
3. Plotted the data series, trend component, and seasonal component for each combination
4. Generated 26-week out-of-sample forecasts with 95% confidence bands for each model

## Findings

The four models diverge sharply in stability. The models with trend enabled (`beta = TRUE`) produced increasingly wide, unstable confidence bands over the forecast horizon, suggesting the trend component was overfitting noise in the data rather than capturing a real long-term drift in temperature. The seasonal-only and level-only models (`beta = FALSE`) produced tighter, more plausible forecasts, with the seasonal component clearly recovering the annual temperature cycle. This suggests that for a seasonal series like weekly temperature, a model with `gamma = TRUE` and `beta = FALSE` is likely the better fit, since temperature has strong seasonality but no persistent linear trend.

## Repo Contents

| File | Description |
|---|---|
| `BAX_442_Homework6.Rmd` | R Markdown source: data prep, four Holt-Winters model fits, decomposition plots, forecasts |
| `BAX442_Homework_6.pdf` | Knitted report with all plots and write-up |

**Note:** `data_week.csv` (the underlying weekly operations dataset) is not included, since it's course/company-provided data rather than ours to redistribute. To reproduce, place your own `data_week.csv` with an `avg_temp` column in the project root and update the `setwd()` path in the Rmd.

## Methods

- **Model:** `stats::HoltWinters()`
- **Forecasting:** `forecast::forecast()`, h = 26, 95% confidence level
- **Language:** R (packages: `readr`, `dplyr`, `ggplot2`, `forecast`)

## Reproducing

```r
install.packages(c("readr", "dplyr", "ggplot2", "forecast"))
rmarkdown::render("BAX_442_Homework6.Rmd")
```
