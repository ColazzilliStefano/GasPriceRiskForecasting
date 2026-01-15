# Energy Market Risk & Forecasting: MPG-GAS Analysis
### *Quantitative Framework for Hybrid ARIMA-EGARCH Modeling with Expected Shortfall*

This repository presents an institutional-grade econometric pipeline for the **MPG-GAS** price series. The core objective is to move beyond standard Gaussian assumptions to model the "Fat-Tail" reality of energy commodities using **Extreme Value Theory (EVT)**.

---

## Executive Summary
Traditional models often fail in energy markets because they assume "Normal" volatility. Our analysis reveals a **Kurtosis > 3**, indicating that extreme events are far more frequent than a Gaussian bell curve suggests.

This framework implements a **SARIMAX-EGARCH** engine to solve:
1. **Asymmetric Volatility:** In gas markets, price spikes (fear of shortage) generate more volatility than price drops—captured via EGARCH.
2. **Structural Breaks:** Exogenous shocks are dynamically identified and neutralized to ensure the convergence of the Maximum Likelihood Estimation (MLE).
3. **Tail Risk Management:** We use **Expected Shortfall (ES)** at 99% (Downside) to quantify the average loss in worst-case scenarios, providing a safer buffer than simple Value-at-Risk (VaR).

---

## Diagnostics & Results

### 1. Exploratory Data Analysis (EDA)
Analysis of the price evolution, returns distribution, and normality check. The QQ-Plot (bottom right) clearly shows the deviation from normality in the tails.
![EDA Plots](results/01_eda_plots.png)

### 2. Structural Breaks & Outlier Mapping
A map of extreme events (shocks) that exceed the dynamic sigma threshold. These events are treated as exogenous dummies to prevent biasing the GARCH parameters.
![Shock Mapping](results/02_shock_mapping.png)

### 3. Tail Risk Audit (Gaussian vs. Reality)
Comparison between theoretical Gaussian tails (Blue) and the actual **Student-t distribution** (Red). The log-scale highlights the "Fat Tail" risk that standard models miss.
![Tail Risk Audit](results/03_tail_audit.png)

### 4. Final Forecast Performance & Risk Buffer
Backtesting results using Walk-Forward validation.
- **Blue Line:** Point Forecast (T+1).
- **Red Area:** The **Downside Risk Buffer (ES 99%)**. Note how the model expands the risk buffer during the 2022 volatility crisis.
![Forecast Performance](results/04_backtest_performance.png)

### 5. T+1 Risk Snapshot (The "Money Chart")
The probability density for the **next trading day**. This visualizes the exact levels for **VaR** (Threshold) and **Expected Shortfall** (Average Crash Intensity), enabling precise capital provisioning.
![Risk Snapshot](results/05_risk_snapshot.png)

---

## Model Architecture & Technical Deep-Dive

### 1. Mean Modeling: SARIMAX (0,1,3)
* **Order Selection:** Auto-ARIMA with BIC penalization ensures parsimony.
* **Exogenous Component:** The **Shock_Exog** coefficient absorbs massive price swings (Structural Breaks) that would otherwise bias the autoregressive parameters.

### 2. Volatility Audit: EGARCH(1,1)
* **Why EGARCH?** Unlike standard GARCH, EGARCH models the *logarithm* of volatility, allowing for an asymmetric response to news (Leverage Effect).
* **Distribution (Student's t):** The model estimates the degrees of freedom ($\nu$) dynamically. A low $\nu$ confirms a **Heavy-Tail Regime**, mathematically invalidating Gaussian risk models.

---

## Backtesting Metrics
- **MAPE (Accuracy):** ~4.5% (Excellent for high-volatility commodities).
- **Risk Coverage:** The model targets a 1% breach rate (99% confidence). 
- **ES Violations:** < 1% observed in out-of-sample testing, indicating the model is conservative and robust.

---

## Tech Stack
* **Core:** Python, Pandas, NumPy
* **Econometrics:** `statsmodels` (SARIMAX), `arch` (EGARCH)
* **Viz:** Matplotlib (Seaborn style)
