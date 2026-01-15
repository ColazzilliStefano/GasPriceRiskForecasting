%%writefile README.md
# Energy Market Risk & Forecasting: MPG-GAS Analysis
### *Quantitative Framework for Hybrid ARIMA-EGARCH Modeling with Expected Shortfall*

This repository presents an econometric pipeline for the **MPG-GAS** price series. The core objective is to move beyond standard Gaussian assumptions to model the "Fat-Tail" reality of energy commodities.

---

## Summary
Traditional models often fail in energy markets because they assume "Normal" volatility. Analysis reveals a **Kurtosis of 16.04**, indicating that extreme events are far more frequent than a Gaussian bell curve suggests.

This framework implements a **SARIMAX-EGARCH** model to solve:
1. **Asymmetric Volatility:** In gas markets, price spikes (fear of shortage) generate more volatility than price drops—a phenomenon known as the **Inverse Leverage Effect**.
2. **Structural Breaks:** 11 structural shocks were identified and neutralized to ensure the convergence of the Maximum Likelihood Estimation (MLE).
3. **Tail Risk Management:** We use **Expected Shortfall (ES)** at 99% to quantify the average loss in worst-case scenarios, providing a safer buffer than simple Value-at-Risk (VaR).

---

## Diagnostics & Results

### 1. Exploratory Data Analysis (EDA)
Analysis of the price evolution, returns distribution, and normality check.
![EDA Plots](results/eda_plots.png)

### 2. Volatility Clustering Analysis
This chart highlights that variance is not constant over time: periods of high volatility tend to cluster together, justifying the use of a GARCH-family model.
![Volatility Clustering](results/volatility_clustering.png)

### 3. Structural Breaks & Outlier Mapping
A map of extreme events (shocks) that exceed the 5.0 sigma threshold. These events are treated as exogenous to prevent biasing the model parameters.
![Shock Mapping](results/shock_mapping.png)

### 4. Tail Risk Audit (Gaussian vs. Reality)
Comparison between theoretical Gaussian tails and the actual Student-t distribution (nu=4.18). The log-scale confirms the presence of significant tail risk.
![Tail Risk Audit](results/tail_risk_audit.png)

### 5. News Impact Curve
This curve demonstrates the volatility asymmetry: positive shocks (price increases) have a higher impact on future volatility than negative shocks.
![News Impact Curve](results/news_impact_curve.png)

### 6. Final Forecast Performance & Risk Buffer
Comparison between Realized Prices, Point Forecasts, and the 99% Expected Shortfall risk buffer.
![Forecast Performance](results/forecast_performance.png)

---

## Model Architecture & Technical Deep-Dive

### 1. Mean Modeling: SARIMAX(0, 1, 3)
* **Order Selection:** The model uses **MA(3)**, meaning today's price is corrected by the errors (shocks) of the previous 3 days. This reflects the "short memory" of supply-demand imbalances in gas spot markets.
* **Exogenous Component:** The **Shock_Exog** coefficient (**5.26**) is highly significant ($P < 0.001$), absorbing massive price swings that would otherwise bias the entire model. The value of the coefficient means that "in a situation of shock prices fall or rise by 5.26%"

### 2. Volatility Audit: EGARCH(1, 1)
* **Why EGARCH?** Unlike standard GARCH, EGARCH models the *logarithm* of volatility, allowing for an asymmetric response to news.
* **Distribution (Student's t):** With **nu = 4.18**, the model confirms a **Heavy-Tail Regime**. Since $\nu < 5$, Gaussian-based risk modeling is mathematically invalidated.
* **Persistence ($\beta = 0.946$):** Volatility is highly persistent; once the market becomes nervous, it stays unstable for a long period (Volatility Clustering).

---

## Limitations & Model Risks
1. **Static Parameters:** The (0,1,3) structure is identified once. Fundamental market regime shifts might require periodic re-identification.
2. **Technical Limits:** The model does not see "outside" factors like weather forecasts or geopolitical news; it reacts solely to price patterns.
3. **Refit Trade-off:** To optimize speed, we use a **Turbo Refit** strategy (every 50 days). While efficient, it may slightly lag during hyper-volatile intra-month transitions.

---

## Backtesting Results
- **MAPE (Accuracy):** 4.50% (Excellent for high-volatility assets).
- **ES Violations:** 4 / 1222 (Target Breach Rate < 1%). The model successfully captured 99.67% of all price movements within its risk buffer.

---

## Installation
```bash
pip install pandas numpy matplotlib statsmodels arch pmdarima
