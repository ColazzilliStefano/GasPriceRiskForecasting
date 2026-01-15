
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
import pmdarima as pm
import warnings
import os


#Professional Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
warnings.filterwarnings('ignore')



class GasPriceForecaster:
    """
    QUANTITATIVE RISK ENGINE (SARIMAX-EGARCH)
    
    Advanced Quantitative Framework for Gas Price Risk Analysis.
    
    Methodology:
        1. Mean Dynamics: SARIMAX with Structural Break detection.
        2. Volatility Dynamics: EGARCH(1,1) to capture asymmetry (Leverage Effect).
        3. Tail Risk: Extreme Value Theory using Student-t distribution (Fat Tails).
    """

    def __init__(self, file_path, threshold=4.0, window_size=1000):
        self.file_path = file_path
        self.threshold = threshold # Sigma threshold for shock detection
        self.window_size = window_size
        self.df = None
        self.best_order = None
        self.results_df = None
        self.final_vol_model = None # To store model for density plotting


    def load_data(self):
        """Data Ingestion & Transformation."""
        print(f"LOADING DATA: {self.file_path}")
        try:
            raw_data = pd.read_excel(self.file_path)
            clean_data = raw_data.dropna()
            
            # Logic to find price column (flexible)
            if 'vp' in clean_data.columns:
                prices = clean_data['vp'].values
            else:
                prices = clean_data.iloc[:, 1].values
            
            dates = pd.to_datetime(clean_data.iloc[:, 0].values)
            
            self.df = pd.DataFrame({'Price': prices, 'Log_Price': np.log(prices)}, index=dates)
            self.df['Returns'] = self.df['Log_Price'].diff() * 100
            self.df = self.df.dropna()

            # Stats Output
            rets = self.df['Returns']
            print(f"    Observation Count: {len(rets)}")
            print(f"    Mean Return: {rets.mean():.4f}%")
            print(f"    Volatility (Daily): {rets.std():.4f}%")
            print(f"    Kurtosis: {rets.kurtosis() + 3:.4f} (Target > 3 for Fat Tails)")
            print("DATA LOADED.\n")
            
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            raise

    def perform_diagnostics(self):
        """Statistical Health Check."""
        print("DIAGNOSTICS")
        
        # Stationarity
        adf_res = adfuller(self.df['Returns'])
        print(f"    ADF p-value: {adf_res[1]:.4e} " + 
              ("(Stationary)" if adf_res[1] < 0.05 else "(NON-STATIONARY - WARNING)"))

        # Normality
        jb_stat, jb_p = stats.jarque_bera(self.df['Returns'])
        print(f"    Jarque-Bera p-value: {jb_p:.4e} " + 
              ("(Non-Normal / Fat Tails CONFIRMED)" if jb_p < 0.05 else "(Normal Distribution)"))
        print("-" * 50 + "\n")


    def perform_eda(self):
        """Exploratory Data Analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        
        # Price
        axes[0,0].plot(self.df['Price'], color='#2C3E50', lw=1)
        axes[0,0].set_title('Asset Price History')
        
        # Returns
        axes[0,1].plot(self.df['Returns'], color='#7F8C8D', lw=0.8, alpha=0.8)
        axes[0,1].set_title('Daily Log-Returns')
        
        # Distribution
        axes[1,0].hist(self.df['Returns'], bins=80, density=True, alpha=0.6, color='#27AE60')
        mu, std = stats.norm.fit(self.df['Returns'])
        x = np.linspace(self.df['Returns'].min(), self.df['Returns'].max(), 100)
        axes[1,0].plot(x, stats.norm.pdf(x, mu, std), 'r--', lw=2, label='Normal Dist.')
        axes[1,0].set_title('Empirical Distribution vs Normal')
        axes[1,0].legend()
        
        # QQ Plot
        stats.probplot(self.df['Returns'], dist="norm", plot=axes[1,1])
        axes[1,1].set_title('Q-Q Plot (Tail Deviation Check)')
        
        plt.tight_layout()
        plt.savefig('results/01_eda_plots.png')
        plt.show()


    def identify_shocks(self):
        """Exogenous Shock Mapping (Structural Breaks)."""
        mean_ret = self.df['Returns'].mean()
        std_ret = self.df['Returns'].std()
        upper = mean_ret + self.threshold * std_ret
        lower = mean_ret - self.threshold * std_ret
        shock_mask = (self.df['Returns'] > upper) | (self.df['Returns'] < lower)
        shock_indices = self.df[shock_mask].index
        
        self.df['Shock_Exog'] = 0
        self.df.loc[shock_indices, 'Shock_Exog'] = 1 # Dummy variable

        print(f"SHOCK DETECTION (Threshold: {self.threshold} sigma)")
        print(f"    Identified {len(shock_indices)} structural breaks.")

        plt.figure(figsize=(12, 4))
        plt.plot(self.df['Returns'], color='gray', alpha=0.4, label='Returns')
        plt.scatter(shock_indices, self.df.loc[shock_indices, 'Returns'], color='#C0392B', s=20, label='Exogenous Shock')
        plt.axhline(upper, color='black', ls=':', alpha=0.5)
        plt.axhline(lower, color='black', ls=':', alpha=0.5)
        plt.title("Regime Shift & Outlier Mapping")
        plt.legend()
        plt.savefig('results/02_shock_mapping.png')
        plt.show()


    def find_best_model(self):
        """Auto-ARIMA Selection."""
        print("MODEL SELECTION (Auto-ARIMA)")
        # Using BIC to penalize complexity (parsimony principle)
        model = pm.auto_arima(self.df['Log_Price'] * 100, 
                              X=self.df[['Shock_Exog']],
                              d=1, 
                              seasonal=False, 
                              stepwise=True,
                              suppress_warnings=True,
                              information_criterion='bic')
        
        self.best_order = model.order
        print(f"    Optimal Order Identified: ARIMA{self.best_order}")
        
        # Fit logic for summary
        full_model = SARIMAX(self.df['Log_Price'] * 100, 
                             exog=self.df[['Shock_Exog']],
                             order=self.best_order).fit(disp=False)
        return full_model


    def audit_risk_tails(self, arima_model):
        """
        Volatility Modeling & Tail Audit.
        Visualizing only TRUE fat tails.
        """
        print("\nVOLATILITY AUDIT (EGARCH)")
        
        # EGARCH to capture asymmetry (bad news = higher vol)
        res_vol = arch_model(arima_model.resid, vol='EGARCH', p=1, o=1, q=1, dist='t', mean='Zero').fit(disp='off')
        self.final_vol_model = res_vol
        print(f"    EGARCH Parameters: Omega={res_vol.params['omega']:.3f}, Alpha={res_vol.params['alpha[1]']:.3f}")
        print(f"    Shape Parameter (nu): {res_vol.params['nu']:.2f} (Low nu = High Tail Risk)")

        # Plotting Tail Audit
        nu = res_vol.params['nu']
        x = np.linspace(-10, 10, 1000)
        pdf_student = stats.t.pdf(x, df=nu)
        pdf_norm = stats.norm.pdf(x)

        plt.figure(figsize=(12, 6))
        plt.plot(x, pdf_norm, label='Theoretical Normal', linestyle='--', color='blue', alpha=0.6)
        plt.plot(x, pdf_student, label=f'Estimated Reality (Student-t, nu={nu:.1f})', color='#C0392B', linewidth=2.5)
        tail_threshold = 2.5 # Only highlight deviations beyond 2.5 sigma
        plt.fill_between(x, pdf_student, pdf_norm,
                         where=((pdf_student > pdf_norm) & (np.abs(x) > tail_threshold)),
                         color='#C0392B', alpha=0.3, label='Unmodeled Fat Tail Risk')
        plt.title("Tail Risk Audit: Where the Normal Distribution Fails")
        plt.yscale('log') # Log scale is crucial for tails
        plt.ylim(1e-4, 1)
        plt.ylabel("Probability Density (Log Scale)")
        plt.xlabel("Standard Deviations")
        plt.legend()
        plt.savefig('results/03_tail_audit.png')
        plt.show()


    def run_backtest(self, refit_every=50):
        """
        Walk-Forward Validation.
        FOCUS: Downside Risk (Left Tail).
        """
        log_prices = self.df['Log_Price'].values * 100
        exog_shocks = self.df[['Shock_Exog']].values
        n_test = len(log_prices) - self.window_size
        results = []

        print(f"BACKTESTING STARTED ({n_test} days)")
        print(f"    Strategy: Dynamic Refit every {refit_every} days")

        # Initial Fit
        cur_model = SARIMAX(log_prices[0:self.window_size], exog=exog_shocks[0:self.window_size], order=self.best_order).fit(disp=False)
        cur_vol = arch_model(cur_model.resid, vol='EGARCH', p=1, o=1, q=1, dist='t', mean='Zero').fit(disp='off')

        for i in range(n_test):
            # Rolling Window Data
            y_train = log_prices[i : self.window_size + i]
            x_train = exog_shocks[i : self.window_size + i]
            x_next = exog_shocks[self.window_size + i : self.window_size + i + 1]

            # Refit or Filter
            if i % refit_every == 0 and i > 0:
                cur_model = SARIMAX(y_train, exog=x_train, order=self.best_order).fit(disp=False, maxiter=50)
                cur_vol = arch_model(cur_model.resid, vol='EGARCH', p=1, o=1, q=1, dist='t', mean='Zero').fit(disp='off')
            else:
                cur_model = SARIMAX(y_train, exog=x_train, order=self.best_order).filter(cur_model.params)

            # 1. Forecast Mean
            fc_mean = cur_model.forecast(steps=1, exog=x_next)[0]
            
            # 2. Forecast Volatility
            vol_res = cur_vol.forecast(horizon=1)
            sigma_next = np.sqrt(vol_res.variance.values[-1,0])
            nu = cur_vol.params['nu']
            
            # 3. Calculate DOWNSIDE ES (Left Tail)
            alpha = 0.01 # 99% Confidence
            t_q = stats.t.ppf(alpha, df=nu) # This is negative (e.g., -2.5)
            
            # Expected Shortfall Magnitude Formula for Student-t
            es_std_devs = (stats.t.pdf(t_q, df=nu) / alpha) * ((nu + t_q**2) / (nu - 1))
            
            # Subtract from mean because it's a loss
            fc_es_limit = fc_mean - (es_std_devs * sigma_next)

            results.append({
                'Realized': np.exp(log_prices[self.window_size+i]/100),
                'Forecast': np.exp(fc_mean/100), 
                'ES_99': np.exp(fc_es_limit/100) # The Floor Price
            })

        self.results_df = pd.DataFrame(results, index=self.df.index[self.window_size:])
        self._plot_backtest_results()


    def _plot_backtest_results(self):

        """Plotting Forecast vs Downside Risk."""

        # Metrics
        mape = np.mean(np.abs((self.results_df['Realized'] - self.results_df['Forecast']) / self.results_df['Realized'])) * 100

        # Violation: Price fell below the ES floor
        violations = np.sum(self.results_df['Realized'] < self.results_df['ES_99'])
        violation_pct = (violations / len(self.results_df)) * 100

        print(f"BACKTEST COMPLETE")
        print(f"    MAPE (Accuracy): {mape:.2f}%")
        print(f"    Risk Breaches (Realized < ES): {violations} ({violation_pct:.2f}%)")
        print(f"    *Note: For ES 99%, we expect ~1% breaches. If <1%, model is conservative.")

        plt.figure(figsize=(15, 7))
        plt.plot(self.results_df['Realized'], label='Realized Price', color='black', alpha=0.4, lw=1)
        plt.plot(self.results_df['Forecast'], label='Point Forecast (t+1)', color='#2980B9', ls='--', lw=1.5)
        
        # Correctly filling the downside area
        plt.fill_between(self.results_df.index, 
                         self.results_df['ES_99'], 
                         self.results_df['Forecast'], 
                         color='#C0392B', alpha=0.25, label='Downside Risk Buffer (ES 99%)')
        
        plt.title(f"Risk Model Validation: Forecast vs Downside Limit\nViolations: {violation_pct:.2f}% (Target ~1%)")
        plt.legend(loc='upper left')
        plt.savefig('results/04_backtest_performance.png')
        plt.show()


    def visualize_risk_density_snapshot(self):
        """
        Visualizes the Probability Density for the NEXT trading day.
        Shows the 'Cross-Section' of risk with explicit VaR and ES levels.
        """
        print("GENERATING T+1 RISK SNAPSHOT")
        
        # 1. Refit on ALL data to get T+1 parameters
        log_prices = self.df['Log_Price'].values * 100
        exog_shocks = self.df[['Shock_Exog']].values
        model = SARIMAX(log_prices, exog=exog_shocks, order=self.best_order).fit(disp=False)
        vol_model = arch_model(model.resid, vol='EGARCH', p=1, o=1, q=1, dist='t', mean='Zero').fit(disp='off')
        
        # 2. Get Forecast Params
        fc_vol = vol_model.forecast(horizon=1)
        sigma_t = np.sqrt(fc_vol.variance.values[-1, 0])
        nu = vol_model.params['nu']
        mu_t = 0 # Residual mean is assumed 0 in GARCH
        
        # 3. Setup Plot
        x = np.linspace(mu_t - 6*sigma_t, mu_t + 6*sigma_t, 1000)
        pdf = stats.t.pdf(x, df=nu, loc=mu_t, scale=sigma_t)
        
        # 4. Calculate Risk Metrics (VaR & ES)
        alpha = 0.01 # 99% confidence
        
        # A. Value at Risk (The Threshold)
        var_threshold = stats.t.ppf(alpha, df=nu, loc=mu_t, scale=sigma_t)
        
        # B. Expected Shortfall (The Average Tail Loss)
        t_q = stats.t.ppf(alpha, df=nu) 
        es_std_magnitude = (stats.t.pdf(t_q, df=nu) / alpha) * ((nu + t_q**2) / (nu - 1))  #ES t-tudent formula
        es_threshold = mu_t - (es_std_magnitude * sigma_t)  #scaled on actual volatility
        
        # 5. Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot PDF
        ax.plot(x, pdf, color='#2C3E50', lw=2, label=f'T+1 Forecast Dist (Student-t, nu={nu:.2f})')
        
        # Color Tail 
        x_tail = x[x <= var_threshold]
        y_tail = pdf[x <= var_threshold]
        ax.fill_between(x_tail, y_tail, color='#E74C3C', alpha=0.3, label='Tail Loss Area')
        ax.axvline(var_threshold, color='#C0392B', linestyle='--', linewidth=1.5, label=f'VaR 99%: {var_threshold:.2f}%')  # VaR Line 
        ax.axvline(es_threshold, color='#922B21', linestyle='-', linewidth=2.5, label=f'ES 99% (Avg. Crash): {es_threshold:.2f}%')  # ES Line 
        
        # Annotations
        ax.set_title("Forward Looking Risk: Next Day Return Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("Potential Daily Return (%)")
        ax.set_ylabel("Probability Density")
        

        info_text = (f"Volatility (Sigma): {sigma_t:.2f}%\n"
                     f"Tail Shape (Nu): {nu:.2f}\n"
                     f"VaR (99%): {var_threshold:.2f}%\n"
                     f"ES (99%):  {es_threshold:.2f}%")
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.95, info_text, transform=ax.transAxes, fontsize=11, 
                verticalalignment='top', bbox=props, fontfamily='monospace')
        
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('results/05_risk_snapshot.png')
        plt.show()


# MAIN EXECUTION

if __name__ == "__main__":

    # Create results folder
    if not os.path.exists('results'):
        os.makedirs('results')
        
    # Initialize
    forecaster = GasPriceForecaster('MPG_GAS.xlsx', threshold=5.0)
    
    # Run Pipeline
    forecaster.load_data()
    forecaster.perform_diagnostics()
    forecaster.perform_eda()
    forecaster.identify_shocks()
    
    best_arima = forecaster.find_best_model()
    forecaster.audit_risk_tails(best_arima)
    forecaster.run_backtest(refit_every=50)
    forecaster.visualize_risk_density_snapshot()
