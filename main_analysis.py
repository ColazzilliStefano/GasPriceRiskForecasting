from src.forecaster import GasPriceForecaster

if __name__ == "__main__":
    # Ensure MPG_GAS.xlsx is in the /data folder
    model = GasPriceForecaster('data/MPG_GAS.xlsx')
    model.load_data()
    model.analyze_plots()
    model.perform_diagnostics()
    model.analyze_plots()
    model.identify_shocks()
    best_arima = model.find_best_model()
    model.audit_risk(best_arima)
    model.run_backtest()
