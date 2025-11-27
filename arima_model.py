from statsmodels.tsa.arima.model import ARIMA

def run_arima(series, order=(2,1,2), steps=30):
    model = ARIMA(series, order=order)
    fit = model.fit()
    forecast = fit.forecast(steps=steps)
    return forecast
