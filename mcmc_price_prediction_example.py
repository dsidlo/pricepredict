import pandas as pd
import numpy as np
import yfinance as yf
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pricepredict import PricePredict
from datetime import datetime, timedelta

# %matplotlib
# inline

def mcmc_process_data(data, plot_trace=False):
    """
    MCMC:  Markov Chain Monte Carlo
    This function preprocesses the data and defines the Bayesian AR(1) model.
    """
    # Step 2: Preprocess Data
    data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)

    # Step 3: Define Bayesian AR(1) Model
    # AR(1) Autoregressive Model of order 1
    with pm.Model() as model:
        # Priors for the parameters
        mu = pm.Normal('mu', mu=0, sigma=0.1)
        phi = pm.Normal('phi', mu=0, sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=0.1)

        # Observed data
        Y_t = data['LogReturn'].values[1:]  # Current returns
        Y_tm1 = data['LogReturn'].values[:-1]  # Previous returns

        # Expected value of the current return
        mu_Y_t = mu + phi * (Y_tm1 - mu)

        # Likelihood: Model the current return given the previous return
        Y_obs = pm.Normal('Y_obs', mu=mu_Y_t, sigma=sigma, observed=Y_t)

    # Step 4: Sample from Posterior
    with model:
        trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

    if plot_trace:
        # Check Trace Plots
        az.plot_trace(trace)
        plt.show()

    return data, Y_t, Y_tm1, Y_obs, mu_Y_t, trace

# Step 5: Forecast Future Returns and Prices
def mcmc_forecast_returns(trace, last_return, steps=30):
    mu_samples = trace.posterior['mu'].values.flatten()
    phi_samples = trace.posterior['phi'].values.flatten()
    sigma_samples = trace.posterior['sigma'].values.flatten()

    n_samples = len(mu_samples)
    forecasts = np.zeros((n_samples, steps))

    for i in range(n_samples):
        mu_i = mu_samples[i]
        phi_i = phi_samples[i]
        sigma_i = sigma_samples[i]
        ret = last_return
        for t in range(steps):
            eps = np.random.normal(0, sigma_i)
            ret = mu_i + phi_i * (ret - mu_i) + eps
            forecasts[i, t] = ret
    return forecasts

if __name__ == '__main__':
    # Step 1: Load Data
    symbol = '^GSPC'
    end_date = '2025-01-18'
    # end_date = datetime.now().strftime('%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_date = (end_dt - timedelta(days=30)).strftime('%Y-%m-%d')
    ppo = PricePredict(symbol, period=PricePredict.PeriodDaily)
    data, features = ppo.fetch_data_yahoo(date_start=start_date, date_end=end_date)
    data, Y_t, Y_tm1, Y_obs, mu_Y_t, trace = mcmc_process_data(data)
    last_return = data['LogReturn'].values[-1]
    forecast_steps = 7
    forecasts = mcmc_forecast_returns(trace, last_return, steps=forecast_steps)

    # Convert to Prices
    last_price = data['Close'].values[-1]
    price_forecasts = last_price * np.exp(np.cumsum(forecasts, axis=1))
    median_forecast = np.median(price_forecasts, axis=0)
   # hpd_interval = az.hdi(price_forecasts, hdi_prob=0.94)
    hpd_interval = az.hdi(price_forecasts)

    # Step 6: Visualize Forecast
    forecast_dates = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq='B')[1:]
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-100:], data['Close'].values[-100:], label='Historical Prices')
    plt.plot(forecast_dates, median_forecast, label='Median Forecast')
    plt.fill_between(forecast_dates, hpd_interval[:, 0], hpd_interval[:, 1], color='gray', alpha=0.5,
                     label='94% Credible Interval')
    plt.title('Forecasted Stock Prices for {}'.format(symbol))
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

