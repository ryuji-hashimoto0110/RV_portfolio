import datetime
import numpy as np
import pandas as pd
import pathlib

#---
# Generate mu and sigma randomly.
#---

def generate_mu_sigma(mu_upper, mu_lower, sigma_upper, sigma_lower, stock_num, 
                      seed=None):
    """
    mu_lower < mu^ij < mu_upper
    sigma_lower < sigma^ij < sigma_upper
    0 < sigma^ii < max(sigma_upper, abs(sigma_lower))
    """
    if seed is not None:
        np.random.seed(seed)
    mu    = (mu_upper - mu_lower) * np.random.rand(stock_num, 1) + mu_lower
    while True:
        sigma = (sigma_upper - sigma_lower) * np.random.rand(stock_num, stock_num) \
                + sigma_lower
        sigma = np.tril(sigma)
        sigma = sigma + sigma.T
        np.fill_diagonal(sigma, np.abs(np.diagonal(sigma/2)))
        eigen_values, _ = np.linalg.eig(sigma)
        if min(eigen_values) > 0:
            break
    return mu, sigma

#---
# Generate intraday price data and process it into pandas DataFrame.
#---

def generate_intraday_data(mu, sigma, 
                           day_num,       # Number of days to record (Ex:365)
                           record_times,  # Number of times to record per day (Ex:78)
                           delta_minutes, # How many minutes to record every (Ex:5)
                           stock_num,
                           initial_prices,
                           stock_names=None, # len(stock_names) == stock_num
                           start_year=2015, start_month=1, start_day=1,
                           start_hour=9, start_minute=35,
                           data_path=None,
                           csv_name='random_stock_data.csv',
                           seed=None
                           ):
    if seed is not None:
        np.random.seed(seed)
    log_prices_data = np.zeros((day_num*record_times, stock_num))
    log_prices    = np.log(initial_prices)
    dt            = 1 / record_times
    count         = 0
    datetime_list = []
    start_ydt     = datetime.datetime(start_year, start_month, start_day, 
                                      start_hour, start_minute) # year/date/time
    chol_sigma    = np.linalg.cholesky(sigma)
    diag_sigma    = np.diag(sigma).reshape(stock_num,1)
    for day in range(day_num):
        ydt = start_ydt + datetime.timedelta(days=day)
        for time in range(record_times):
            dw     = np.sqrt(dt) * np.random.randn(stock_num, 1)
            dS_log = (mu - diag_sigma/2)*dt + chol_sigma@dw
            log_prices += dS_log
            log_prices_data[count,:] = log_prices.reshape(stock_num)
            count  += 1
            datetime_list.append(ydt)
            ydt += datetime.timedelta(minutes=delta_minutes)
    if stock_names is None:
        stock_names = [f'stock{i}' for i in range(1,stock_num+1)]
    prices_data = np.exp(log_prices_data)
    prices_df   = pd.DataFrame(prices_data, 
                               columns=stock_names, index=datetime_list)
    if data_path is not None:
        csv_path = data_path / csv_name
        prices_df.to_csv(csv_path)
    return prices_df