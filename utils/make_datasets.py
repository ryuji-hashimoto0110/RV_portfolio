import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

#---
# Calculate Realized Volatility
#---

def calculate_realized_volatility(prices_df,
                                  day_num,      # Number of days to record (Ex:365)
                                  record_times, # Number of times to record per day (Ex:78)
                                  stock_num
                                  ):
    dates      = prices_df.index
    dates_idx  = []
    previous_date = None
    for date in dates:
        if previous_date is not None:
            if previous_date.day != date.day:
                dates_idx.append(date)
        else:
            dates_idx.append(date)
        previous_date = date
    prices_arr = prices_df.to_numpy()
    prices_arr = np.reshape(prices_arr, (day_num, record_times, stock_num))
    return_arr = np.log(prices_arr[:,1:,:]) - np.log(prices_arr[:,:-1,:])
    return_arr = return_arr * return_arr
    rv_arr     = np.sum(return_arr, axis=1)
    rv_df      = pd.DataFrame(rv_arr, 
                              columns=prices_df.columns, index=dates_idx)
    prices_open_df = prices_df.loc[dates_idx]
    return prices_open_df, rv_df

#---
# Make PyTorch Dataset for LSTM_RV_PF model
#---

class LSTM_RV_PF_Dataset(Dataset):
    def __init__(self, 
                 prices_open_df, # Intraday price dataframe
                 rv_df,          # Realized volatility dataframe
                 dates,          
                 input_length
                 ):
        self.prices_open_df = prices_open_df
        self.rv_df          = rv_df
        self.input_length   = input_length
        self.dates = dates[input_length+1:-2]   

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, index):
        idx = self.prices_open_df.index.get_loc(self.dates[index])
        prices_open_arr = self.prices_open_df.iloc[idx:idx+2,:].to_numpy()
        prices_open_tensor = torch.from_numpy(prices_open_arr.astype(np.float32))
        rv_input_arr = self.rv_df.iloc[idx-self.input_length:idx+1,:].to_numpy()
        rv_input_tensor = torch.from_numpy(rv_input_arr.astype(np.float32))
        rv_target_arr = self.rv_df.iloc[idx+1,:].to_numpy()
        rv_target_tensor = torch.from_numpy(rv_target_arr.astype(np.float32))
        return prices_open_tensor, rv_input_tensor, rv_target_tensor

def make_pf_dataset(prices_open_df, rv_df, 
                    train_years, valid_years, input_length):
    train_dates = []
    valid_dates = []
    for date in prices_open_df.index:
        if date.year in train_years:
            train_dates.append(date)
        elif date.year in valid_years:
            valid_dates.append(date)
    train_dataset = LSTM_RV_PF_Dataset(prices_open_df, rv_df, 
                                       train_dates, input_length)
    valid_dataset = LSTM_RV_PF_Dataset(prices_open_df, rv_df, 
                                       valid_dates, input_length)
    return train_dataset, valid_dataset

