import datetime
import numpy as np
import pandas as pd
import pathlib
import torch
from torch import nn
from train import calculate_pv_after_commission
from utils.make_datasets import LSTM_RV_PF_Dataset

#---
# Backtest the trained model for multiple time periods.
#---

def backtest_multiple_period(model, device, initial_pv,
                             prices_open_df, rv_df, input_length,
                             dates,
                             commission_rate,
                             period_length, # (int) Length of one backtest period. (Ex: 30 days)
                             interval, # (int) Interval of the start points of the backtest periods. (Ex: 7 days)
                             data_path=None, 
                             csv_name='backtest_result.csv'
                             ):
    test_start_dates = []
    test_end_dates = []
    test_SRs = []
    test_rv_losses = []
    model.eval()
    model.to(device)
    for start_date_idx in range(0, len(dates)-period_length, interval):

        #---
        # Specify start_date and end_date (datetime)
        #---

        start_date = dates[start_date_idx]
        test_start_dates.append(start_date)
        end_date_idx = start_date_idx + period_length
        end_date = dates[end_date_idx]
        test_end_dates.append(end_date)
        backtest_dates = dates[start_date_idx:end_date_idx]

        #---
        # dataset, dataloader, criterion
        #---

        test_dataset = LSTM_RV_PF_Dataset(prices_open_df, rv_df, 
                                          backtest_dates, input_length)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=1,
                                                      shuffle=False)
        criterion = nn.MSELoss()

        #---
        # test
        #---

        test_rv_loss = 0.0
        test_R = 0.0
        test_R2 = 0.0
        pv = torch.tensor(float(initial_pv)).to(device)
        with torch.no_grad():
            for j, batch in enumerate(test_dataloader):
                prices = batch[0]
                rate_change = prices[0,1,:] / prices[0,0,:]
                rate_change = torch.cat([torch.ones(1), rate_change]).to(device)
                rv_x = batch[1].to(device)
                rv_t = batch[2].to(device)
                rv_y, w = model(rv_x)
                if j == 0:
                    w_ = torch.zeros(prices.shape[2]+1).to(device)
                    w_[0] = 1
                test_rv_loss += float(criterion(rv_y, rv_t)) / period_length
                cpv = calculate_pv_after_commission(w, w_, commission_rate)
                pv_ = cpv * pv * torch.sum(w * rate_change)
                r = torch.log(pv_) - torch.log(pv)
                test_R += float(r / period_length)
                test_R2 += float(r**2 / period_length)
                pv_ = pv
        test_var_R = test_R2 - test_R**2
        test_SR = test_R / np.sqrt(test_var_R)
        test_SRs.append(test_SR)
        test_rv_losses.append(test_rv_loss)
    #print(test_SRs)
    #print(test_rv_losses)
    test_df = pd.DataFrame(
        {
            'start date': test_start_dates,
            'end date': test_end_dates,
            'Sharp Ratio': test_SRs,
            'RV Loss': test_rv_losses
        }
    )
    if data_path is not None:
        csv_path = data_path / csv_name
        test_df.to_csv(csv_path)
    return test_df