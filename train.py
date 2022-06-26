import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

def calculate_pv_after_commission(w1, w0, commission_rate):
    mu = commission_rate * torch.sum(torch.abs(w1 - w0))
    return 1 - mu

def train_rv_pf(model, device, initial_pv,
                lam1, lam2, lam3,
                train_dataset, valid_dataset,
                best_save_path, last_save_path, 
                num_episode, commission_rate,
                seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    print(f'device:{device}')
    model.to(device)
    train_n = len(train_dataset)
    valid_n = len(valid_dataset)
    train_rv_losses = []
    valid_rv_losses = []
    train_SRs = []
    valid_SRs = []
    best_SR = -1e+05
    best_rv_loss = 1e+05

    #---
    # criterion, optimizer
    #---

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lam3)

    #---
    # dataloader
    #---

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=1,
                                                   shuffle=False)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=1,
                                                   shuffle=False)
    

    #---
    # episode
    #---

    for episode in range(num_episode):

        #---
        # train
        #---

        model.train()
        loss_rv = 0.0
        train_rv_loss = 0.0
        train_r = 0.0
        train_R = 0.0
        train_r2 = 0.0
        train_R2 = 0.0
        pv = torch.tensor(float(initial_pv)).to(device)
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_dataloader):
            prices = batch[0]
            rate_change = prices[0,1,:] / prices[0,0,:]
            rate_change = torch.cat([torch.ones(1), rate_change]).to(device)
            rv_x   = batch[1].to(device)
            rv_t   = batch[2].to(device)
            rv_y, w = model(rv_x)
            loss_rv += criterion(rv_y, rv_t)
            if i == 0:
                w_ = torch.zeros(prices.shape[2]+1)
                w_[0] = 1
            cpv = calculate_pv_after_commission(w, w_, commission_rate)
            pv_ = cpv * pv * torch.sum(w * rate_change)
            r = torch.log(pv_) - torch.log(pv)
            train_r += r / 100
            train_r2 += r**2 / 100
            prices = prices.detach()
            rate_change = rate_change.detach()
            if i % 100 == 99:
                train_var_r = train_r2 - train_r**2
                loss_sr = - (train_r / torch.sqrt(train_var_r))
                loss = lam1*loss_rv + lam2*loss_sr
                loss.backward()
                optimizer.step()
                train_r = train_r.detach()
                train_r2 = train_r2.detach()
                optimizer.zero_grad()
                train_rv_loss += float(loss_rv) / train_n
                loss_rv = 0
            train_R += float(r / train_n)
            train_R2 += float(r**2 / train_n)
            pv_ = pv_.detach()
            cpv = cpv.detach()
            pv  = pv.detach()
            pv  = pv_
            w_  = w_.detach()
            w   = w.detach()
            w_  = w
        train_var_R = train_R2 - train_R**2
        train_SR = train_R / np.sqrt(train_var_R)
        train_SRs.append(train_SR)
        train_rv_losses.append(train_rv_loss)

        #---
        # valid
        #---

        model.eval()
        valid_rv_loss = 0.0
        valid_r = 0.0
        valid_R = 0.0
        valid_r2 = 0.0
        valid_R2 = 0.0
        pv = torch.tensor(float(initial_pv)).to(device)
        with torch.no_grad():
            for j, batch in enumerate(valid_dataloader):
                prices = batch[0].to(device)
                rate_change = prices[0,1,:] / prices[0,0,:]
                rate_change = torch.cat([torch.ones(1), rate_change])
                rv_x   = batch[1].to(device)
                rv_t   = batch[2].to(device)
                rv_y, w = model(rv_x)
                if i == 0:
                    w_ = torch.zeros(prices.shape[2]+1)
                    w_[0] = 1
                valid_rv_loss += float(criterion(rv_y, rv_t)) / valid_n
                cpv = calculate_pv_after_commission(w, w_, commission_rate)
                pv_ = cpv * pv * torch.sum(w * rate_change)
                r = torch.log(pv_) - torch.log(pv)
                valid_R += float(r / valid_n)
                valid_R2 += float(r**2 / valid_n)
                pv = pv_
        valid_var_R = valid_R2 - valid_R**2
        valid_SR = valid_R / np.sqrt(valid_var_R)
        valid_SRs.append(valid_SR)
        valid_rv_losses.append(valid_rv_loss)

        #---
        # save model
        #---

        if (lam1*best_rv_loss - lam2*best_SR) > (lam1*valid_rv_loss - lam2*valid_SR):
            best_rv_loss = valid_rv_loss
            best_SR = valid_SR
            save_path = best_save_path
            print(f'episode{episode+1}/{num_episode}' +
                  f' [Sharp Ratio]tra:{train_SR:.4f} val:{valid_SR:.4f}' +
                  f' [RV loss]tra:{train_rv_loss:.8f} val:{valid_rv_loss:.8f}')
            print(f'model saving to >> {str(save_path)}')
        else:
            save_path = last_save_path
            print(f'model saving to >> {str(save_path)}')

        torch.save(
            {
                'num_episode': episode+1,
                'train_SRs': train_SRs,
                'valid_SRs': valid_SRs,
                'train_rv_losses': train_rv_losses,
                'valid_rv_losses': valid_rv_losses,
                'model_state_dict': model.state_dict(),
                'best_SR': best_SR,
                'best_rv_loss': best_rv_loss
            },
            str(save_path)
        )

    #---
    # plot
    #---

    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.grid()
    ax1.plot(train_SRs,
             marker='', color='red', label='train SR')
    ax1.plot(valid_SRs,
             marker='', color='blue', label='valid SR')
    ax1.legend(loc='upper left')
    ax1.set(xlabel='episode', ylabel='Sharp Ratio')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.grid()
    ax2.plot(train_rv_losses,
             marker='', color='orange', label='train RV Loss')
    ax2.plot(valid_rv_losses,
             marker='', color='green', label='valid RV Loss')
    ax2.legend(loc='upper right')
    ax2.set(xlabel='episode', ylabel='RV Loss')
    plt.show()

    return model