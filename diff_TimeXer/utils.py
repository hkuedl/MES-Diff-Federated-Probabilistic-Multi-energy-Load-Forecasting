import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import copy
from Data_loader import *
import torch.nn as nn
import properscoring as ps
from scipy.stats import norm
from scipy.stats import cauchy
from matplotlib import rcParams
from typing import List, Union, Optional
import os

def forward_diffusion(m_0, n_0,alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    m_result = []
    n_result = []
    for t in range(n_steps):
        batch_size = m_0.size(0)
        t = torch.tensor(t).repeat(batch_size)
        a = alphas_bar_sqrt[t]
        aml = one_minus_alphas_bar_sqrt[t]
        e = torch.randn_like(m_0)
        m = m_0 * a.unsqueeze(-1) + e * aml.unsqueeze(-1)
        n = n_0 * a.unsqueeze(-1) + e * aml.unsqueeze(-1)
        m_result.append(m)
        n_result.append(n)
    m_stack = torch.stack(m_result, dim=0)
    n_stack = torch.stack(n_result, dim=0)
    return m_stack.permute(1,0,2),n_stack.permute(1,0,2)

def cosine_schedule(n_steps):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = n_steps + 1  # Number of steps is one more than timesteps
    x = torch.linspace(0, n_steps, steps)  # Create a linear space from 0 to timesteps
    # Calculate the cumulative product of alphas using the cosine schedule formula
    alphas_cumprod = (
        torch.cos(((x / n_steps) + 8e-3) / (1 + 8e-3) * torch.pi * 0.5) ** 2
    )
    alphas_cumprod = (
        alphas_cumprod / alphas_cumprod[0]
    )  # Normalize by the first element
    betas = 1 - (
        alphas_cumprod[1:] / alphas_cumprod[:-1]
    )  # Calculate betas from alphas
    betas = torch.clip(betas, 1e-4, 1e-1)
    alphas = 1 - betas
    alphas_bars = torch.cumprod(alphas, dim=0)
    # alphas_bars = torch.clip(alphas_bars, 0.01, 0.99)
    return {
        "alpha_bars": alphas_bars.float(),
        "beta_bars": None,
        "alphas": alphas.float(),
        "betas": betas.float(),
    }

def plot_train_loss(train_loss, val_loss=None):
    plt.plot(train_loss, label='Train Loss')
    if val_loss is not None:
        plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()


def cauchy_quantile(x0, gamma, p):
    return x0 + gamma * np.tan(np.pi * (p - 0.5))

def winkler_score(y_true, q_lower, q_upper, alpha):
    delta = q_upper - q_lower
    score = np.where(
        y_true < q_lower,
        delta + (2 / alpha) * (q_lower - y_true),
        np.where(
            y_true > q_upper,
            delta + (2 / alpha) * (y_true - q_upper),
            delta
        )
    )
    return np.mean(score)  # 返回平均 Winkler Score

def compute_winkler_score(y_true, x0, gamma):
    quantiles = {
        "q12_5": cauchy_quantile(x0, gamma, 0.125),  # 12.5% 分位数
        "q25": cauchy_quantile(x0, gamma, 0.25),     # 25% 分位数
        "q37_5": cauchy_quantile(x0, gamma, 0.375),  # 37.5% 分位数
        "q75": cauchy_quantile(x0, gamma, 0.75),     # 75% 分位数
        "q87_5": cauchy_quantile(x0, gamma, 0.875)   # 87.5% 分位数
    }
    
    # 计算 Winkler Score
    ws_25 = winkler_score(y_true, quantiles["q12_5"], quantiles["q37_5"], alpha=0.75)  # 25% 区间
    ws_50 = winkler_score(y_true, quantiles["q25"], quantiles["q75"], alpha=0.5)       # 50% 区间
    ws_75 = winkler_score(y_true, quantiles["q12_5"], quantiles["q87_5"], alpha=0.25)  # 75% 区间
    
    return ws_25, ws_50, ws_75

def compute_winkler_score_2(y_true, x0, gamma):
    quantiles = {
        "q10": cauchy_quantile(x0, gamma, 0.10),    # 10% 分位数 (对应80%置信区间下限)
        "q20": cauchy_quantile(x0, gamma, 0.20),    # 20% 分位数 (对应60%置信区间下限)
        "q40": cauchy_quantile(x0, gamma, 0.40),    # 40% 分位数 (对应20%置信区间下限)
        "q60": cauchy_quantile(x0, gamma, 0.60),    # 60% 分位数 (对应20%置信区间上限)
        "q80": cauchy_quantile(x0, gamma, 0.80),    # 80% 分位数 (对应60%置信区间上限)
        "q90": cauchy_quantile(x0, gamma, 0.90)     # 90% 分位数 (对应80%置信区间上限)
    }
    
    ws_20 = winkler_score(y_true, quantiles["q40"], quantiles["q60"], alpha=0.80)  # 20% 区间 (40%-60%)
    ws_60 = winkler_score(y_true, quantiles["q20"], quantiles["q80"], alpha=0.40)  # 60% 区间 (20%-80%)
    ws_80 = winkler_score(y_true, quantiles["q10"], quantiles["q90"], alpha=0.20)  # 80% 区间 (10%-90%)
    
    return ws_20, ws_60, ws_80

def metric(mu, sigma, actual):
    mape = MAPE(actual, mu)
    rmse = RMSE(actual, mu)
    mae = MAE(actual, mu)
    quantiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    pinball_loss = eval_pinball(actual, mu, sigma, quantiles)
    crps = CRPS(actual, mu, sigma)
    winkler_score_25, winkler_score_50, winkler_score_75 = compute_winkler_score(actual, mu, sigma)
    return mape, rmse, mae, pinball_loss, crps, winkler_score_25, winkler_score_50, winkler_score_75

    
class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
        
class cauchy_likelihood(nn.Module):
    def __init__(self):
        super(cauchy_likelihood, self).__init__()
    def forward(self, label, mu, sigma):
        distribution = torch.distributions.cauchy.Cauchy(mu,sigma)
        loss = distribution.log_prob(label)
        return -torch.mean(loss)

class NoiseInjectionModule:
    def __init__(self, 
                 gaussian_std: float = 0.1,
                 offset_scale: float = 0.2):
        self.gaussian_std = gaussian_std
        self.offset_scale = offset_scale

    def offset_noise(self, data, selected_indices):
        # 确保在相同设备上计算
        means = torch.mean(data, dim=1, keepdim=True).to(data.device)
        return self.offset_scale * means[selected_indices]

    def gaussian_noise(self, data, selected_indices):
        noise_shape = data[selected_indices].shape
        noise = torch.from_numpy(np.random.normal(loc=0.0, scale=self.gaussian_std, size=noise_shape)).float()
        return noise

    def inject_noise(self,
                   data: torch.Tensor,
                   noise_type,
                   ratio):

        noisy_data = data.clone()
        num_samples = data.size(0)
        num_to_select = int(num_samples * ratio)
        selected_indices = np.random.choice(num_samples, num_to_select, replace=False)

        if noise_type == 'reset':
            noisy_data[selected_indices] = 0
        else:
            if noise_type == 'offset':
                offset = self.offset_noise(data, selected_indices)  # 转换噪声类型
                noisy_data[selected_indices] += offset
            elif noise_type == 'gaussian':
                noise = self.gaussian_noise(data, selected_indices)  # 转换噪声类型
                noisy_data[selected_indices] += noise
            
        return noisy_data
    

class DiffDataset(Dataset):
    def __init__(self, hidden, ex, label):
        self.hidden = torch.Tensor(hidden)
        self.ex = torch.Tensor(ex)
        self.label = torch.Tensor(label)

    def __len__(self):
        return len(self.hidden)

    def __getitem__(self, idx):
        hidden_idx = self.hidden[idx]
        ex_idx = self.ex[idx]
        label_idx = self.label[idx]
        return hidden_idx,ex_idx,label_idx
    
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
    

def eval_pinball(true,mu,sigma,quantiles):
    losses = []
    true = torch.Tensor(true)
    mu = torch.Tensor(mu)
    sigma = torch.Tensor(sigma)
    for k in range(len(true)):
        ppf_list = cauchy.ppf(q = quantiles,loc = mu[k],scale = sigma[k])
        # ppf_list = norm.ppf(q=quantiles, loc=mu[k], scale=sigma[k])
        for i, q in enumerate(quantiles):
            errors = true[k] - ppf_list[i]
            losses.append(torch.max((q - 1) * errors, q * errors))
    return(np.mean(losses))

def CRPS(true,mu,sigma):
    # distribution = torch.distributions.normal.Normal(torch.Tensor(mu),torch.Tensor(sigma))
    distribution = torch.distributions.cauchy.Cauchy(torch.Tensor(mu),torch.Tensor(sigma))
    sample = distribution.sample([100])
    print(sample.shape)
    return(np.mean(ps.crps_ensemble(true,sample.permute(1,0))))
    #return(np.mean(ps.crps_gaussian(true,mu,sigma)))

class PinballLoss:
    def __init__(self, quantiles: torch.Tensor, reduction: str = 'none'):
        self.quantiles = quantiles
        self.reduction = reduction

    def __call__(self, predictions, actuals):
        if not torch.is_tensor(predictions):
            predictions = torch.tensor(predictions, dtype=torch.float32)
        
        if not torch.is_tensor(actuals):
            actuals = torch.tensor(actuals, dtype=torch.float32)

        losses = []

        for quantile in self.quantiles:
            q_predictions = torch.quantile(predictions, quantile, dim=1)

            error = actuals - q_predictions
            
            first_term = quantile * error
            second_term = (quantile - 1) * error
            loss = torch.maximum(first_term, second_term)

            if self.reduction == 'sum':
                loss = loss.sum()
            elif self.reduction == 'mean':
                loss = loss.mean()

            losses.append(loss)

        return np.mean(losses)


def get_data(args,data_train,data_val,data_test,flag='train',load='electricity'):
    if flag=='train':
        shuffle_flag=True
        drop_last=False
    elif flag == 'val':
        shuffle_flag=True
        drop_last=False
    elif flag == 'test':
        shuffle_flag=False
        drop_last=False
    data_set = CreateDataset(data_train=data_train,data_val=data_val,data_test=data_test,load=load,flag=flag,seq_len=args.seq_len,label_len=args.label_len)
    
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    
    return data_set,data_loader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def MAPE(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    MAPE=np.mean(abs((y_actual-y_predicted)/y_actual))
    return MAPE

def R2(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    R2 = 1 - np.sum(np.square(y_actual-y_predicted)) / np.sum(np.square(y_actual-np.mean(y_actual)))
    return R2

def RMSE(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    RMSE = np.sqrt(np.mean(np.square(y_actual-y_predicted)))
    return RMSE

def MSE(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    MSE = np.mean(np.square(y_actual-y_predicted))
    return MSE

def MAE(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    return np.mean(np.abs(y_actual-y_predicted))

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def Normalize(data_train,data_test,features_name,cal=True):
    if cal:
        normalize_data_train=data_train[features_name]
        normalize_data_test=data_test[features_name]
        for feature in features_name:
            max_values=normalize_data_train[feature].max()
            min_values=normalize_data_train[feature].min()
            normalize_data_train[feature]=(normalize_data_train[feature]-min_values)/(max_values-min_values)
            normalize_data_test[feature]=(normalize_data_test[feature]-min_values)/(max_values-min_values)
        return normalize_data_train,normalize_data_test
    else:
        max_values=data_train[features_name].max().max()
        min_values=data_train[features_name].min().min()
        normalize_data_train=(data_train[features_name]-min_values)/(max_values-min_values)
        normalize_data_test=(data_test[features_name]-min_values)/(max_values-min_values)
        return normalize_data_train,normalize_data_test,max_values,min_values


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
