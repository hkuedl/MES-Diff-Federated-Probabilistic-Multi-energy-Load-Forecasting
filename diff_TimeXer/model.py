
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from utils import MAPE, RMSE
import argparse
from einops import rearrange
from torch import einsum
from functools import partial
from itertools import repeat
import collections.abc
import math
import matplotlib.pyplot as plt
from utils import *
from typing import Callable, List, Optional, Tuple,Union
from blitz.modules import BayesianLinear
from blitz.losses import kl_divergence_from_nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#local_model
class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.input_size = args.seq_len
        self.layer_size = args.local_layer_size
        self.output_size = args.label_len
        self.ex_dim = args.ex_dim
        self.layers = nn.ModuleList()
        num_features = 1
        
        self.layers.append(nn.Linear(self.input_size, self.layer_size[0]))
        for i in range(len(self.layer_size) - 1):
            self.layers.append(nn.Linear(self.layer_size[i], self.layer_size[i + 1]))
        self.layers.append(nn.Linear(self.layer_size[-1]+self.ex_dim, self.output_size))
        self.sigma = nn.Linear(self.layer_size[-1]+self.ex_dim,self.output_size)

        self.revin_layer = RevIN(num_features)

    def forward(self, x,x_ex):
        x = self.revin_layer(x.unsqueeze(-1), 'norm').squeeze(-1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.revin_layer(x.unsqueeze(-1), 'denorm').squeeze(-1)
        x_ex = x_ex.reshape(x_ex.size(0), -1)
        mu = self.layers[-1](torch.cat([x,x_ex],-1)) 
        sigma = F.softplus(self.sigma(torch.cat([x,x_ex],-1)))
        
        return mu, sigma
    
    def get_hidden(self, x):
        x = self.revin_layer(x.unsqueeze(-1), 'norm').squeeze(-1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        
        x = self.revin_layer(x.unsqueeze(-1), 'denorm').squeeze(-1)
        return x


class MLP_central(nn.Module):
    def __init__(self, args):
        super(MLP_central, self).__init__()
        self.input_size = args.seq_len
        self.layer_size = args.local_layer_size
        self.output_size = args.label_len
        self.ex_dim = args.ex_dim
        self.layers = nn.ModuleList()
        num_features = 3
        
        self.layers.append(nn.Linear(self.input_size, self.layer_size[0]))
        for i in range(len(self.layer_size) - 1):
            self.layers.append(nn.Linear(self.layer_size[i], self.layer_size[i + 1]))
        self.layers.append(nn.Linear(self.layer_size[-1]+self.ex_dim, self.output_size))
        self.sigma = nn.Linear(self.layer_size[-1]+self.ex_dim,self.output_size)

        self.revin_layer = RevIN(num_features)

    def forward(self, x,x_ex):
        x_ex = x_ex.permute(0, 2, 1)  
        x_ex = x_ex.repeat(1, 3, 1) 
        # x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1) 
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        mu = self.layers[-1](torch.cat([x,x_ex],-1)) 
        sigma = F.softplus(self.sigma(torch.cat([x,x_ex],-1)))
        
        return mu.permute(0,2,1), sigma.permute(0,2,1)

class GRU_central(nn.Module):
    def __init__(self, args):

        super(GRU_central, self).__init__()
        num_features = 3
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=args.hidden_dim,
            num_layers=args.num_layer,
            batch_first=True
        )

        self.fc = nn.Linear(args.hidden_dim + args.ex_dim, args.label_len * 3)
        self.sigma = nn.Linear(args.hidden_dim + args.ex_dim, args.label_len * 3)

        self.revin_layer = RevIN(num_features)
        self.hidden_dim = args.hidden_dim
        self.num_layer = args.num_layer
        self.label_len = args.label_len

    def forward(self, x, x_ex):
        batch_size = x.size(0)
        x_ex = x_ex.squeeze(-1)  
        # x = self.revin_layer(x, 'norm')
        h0 = torch.zeros(self.num_layer, batch_size, self.hidden_dim).to(x.device)

        gru_out, _ = self.gru(x, h0) 

        gru_out = gru_out[:, -1, :] 
        # gru_out = self.revin_layer(gru_out, 'denorm')

        combined = torch.cat([gru_out, x_ex], dim=-1)

        mu = self.fc(combined).reshape(batch_size, self.label_len, 3)
        sigma = F.softplus(self.sigma(combined)).reshape(batch_size, self.label_len, 3)
        
        return mu, sigma
    
# 三个GRU
class GRU_central_multi(nn.Module):
    def __init__(self, args):

        super(GRU_central_multi, self).__init__()
        num_features = 1
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=args.hidden_dim,
            num_layers=args.num_layer,
            batch_first=True
        )
        

        self.fc = nn.Linear(args.hidden_dim + args.ex_dim, args.label_len)
        self.sigma = nn.Linear(args.hidden_dim + args.ex_dim, args.label_len)

        self.revin_layer = RevIN(num_features)
        self.hidden_dim = args.hidden_dim
        self.num_layer = args.num_layer
        self.label_len = args.label_len

    def forward(self, x, x_ex):

        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        # x = self.revin_layer(x.unsquueze(-1), 'norm').squueze(-1)

        h0 = torch.zeros(self.num_layer, batch_size, self.hidden_dim).to(x.device)
        gru_out, _ = self.gru(x, h0)  # gru_out形状: (batch_size, seq_len, hidden_size)
        gru_out = gru_out[:, -1, :]  # 形状: (batch_size, hidden_size)

        # gru_out = self.revin_layer(gru_out.unsqueeze(-1), 'denorm').squeeze(-1)

        x_ex = x_ex.reshape(x_ex.size(0), -1)  # 调整形状为 (batch_size, ex_dim)
        combined = torch.cat([gru_out, x_ex], dim=-1)

        mu = self.fc(combined)
        sigma = F.softplus(self.sigma(combined))
        
        return mu, sigma
    

class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        num_features = 1

        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=args.hidden_dim,
            num_layers=args.num_layer,
            batch_first=True
        )
        
        self.fc = nn.Linear(args.hidden_dim + args.ex_dim, args.label_len)
        self.sigma = nn.Linear(args.hidden_dim + args.ex_dim, args.label_len)

        self.revin_layer = RevIN(num_features)
        self.hidden_dim = args.hidden_dim
        self.num_layer = args.num_layer

    def forward(self, x, x_ex):
        x = x.unsqueeze(-1) 
        batch_size = x.size(0)
        x = self.revin_layer(x, 'norm')
        
        h0 = torch.zeros(self.num_layer, batch_size, self.hidden_dim).to(x.device)
        gru_out, _ = self.gru(x, h0)  # gru_out形状: (batch_size, seq_len, hidden_size)
        
        gru_out = gru_out[:, -1, :]  # 形状: (batch_size, hidden_size)
        
        gru_out = self.revin_layer(gru_out.unsqueeze(1), 'denorm').squeeze(1)
        x_ex = x_ex.reshape(x_ex.size(0), -1)

        combined = torch.cat([gru_out, x_ex], dim=-1)
        
        mu = self.fc(combined)
        sigma = F.softplus(self.sigma(combined))
        
        return mu, sigma
    
    def get_hidden(self, x):
        x = x.unsqueeze(-1) 
        batch_size = x.size(0)
        x = self.revin_layer(x, 'norm')

        h0 = torch.zeros(self.num_layer, batch_size, self.hidden_dim).to(x.device)
        gru_out, _ = self.gru(x, h0)

        gru_out = gru_out[:, -1, :]
        
        gru_out = self.revin_layer(gru_out.unsqueeze(1), 'denorm').squeeze(1)
        
        return gru_out

def variational_estimator(nn_class):
    """
    This decorator adds some util methods to a nn.Module, in order to facilitate the handling of Bayesian Deep Learning features

    Parameters:
        nn_class: torch.nn.Module -> Torch neural network module

    Returns a nn.Module with methods for:
        (1) Gathering the KL Divergence along its BayesianModules;
        (2) Sample the Elbo Loss along its variational inferences (helps training)
        (3) Freeze the model, in order to predict using only their weight distribution means
        (4) Specifying the variational parameters by using some prior weights after training the NN as a deterministic model
    """

    def nn_kl_divergence(self):
        """Returns the sum of the KL divergence of each of the BayesianModules of the model, which are from
            their posterior current distribution of weights relative to a scale-mixtured prior (and simpler) distribution of weights

            Parameters:
                N/a

            Returns torch.tensor with 0 dim.      
        
        """
        return kl_divergence_from_nn(self)
    
    setattr(nn_class, "nn_kl_divergence", nn_kl_divergence)

    def sample_elbo(self,
                    his,
                    ex,
                    labels,
                    criterion,
                    sample_nbr,
                    complexity_cost_weight=1):

        """ Samples the ELBO Loss for a batch of data, consisting of inputs and corresponding-by-index labels
                The ELBO Loss consists of the sum of the KL Divergence of the model
                 (explained above, interpreted as a "complexity part" of the loss)
                 with the actual criterion - (loss function) of optimization of our model
                 (the performance part of the loss).
                As we are using variational inference, it takes several (quantified by the parameter sample_nbr) Monte-Carlo
                 samples of the weights in order to gather a better approximation for the loss.
            Parameters:
                inputs: torch.tensor -> the input data to the model
                labels: torch.tensor -> label data for the performance-part of the loss calculation
                        The shape of the labels must match the label-parameter shape of the criterion (one hot encoded or as index, if needed)
                criterion: torch.nn.Module, custom criterion (loss) function, torch.nn.functional function -> criterion to gather
                            the performance cost for the model
                sample_nbr: int -> The number of times of the weight-sampling and predictions done in our Monte-Carlo approach to
                            gather the loss to be .backwarded in the optimization of the model.

        """

        loss = 0
        for _ in range(sample_nbr):
            mu,sigma = self(his,ex)
            loss += criterion(labels, mu, sigma)
            loss += self.nn_kl_divergence() * complexity_cost_weight
        return loss / sample_nbr

    setattr(nn_class, "sample_elbo", sample_elbo)
    return nn_class

#一个贝叶斯
@variational_estimator    
class Bayes_central(nn.Module):
    def __init__(self, args, num_features=3):
        """
        :param input_size: 输入特征维度
        :param layer_sizes: 隐藏层的维度列表，例如 [64, 128, 64]
        :param output_size: 输出目标值的维度
        """
        super(Bayes_central, self).__init__()
        self.layers = nn.ModuleList()

        # 添加输入层
        self.layers.append(BayesianLinear(args.seq_len, args.layer_size[0]))
        
        # 添加隐藏层
        for i in range(len(args.layer_size) - 1):
            self.layers.append(BayesianLinear(args.layer_size[i], args.layer_size[i + 1]))

        self.mu = BayesianLinear(args.layer_size[-1]+args.ex_dim, args.label_len) 
        self.sigma = BayesianLinear(args.layer_size[-1]+args.ex_dim, args.label_len)

        self.revin_layer = RevIN(num_features)

    def forward(self, x, ex):
        '''
        :param x: 输入数据，形状为 (batch_size, seq_len, 3)
        :param ex: 外部特征，形状为 (batch_size, ex_dim, 1)
        '''
        # x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1)  # 调整形状为 (batch_size, 3, seq_len)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        
        # x = self.revin_layer(x.permute(0,2,1), 'denorm').permute(0,2,1)
        ex = ex.permute(0, 2, 1)  # 调整形状为 (batch_size, 1, ex_dim)
        ex = ex.repeat(1, 3, 1)  # 扩展为 (batch_size, 3, ex_dim)
        # 最后一层输出 mu 和 sigma
        mu = self.mu(torch.cat([x,ex],-1))  # 形状: (batch_size, 3, seq_len+ex_dim)
        sigma = F.softplus(self.sigma(torch.cat([x,ex],-1)))
        return mu.permute(0,2,1), sigma.permute(0,2,1)

# 三个bayes
@variational_estimator    
class Bayes_central_multi(nn.Module):
    def __init__(self, args, num_features=1):
        """
        :param input_size: 输入特征维度
        :param layer_sizes: 隐藏层的维度列表，例如 [64, 128, 64]
        :param output_size: 输出目标值的维度
        """
        super(Bayes_central_multi, self).__init__()
        self.layers = nn.ModuleList()
        num_features = 1
        # 添加输入层
        self.layers.append(BayesianLinear(args.seq_len *3, args.layer_size[0]))
        
        # 添加隐藏层
        for i in range(len(args.layer_size) - 1):
            self.layers.append(BayesianLinear(args.layer_size[i], args.layer_size[i + 1]))

        self.mu = BayesianLinear(args.layer_size[-1]+args.ex_dim, args.label_len) 
        self.sigma = BayesianLinear(args.layer_size[-1]+args.ex_dim, args.label_len)

        self.revin_layer = RevIN(num_features)

    def forward(self, x, ex):
        '''
        :param x: 输入数据，形状为 (batch_size, seq_len*3)
        :param ex: 外部特征，形状为 (batch_size, 24, 6)
        '''
        # x = self.revin_layer(x.unsqueeze(-1), 'norm').squeeze(1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # x = self.revin_layer(x.unsqueeze(-1), 'denorm').squeeze(-1)
        ex = ex.reshape(ex.size(0), -1)  # 调整形状为 (batch_size, ex_dim)
        # 最后一层输出 mu 和 sigma
        mu = self.mu(torch.cat([x,ex],-1))  # 形状: (batch_size, seq_len+ex_dim)
        sigma = F.softplus(self.sigma(torch.cat([x,ex],-1)))
        return mu, sigma
    

@variational_estimator    
class Bayes(nn.Module):
    def __init__(self, args, num_features=1):
        """
        :param input_size: 输入特征维度
        :param layer_sizes: 隐藏层的维度列表，例如 [64, 128, 64]
        :param output_size: 输出目标值的维度
        """
        super(Bayes, self).__init__()
        self.layers = nn.ModuleList()

        # 添加输入层
        self.layers.append(BayesianLinear(args.seq_len, args.layer_size[0]))
        
        # 添加隐藏层
        for i in range(len(args.layer_size) - 1):
            self.layers.append(BayesianLinear(args.layer_size[i], args.layer_size[i + 1]))

        self.mu = BayesianLinear(args.layer_size[-1]+args.ex_dim, args.label_len) 
        self.sigma = BayesianLinear(args.layer_size[-1]+args.ex_dim, args.label_len)

        self.revin_layer = RevIN(num_features)

    def forward(self, x, ex):
        '''
        :param x: 输入数据，形状为 (batch_size, seq_len)
        :param ex: 外部特征，形状为 (batch_size, 24, 6)
        '''
        x = self.revin_layer(x.unsqueeze(-1), 'norm').squeeze(-1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        x = self.revin_layer(x.unsqueeze(-1), 'denorm').squeeze(-1)
        ex = ex.reshape(ex.size(0),-1)  # 调整形状为 (batch_size, ex_dim)
        # 最后一层输出 mu 和 sigma
        mu = self.mu(torch.cat([x,ex],-1))  # 形状: (batch_size, seq_len+ex_dim)
        sigma = F.softplus(self.sigma(torch.cat([x,ex],-1)))
        return mu, sigma
    
        
class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        num_features = 1
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=args.hidden_dim,
            num_layers=args.num_layer,
            batch_first=True
        )
        
        self.fc = nn.Linear(args.hidden_dim + args.ex_dim, args.label_len)
        self.sigma = nn.Linear(args.hidden_dim + args.ex_dim, args.label_len)
        
        self.revin_layer = RevIN(num_features)
        self.hidden_dim = args.hidden_dim
        self.num_layer = args.num_layer

    def forward(self, x, x_ex):
        x = x.unsqueeze(-1)  # 增加一个维度，变为 (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        x = self.revin_layer(x, 'norm')
        h0 = torch.zeros(self.num_layer, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layer, batch_size, self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out形状: (batch_size, seq_len, hidden_size)
        lstm_out = lstm_out[:, -1, :]  # 形状: (batch_size, hidden_size)
        lstm_out = self.revin_layer(lstm_out.unsqueeze(1), 'denorm').squeeze(1)
        x_ex = x_ex.reshape(x_ex.size(0), -1)
        combined = torch.cat([lstm_out, x_ex], dim=-1)
        
        mu = self.fc(combined)
        sigma = F.softplus(self.sigma(combined))
        
        return mu, sigma
    
    def get_hidden(self, x):
        x = x.unsqueeze(-1) 
        batch_size = x.size(0)
        x = self.revin_layer(x, 'norm')
        h0 = torch.zeros(self.num_layer, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layer, batch_size, self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.revin_layer(lstm_out.unsqueeze(1), 'denorm').squeeze(1)
        
        return lstm_out
    

class MLP_server(nn.Module):
    def __init__(self, args):
        super(MLP_server, self).__init__()
        self.input_size = args.hidden_dim * 3
        self.layer_size = args.layer_size
        self.output_size = args.label_len
        self.ex_dim = args.ex_dim
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.input_size, self.layer_size[0]))

        for i in range(len(self.layer_size) - 1):
            self.layers.append(nn.Linear(self.layer_size[i], self.layer_size[i + 1]))
        
        self.layers.append(nn.Linear(self.layer_size[-1]+self.ex_dim, self.output_size))
        self.sigma = nn.Linear(self.layer_size[-1]+self.ex_dim,self.output_size)
        self.device = args.device
        num_features = 1
        self.revin_layer = RevIN(num_features)


    def forward(self, x, ex):
        x = x.to(self.device)
        ex = ex.to(self.device)
        # x = self.revin_layer(x.unsqueeze(-1), 'norm').squeeze(-1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # x = self.revin_layer(x.unsqueeze(-1), 'denorm').squeeze(-1)
        ex = ex.reshape(ex.size(0), -1)  #(batch_size, ex_dim)
        mu = self.layers[-1](torch.cat([x,ex],-1)) 
        sigma = F.softplus(self.sigma(torch.cat([x,ex],-1)))
        
        return mu, sigma

# 三个mlp
class MLP_central_multi(nn.Module):
    def __init__(self, args):
        super(MLP_central_multi, self).__init__()
        self.input_size = args.seq_len * 3
        self.layer_size = args.layer_size
        self.output_size = args.label_len
        self.ex_dim = args.ex_dim
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(self.input_size, self.layer_size[0]))
        
        for i in range(len(self.layer_size) - 1):
            self.layers.append(nn.Linear(self.layer_size[i], self.layer_size[i + 1]))
        
        self.layers.append(nn.Linear(self.layer_size[-1]+self.ex_dim, self.output_size))
        self.sigma = nn.Linear(self.layer_size[-1]+self.ex_dim,self.output_size)
        self.device = args.device
        num_features = 1
        self.revin_layer = RevIN(num_features)


    def forward(self, x, ex):
        x = x.to(self.device)
        ex = ex.to(self.device)
        # x = self.revin_layer(x.unsqueeze(-1), 'norm').squeeze(-1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # x = self.revin_layer(x.unsqueeze(-1), 'denorm').squeeze(-1)
        ex = ex.reshape(ex.size(0), -1)  #(batch_size, ex_dim)
        mu = self.layers[-1](torch.cat([x,ex],-1)) 
        sigma = F.softplus(self.sigma(torch.cat([x,ex],-1)))
        
        return mu, sigma

def p_sample_loop(model, shape, n_steps, betas, alphas, alphas_bar, his, c):
    device = next(model.parameters()).device
    x = torch.randn(shape).to(device)
    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_bar = alphas_bar.to(device)
    for t in range(n_steps - 1, -1, -1):
        z = torch.randn_like(x)
        t_tensor = torch.tensor(t, device=device).repeat(x.shape[0])
        x0_conditioned,sigma0_conditoned = model(x, t_tensor, his, c)
        
        x0_theta_pre = x0_conditioned
        x0_theta = x0_theta_pre
        sigma0_theta = sigma0_conditoned

        if t > 0:
            mu_pred = (
                torch.sqrt(alphas[t])
                * (1 - alphas_bar[t - 1])
                * x
                + torch.sqrt(alphas_bar[t - 1])
                * betas[t]
                * x0_theta
            )
            mu_pred = mu_pred / (1 - alphas_bar[t])
        else:
            mu_pred = x0_theta
        if t == 0:
            sigma = 0
        else:
            sigma = torch.sqrt(
                (1 - alphas_bar[t - 1])
                / (1 - alphas_bar[t])
                * betas[t]
            )

        x = mu_pred + sigma * z
    return x,sigma0_theta

class cauchy_likelihood(nn.Module):
    def __init__(self):
        super(cauchy_likelihood, self).__init__()
    def forward(self, label, mu, sigma):
        distribution = torch.distributions.cauchy.Cauchy(mu,sigma)
        loss = distribution.log_prob(label)
        return -torch.mean(loss)


def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps, his, c):
    device = x_0.device
    batch_size = x_0.shape[0]
    loss_function = cauchy_likelihood() #可以替换成cauchy_likelihood
    # t = torch.randint(0, n_steps, size=(batch_size // 2,))
    t = torch.randint(0, n_steps, size=(batch_size,))
    # t = torch.cat([t, n_steps - 1 - t], dim=0)
    t = t.unsqueeze(-1).to(device)

    a = alphas_bar_sqrt[t].to(device)
    aml = one_minus_alphas_bar_sqrt[t].to(device)
    e = torch.randn_like(x_0).to(device)
    x = x_0 * a.unsqueeze(-1) + e * aml.unsqueeze(-1)

    x0_conditioned,sigma_0_conditioned = model(x, t.squeeze(-1), his ,c)
    
    # 计算最终输出
    x0_theta = x0_conditioned
    sigma_0_theta = sigma_0_conditioned
    return loss_function(x_0,x0_theta,sigma_0_theta)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse

class SinusoidalPosEmb(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, latent_dim: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = latent_dim
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = nn.SiLU()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
    
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None
        

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        x = self.value_embedding(x)
        # x: [Batch Variate d_model]
        return self.dropout(x)
    

class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        # x.shape= (batch_size, n_vars, seq_len)
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len) #(batch_size, n_vars, seq_len) -> (batch_size, n_vars, num_patch, patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x) #(batch_size*n_vars, num_patch, d_model)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1])) #(batch_size, n_vars, num_patch, d_model)
        x = torch.cat([x, glb], dim=2) #(batch_size, n_vars, num_patch+1, d_model)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) #(batch_size*n_vars, num_patch+1, d_model)
        return self.dropout(x), n_vars

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)
    
    
class TimeXerDiff(nn.Module):
    def __init__(self,args):
        super(TimeXerDiff, self).__init__()
        self.use_norm = args.use_norm
        self.label_len = args.label_len
        self.patch_len = args.patch_len
        self.patch_num = int(args.label_len // args.patch_len)
        self.n_vars = 1 # 内生变量的特征维度
        self.pe = SinusoidalPosEmb(args.d_model)
        # Embedding
        self.en_embedding = EnEmbedding(self.n_vars, args.d_model, self.patch_len, args.dropout)
        self.his_embedding = DataEmbedding_inverted(args.hidden_dim, args.d_model, args.dropout)
        self.c_embedding = DataEmbedding_inverted(args.label_len, args.d_model, args.dropout)
        self.device = args.device

        # 只有Encoder的transformer
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.n_heads),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation,
                )
                for l in range(args.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        self.head_nf = args.d_model * (self.patch_num + 1)
        self.head = FlattenHead(self.n_vars, self.head_nf, args.label_len, head_dropout=args.dropout)
        self.mu = nn.Linear(self.label_len, self.label_len)
        self.presigma = nn.Linear(self.label_len, self.label_len)
        self.sigma = nn.Softplus()
        self.con_pred = nn.Linear(self.n_vars, self.n_vars)
 
    
    def forward(self, x_enc, t, his, c, mask=None):
        """
        x_enc.shape = (batch_size, label_len, 1)
        his.shape = (batch_size, seq_len, 3) 冷热电历史负荷
        c.shape = (batch_size, label_len, 6) 天气日历
        """
        x_enc = x_enc.to(self.device)
        his = his.to(self.device)
        c = c.to(self.device)
        t = t.to(self.device)
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        t = self.pe(t).unsqueeze(1) #(batch_size,1,d_model)
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1)) #(batch_size, label_len, 1) -> (batch_size*n_vars, num_patch+1, d_model)
        en_embed = en_embed + t #(batch_size*n_vars, num_patch+1, d_model)
        his_embed = self.his_embedding(his) #(batch_size,feature_dim,d_model)
        c_embed = self.c_embedding(c)
        ex_embed = torch.cat([his_embed, c_embed], dim=1) #(batch_size,6+3,d_model)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.label_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.label_len, 1))

        mu = self.mu(dec_out[:, -self.label_len:, :].permute(0, 2, 1)) #(batch_size,1,label_len)
        sigma = self.sigma(self.presigma(dec_out[:, -self.label_len:, :].permute(0, 2, 1)))+1e-6


        return mu.permute(0, 2, 1),sigma.permute(0, 2, 1)
    
# # 三个diff
# class TimeXerDiff_central(nn.Module):
#     def __init__(self,args):
#         super(TimeXerDiff_central, self).__init__()
#         self.use_norm = args.use_norm
#         self.label_len = args.label_len
#         self.patch_len = args.patch_len
#         self.patch_num = int(args.label_len // args.patch_len)
#         self.n_vars = 1 # 内生变量的特征维度
#         self.pe = SinusoidalPosEmb(args.d_model)
#         # Embedding
#         self.en_embedding = EnEmbedding(self.n_vars, args.d_model, self.patch_len, args.dropout)
#         self.his_embedding = DataEmbedding_inverted(args.seq_len, args.d_model, args.dropout)
#         self.c_embedding = DataEmbedding_inverted(args.label_len, args.d_model, args.dropout)
#         self.device = args.device

#         # 只有Encoder的transformer
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, attention_dropout=args.dropout, output_attention=False),
#                         args.d_model, args.n_heads),
#                     AttentionLayer(
#                         FullAttention(False, attention_dropout=args.dropout, output_attention=False),
#                         args.d_model, args.n_heads),
#                     args.d_model,
#                     args.d_ff,
#                     dropout=args.dropout,
#                     activation=args.activation,
#                 )
#                 for l in range(args.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(args.d_model)
#         )
#         self.head_nf = args.d_model * (self.patch_num + 1)
#         self.head = FlattenHead(self.n_vars, self.head_nf, args.label_len, head_dropout=args.dropout)
#         self.mu = nn.Linear(self.label_len, self.label_len)
#         self.presigma = nn.Linear(self.label_len, self.label_len)
#         self.sigma = nn.Softplus()
#         self.con_pred = nn.Linear(self.n_vars, self.n_vars)
 
    
#     def forward(self, x_enc, t, his, c, mask=None):
#         """
#         x_enc.shape = (batch_size, label_len, 1)
#         his.shape = (batch_size, seq_len, 3) 冷热电历史负荷
#         c.shape = (batch_size, label_len, 6) 天气日历
#         """
#         x_enc = x_enc.to(self.device)
#         his = his.to(self.device)
#         c = c.to(self.device)
#         t = t.to(self.device)
#         if self.use_norm:
#             # Normalization from Non-stationary Transformer
#             means = x_enc.mean(1, keepdim=True).detach()
#             x_enc = x_enc - means
#             stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             x_enc /= stdev

#         _, _, N = x_enc.shape
#         t = self.pe(t).unsqueeze(1) #(batch_size,1,d_model)
#         en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1)) #(batch_size, label_len, 1) -> (batch_size*n_vars, num_patch+1, d_model)
#         en_embed = en_embed + t #(batch_size*n_vars, num_patch+1, d_model)
#         his_embed = self.his_embedding(his) #(batch_size,feature_dim,d_model)
#         c_embed = self.c_embedding(c)
#         ex_embed = torch.cat([his_embed, c_embed], dim=1) #(batch_size,6+3,d_model)

#         enc_out = self.encoder(en_embed, ex_embed)
#         enc_out = torch.reshape(
#             enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
#         # z: [bs x nvars x d_model x patch_num]
#         enc_out = enc_out.permute(0, 1, 3, 2)

#         dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
#         dec_out = dec_out.permute(0, 2, 1)

#         if self.use_norm:
#             # De-Normalization from Non-stationary Transformer
#             dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.label_len, 1))
#             dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.label_len, 1))

#         mu = self.mu(dec_out[:, -self.label_len:, :].permute(0, 2, 1)) #(batch_size,1,label_len)
#         sigma = self.sigma(self.presigma(dec_out[:, -self.label_len:, :].permute(0, 2, 1)))+1e-6


#         return mu.permute(0, 2, 1),sigma.permute(0, 2, 1)


class MLP_layer(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False, **kwargs
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        to_2tuple = _ntuple(2)
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class CrossAttentionBlock(nn.Module):
    """
    ### Cross Attention Block for inputs of shape (batch_size, n_channels, d_model)
    """

    def __init__(
        self, d_model: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32
    ):
        """
        * `d_model` is the number of features in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for group normalization
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = d_model // n_heads  # Ensure d_k * n_heads = d_model

        # Projections for query, key, and values
        self.query_projection = nn.Linear(d_model, n_heads * d_k)
        self.key_projection = nn.Linear(d_model, n_heads * d_k)
        self.value_projection = nn.Linear(d_model, n_heads * d_k)

        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, d_model)

        # Scale for dot-product attention
        self.scale = d_k**-0.5

        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, else_condition: torch.Tensor):
        """
        * `x` has shape `[batch_size, n_channels, d_model]`
        * `else_condition` has shape `[batch_size, n_channels, d_model]`
        """
        batch_size, n_channels, d_model = x.shape

        # Reshape `x` and `else_condition` for multi-head attention
        # Project `x` to get queries
        q = self.query_projection(x).view(batch_size, n_channels, self.n_heads, self.d_k)  # (batch_size, n_channels, n_heads, d_k)

        # Project `else_condition` to get keys and values
        k = self.key_projection(else_condition).view(batch_size, n_channels, self.n_heads, self.d_k)  # (batch_size, n_channels, n_heads, d_k)
        v = self.value_projection(else_condition).view(batch_size, n_channels, self.n_heads, self.d_k)  # (batch_size, n_channels, n_heads, d_k)

        # Calculate scaled dot-product attention
        attn = torch.einsum("bnhe,bnhe->bnh", q, k) * self.scale  # (batch_size, n_channels, n_heads)
        attn = attn.softmax(dim=-1)  # Softmax over heads

        # Multiply by values
        res = torch.einsum("bnh,bnhe->bnhe", attn, v)  # (batch_size, n_channels, n_heads, d_k)
        res = res.reshape(batch_size, n_channels, -1)  # (batch_size, n_channels, n_heads * d_k)

        # Transform to `[batch_size, n_channels, d_model]`
        res = self.output(res)

        # Add skip connection
        res += x

        return res
    

class AttentionBlock(nn.Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(
        self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32
    ):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        # self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, seq_len]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, seq_length = x.shape
        # Change `x` to shape `[batch_size, seq_len, n_channels]`
        x = x.permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, seq_length]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, seq_length)

        #
        return res
    


class MLPDiff(nn.Module):
    """
    ## MLP backbone for denoising a time-domian data
    """

    def __init__(self,args):
        """
        * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
        * `seq_length` is the number of timesteps in the time series.
        * `hidden_size` is the hidden size
        * `n_layers` is the number of MLP
        """
        super().__init__()
        self.device = args.device
        label_length = args.label_len
        seq_channels = 1
        d_model = args.d_model
        d_mlp = args.d_model
        latent_dim = args.hidden_dim *3
        ex_dim = args.ex_dim
        n_layers = args.diff_layers
        dropout = args.dropout
        self.embedder = nn.Linear(label_length, d_model)
        self.unembedder = nn.Linear(d_model, label_length)
        self.pe = SinusoidalPosEmb(d_model)
        # self.pe = GaussianFourierProjection(d_model)
        self.net = nn.ModuleList(  # type: ignore
            [
                MLP_layer(
                    in_features=d_model,
                    hidden_channels=[d_mlp, d_model],
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        if latent_dim is not None:
            if d_model != latent_dim:
                self.his_linear = nn.Linear(latent_dim, d_model)
            else:
                self.his_linear = nn.Identity()
        else:
            pass

        self.mu = nn.Linear(seq_channels,seq_channels)
        self.presigma = nn.Linear(seq_channels, seq_channels)
        self.sigma = nn.Softplus()
        self.attention = AttentionBlock(seq_channels)
        self.con_ex = nn.Linear(ex_dim,d_model)
        self.con_c = nn.Linear(latent_dim+ex_dim,d_model)
        self.con_pred = nn.Linear(seq_channels, seq_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, his: torch.Tensor = None, ex: torch.Tensor = None):
        """
        x.shape = (batch_size, seq_len, 1)
        his.shape = (batch_size,latent_dim,3)
        ex.shape = (batch_size,24,6)
        """
        x = x.to(self.device)
        t = t.to(self.device)
        his = his.to(self.device)
        ex = ex.to(self.device)
        his = his.reshape(his.size(0),-1).unsqueeze(1) #(batch_size,1,latent_dim*3)
        # his = self.his_linear(his) #(batch_size,1,latent_dim*3)->(batch_size,1,d_model)
        ex = ex.reshape(ex.size(0),-1).unsqueeze(1) #(batch_size,1,ex_dim)
        c = torch.cat([his, ex], dim=-1) #(batch_size,1,latent_dim*3+ex_dim)

        x_pred = self.con_pred(x)
        c = self.con_c(c) #(batch_size,1,latent_dim*3+ex_dim)->(batch_size,1,d_model)
        # ex = self.con_ex(ex) #(batch_size,1,ex_dim)->(batch_size,1,d_model)

        x = self.embedder(x.permute(0, 2, 1)) #(batch_size,1,d_model)
        t = self.pe(t).unsqueeze(1)#(batch_size,1,d_model)

        x = x + c + t #(batch_size,1,d_model)
        x = self.attention(x) #(batch_size,1,d_model)
        for layer in self.net:
            x = x + layer(x)
        x = self.unembedder(x).permute(0, 2, 1) #(batch_size,seq_length,1)
        x = x + x_pred #(batch_size,seq_length,1)
        mu = self.mu(x)
        sigma = self.sigma(self.presigma(x))+1e-6
        
        return mu, sigma
    