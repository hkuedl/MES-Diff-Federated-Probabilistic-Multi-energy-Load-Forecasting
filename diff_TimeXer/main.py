import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pickle
import copy
import warnings
from Data_loader import get_data
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import os
import time
from utils import EarlyStopping, cauchy_likelihood, metric, NoiseInjectionModule, DiffDataset, plot_train_loss, cosine_schedule, set_seed
from model import MLP, LSTM, diffusion_loss_fn, TimeXerDiff, MLP_server, p_sample_loop, MLPDiff, GRU, MLP_central, GRU_central, Bayes_central, MLPDiff_central,TimeXerDiff_central, Bayes,MLP_central_multi, GRU_central_multi, Bayes_central_multi
import csv

class LocalModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        set_seed(args.seed)
        self.model = self._build_model(args).to(args.device)

    def _build_model(self, args):
        if args.local_model == "MLP":
            return MLP(args)
        elif args.local_model == "LSTM":
            return LSTM(args)
        elif args.local_model == "GRU":
            return GRU(args)
        elif args.local_model == "Bayes":
            return Bayes(args)
        else:
            raise ValueError(f"Unknown model type: {args.local_model}")
    
    def _get_data(self, args, flag, load):
        data_set, data_loader = get_data(args,flag,load)
        return data_set, data_loader

    def val(self, args, val_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (X_his,X_ex,y) in enumerate(val_loader):
                X_his,X_ex,y = X_his.to(args.device),X_ex.to(args.device),y.to(args.device)
                mu,sigma = self.model(X_his,X_ex)
                loss = criterion(y,mu,sigma).detach().cpu()
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, args, setting, load):
        # Load data
        train_data, train_loader = self._get_data(args, flag='train', load=load)
        val_data, val_loader = self._get_data(args, flag='val', load=load)
        path = os.path.join(args.checkpoints, setting, 'local_'+load)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=args.patience, verbose=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        criterion = cauchy_likelihood()
        schedule = cosine_schedule(args.num_steps)
        alphas = schedule['alphas'].to(args.device)
        betas = schedule['betas'].to(args.device)
        alphas_bar = schedule['alpha_bars'].to(args.device)
        alphas_bar_sqrt = torch.sqrt(alphas_bar).to(args.device)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_bar).to(args.device)

        train_losses = []
        val_losses = []
        # Training loop
        for epoch in range(args.epochs_local):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (X_his,X_ex,y) in enumerate(train_loader):
                """
                X_his: [batch_size, seq_len]
                X_ex: [batch_size, label_len, feature_num(6)]
                y: [batch_size, label_len]
                """
                iter_count += 1
                optimizer.zero_grad()
                X_his,X_ex,y = X_his.to(args.device),X_ex.to(args.device),y.to(args.device)
                mu,sigma = self.model(X_his,X_ex)
                loss = criterion(y,mu,sigma)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_losses.append(train_loss)
            val_loss = self.val(args, val_loader, criterion)
            val_losses.append(val_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} val Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, val_loss))
            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        plot_train_loss(train_losses, val_losses)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    

    def test(self, args, setting, load, test=0):
        test_data, test_loader = self._get_data(args, flag='test', load=load)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'local_'+load, 'checkpoint.pth')))
        
        mus = []
        sigmas = []
        actuals = []

        self.model.eval()
        with torch.no_grad():
            for i, (X_his,X_ex,y) in enumerate(test_loader):
                X_his,X_ex,y = X_his.to(args.device),X_ex.to(args.device),y.to(args.device)
                mu,sigma = self.model(X_his,X_ex)
                mu = mu.detach().cpu()
                sigma = sigma.detach().cpu()
                y = y.detach().cpu()
                mus.append(mu)
                sigmas.append(sigma)
                actuals.append(y)
            mus_normed = np.concatenate(mus, axis=0)
            sigmas_normed = np.concatenate(sigmas, axis=0)
            actuals_normed = np.concatenate(actuals, axis=0)

            pred_mu = test_data.inverse_transform(mus_normed)
            pred_sigma = test_data.scale_()*sigmas_normed
            actual = test_data.inverse_transform(actuals_normed)

            pred_mu = pred_mu.reshape(-1)
            pred_sigma = pred_sigma.reshape(-1)
            actual = actual.reshape(-1)

            folder_path = './test_results/' + setting +'/'+ 'local_' + load + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            mape, rmse, mae, pinball_loss, crps, winkler_score_25, winkler_score_50, winkler_score_75 = metric(args, pred_mu, pred_sigma, actual)


            np.save(folder_path + 'mu.npy', pred_mu)
            np.save(folder_path + 'sigma.npy', pred_sigma)
            np.save(folder_path + 'actual.npy', actual)
            
            print(f'mape:{mape}, rmse:{rmse}, mae:{mae}, pinball_loss:{pinball_loss}')
            # CSV 文件名
            csv_file = folder_path + 'metrics.csv'
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(['Setting', 'MAPE', 'MAE', 'RMSE', 'PINBALL_LOSS', 'CRPS', 
                                'WINKLER_SCORE_25', 'WINKLER_SCORE_50', 'WINKLER_SCORE_75'])
                # 写入当前数据
                writer.writerow([setting, mape, mae, rmse, pinball_loss, crps, 
                                winkler_score_25, winkler_score_50, winkler_score_75])
            return
        
    def predict(self,args,setting,load,X_his,X_ex):
        self.model.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'local_'+load, 'checkpoint.pth')))
        self.model.eval()
        with torch.no_grad():
            X_his,X_ex = X_his.to(args.device),X_ex.to(args.device)
            mu,sigma = self.model(X_his,X_ex)
        return mu, sigma
        
    def embedding(self,args,setting,load,X_his):
        self.model.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'local_'+load, 'checkpoint.pth')))
        self.model.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            X_his = X_his.to(args.device)
            x = self.model.get_hidden(X_his)
        return x

class ServerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        set_seed(args.seed)
        self.model_elec = self._build_model(args).to(args.device)
        self.model_steam = self._build_model(args).to(args.device)
        self.model_chill = self._build_model(args).to(args.device)

    def _build_model(self, args):
        if args.server_model == "MLP":
            return MLP_server(args)
        elif args.server_model == "Diff(TimeXer)":
            return TimeXerDiff(args)
        elif args.server_model == "Diff(MLP)":
            return MLPDiff(args)
        else:
            raise ValueError(f"Unknown model type: {args.server_model}")
    
    # def residual_process(self, args, setting, load, data_set):
    #     X_his = data_set.X_his
    #     X_ex = data_set.X_ex
    #     y = data_set.y
    #     local_model = LocalModel(args)
    #     hidden = local_model.embedding(args, setting, load, X_his).cpu().detach()
    #     res = torch.tensor(y) - local_model.predict(args, setting, load, X_his, X_ex)[0].cpu().detach()
    #     return hidden, X_ex, res
    
    def residual_process(self, args, setting, load, data_set, batch_size=256):
        """
        分批处理数据以减少内存占用
        Args:
            batch_size: 每批处理的数据量，根据GPU内存调整
        """
        X_his = data_set.X_his
        X_ex = data_set.X_ex
        y = data_set.y
        local_model = LocalModel(args)
        
        # 初始化结果容器
        hidden_parts = []
        res_parts = []
        
        # 分批处理
        num_samples = len(X_his)
        for i in range(0, num_samples, batch_size):
            # 获取当前批次
            batch_his = X_his[i:i+batch_size].to(args.device)
            batch_ex = X_ex[i:i+batch_size].to(args.device)
            batch_y = y[i:i+batch_size]
            
            # 处理当前批次
            with torch.no_grad():
                batch_hidden = local_model.embedding(args, setting, load, batch_his).cpu()
                batch_pred = local_model.predict(args, setting, load, batch_his, batch_ex)[0].cpu()
                batch_res = torch.tensor(batch_y) - batch_pred
                
            # 保存结果
            hidden_parts.append(batch_hidden)
            res_parts.append(batch_res)
            
            # 及时释放显存
            del batch_his, batch_ex, batch_hidden, batch_pred
            torch.cuda.empty_cache()
        
        # 合并所有批次结果
        hidden = torch.cat(hidden_parts, dim=0)
        res = torch.cat(res_parts, dim=0)
        
        return hidden, X_ex, res  # X_ex保持原样返回（通常已分批加载）

    
    def _get_data(self, args, setting, load, goal):
        train_data, train_loader = get_data(args, flag='train', load=load)
        val_data, val_loader = get_data(args, flag='val', load=load)
        test_data, test_loader = get_data(args, flag='test', load=load)
        hidden_train, ex_train, res_train = self.residual_process(args, setting, load, train_data)
        hidden_val, ex_val, res_val = self.residual_process(args, setting, load, val_data)
        hidden_test, ex_test, res_test = self.residual_process(args, setting, load, test_data)
        if goal == 'train':
            hidden = torch.cat((hidden_train, hidden_val), dim=0)
            ex = torch.cat((ex_train, ex_val), dim=0)
            res = torch.cat((res_train, res_val), dim=0)
            shuffle = True
            if args.add_noise:
                noise_injection = NoiseInjectionModule()
                res = noise_injection.inject_noise(res,args.noise_type,args.noise_ratio)
        elif goal == 'val':
            hidden = hidden_val
            ex = ex_val
            res = res_val
            shuffle = True
        elif goal == 'test':
            hidden = hidden_test
            ex = ex_test
            res = res_test
            shuffle = False
            if args.add_noise:
                noise_injection = NoiseInjectionModule()
                res = noise_injection.inject_noise(res,args.noise_type,args.noise_ratio)

        data_set = DiffDataset(hidden, ex, res)
        data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=shuffle)
        return data_set, data_loader
    
    def multi_task_train(self, args, setting):
        ## Load data
        elec_train_data, elec_train_loader = self._get_data(args, setting, load='Electricity',goal='train')
        steam_train_data, steam_train_loader = self._get_data(args, setting, load='Steam',goal='train')
        chill_train_data, chill_train_loader = self._get_data(args, setting, load='Chillwater',goal='train')
        elec_path = os.path.join(args.checkpoints, setting, 'server_Electricity')
        steam_path = os.path.join(args.checkpoints, setting, 'server_Steam')
        chill_path = os.path.join(args.checkpoints, setting, 'server_Chillwater')
        if not os.path.exists(elec_path):
            os.makedirs(elec_path)
        if not os.path.exists(steam_path):
            os.makedirs(steam_path)
        if not os.path.exists(chill_path):
            os.makedirs(chill_path)

        time_now = time.time()
        train_steps = len(elec_train_loader)

        optimizer = torch.optim.Adam(list(self.model_elec.parameters())
                                     +list(self.model_steam.parameters())
                                     +list(self.model_chill.parameters()), lr=args.lr)
        criterion = cauchy_likelihood()
        schedule = cosine_schedule(args.num_steps)
        alphas = schedule['alphas'].to(args.device)
        betas = schedule['betas'].to(args.device)
        alphas_bar = schedule['alpha_bars'].to(args.device)
        alphas_bar_sqrt = torch.sqrt(alphas_bar).to(args.device)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_bar).to(args.device)

        train_losses = []
        elec_losses = []
        steam_losses = []
        chill_losses = []
        val_losses = []
        # Training loop
        for epoch in range(args.epochs_server):
            iter_count = 0
            train_loss = []
            elec_loss0 = []
            steam_loss0 = []
            chill_loss0 = []
            self.model_elec.train(), self.model_steam.train(), self.model_chill.train()
            epoch_time = time.time()
            for i, (elec_data, steam_data, chill_data) in enumerate(zip(elec_train_loader, steam_train_loader, chill_train_loader)):
                """
                X_his: [batch_size, seq_len]
                X_ex: [batch_size, label_len, feature_num(6)]
                y: [batch_size, label_len]
                """
                iter_count += 1
                optimizer.zero_grad()
                elec_his=elec_data[0].to(args.device)
                elec_ex=elec_data[1].to(args.device)
                elec_label=elec_data[2].to(args.device)
                steam_his=steam_data[0].to(args.device)
                steam_ex=steam_data[1].to(args.device)
                steam_label=steam_data[2].to(args.device)
                chill_his=chill_data[0].to(args.device) #(batch_size,seq_len,1)
                chill_ex=chill_data[1].to(args.device) #(batch_size,pred_len,6)
                chill_label=chill_data[2].to(args.device) #(batch_size,pred_len,1)

                if args.server_model == "Diff(TimeXer)" or args.server_model == "Diff(MLP)":
                    combined_his = torch.cat([elec_his.unsqueeze(-1),steam_his.unsqueeze(-1),chill_his.unsqueeze(-1)],dim=-1) #(batch_size,hidden_dim,3)
                    elec_loss = diffusion_loss_fn(self.model_elec, elec_label.unsqueeze(-1), alphas_bar_sqrt, one_minus_alphas_bar_sqrt, args.num_steps-1, combined_his,elec_ex)
                    steam_loss = diffusion_loss_fn(self.model_steam, steam_label.unsqueeze(-1), alphas_bar_sqrt, one_minus_alphas_bar_sqrt, args.num_steps-1, combined_his,steam_ex)   
                    chill_loss = diffusion_loss_fn(self.model_chill, chill_label.unsqueeze(-1), alphas_bar_sqrt, one_minus_alphas_bar_sqrt, args.num_steps-1, combined_his,chill_ex)
                elif args.server_model == "MLP":
                    combined_his = torch.cat([elec_his,steam_his,chill_his],dim=-1) #(batch_size,hidden_dim*3)
                    elec_mu, elec_sigma = self.model_elec(combined_his, elec_ex)
                    steam_mu, steam_sigma = self.model_steam(combined_his, steam_ex)
                    chill_mu, chill_sigma = self.model_chill(combined_his, chill_ex)
                    elec_loss = criterion(elec_label, elec_mu, elec_sigma)
                    steam_loss = criterion(steam_label, steam_mu, steam_sigma)
                    chill_loss = criterion(chill_label, chill_mu, chill_sigma)
                    
                loss = elec_loss + steam_loss + chill_loss #multi-task loss
                elec_loss0.append(elec_loss.item())
                steam_loss0.append(steam_loss.item())
                chill_loss0.append(chill_loss.item())
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            elec_loss0 = np.average(elec_loss0)
            elec_losses.append(elec_loss0)
            steam_loss0 = np.average(steam_loss0)
            steam_losses.append(steam_loss0)
            chill_loss0 = np.average(chill_loss0)
            chill_losses.append(chill_loss0)
            train_loss = np.average(train_loss)
            train_losses.append(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))

        plot_train_loss(train_losses)
        elec_best_model_path = elec_path + '/' + 'checkpoint.pth'
        steam_best_model_path = steam_path + '/' + 'checkpoint.pth'
        chill_best_model_path = chill_path + '/' + 'checkpoint.pth'
        torch.save(self.model_elec.state_dict(), elec_best_model_path)
        torch.save(self.model_steam.state_dict(), steam_best_model_path)
        torch.save(self.model_chill.state_dict(), chill_best_model_path)

        self.model_elec.load_state_dict(torch.load(elec_best_model_path))
        self.model_steam.load_state_dict(torch.load(steam_best_model_path))
        self.model_chill.load_state_dict(torch.load(chill_best_model_path))

        return elec_losses, steam_losses, chill_losses, train_losses
    


class GlobalSample:
    def __init__(self, args):
        set_seed(args.seed)
        self.local_elec = LocalModel(args)
        self.local_steam = LocalModel(args)
        self.local_chill = LocalModel(args)
        
        self.server = ServerModel(args)
    
    def _model_loading(self, args, setting):
        self.local_elec.model.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'local_Electricity', 'checkpoint.pth')))
        self.local_steam.model.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'local_Steam', 'checkpoint.pth')))
        self.local_chill.model.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'local_Chillwater', 'checkpoint.pth')))
        self.server.model_elec.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'server_Electricity', 'checkpoint.pth')))
        self.server.model_steam.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'server_Steam', 'checkpoint.pth')))
        self.server.model_chill.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'server_Chillwater', 'checkpoint.pth')))

    def _get_data(self, args, flag, load):
        data_set, data_loader = get_data(args,flag,load)
        return data_set, data_loader
    
    def sample(self, args, setting):
        elec_val_data, elec_val_loader = self._get_data(args, flag='val', load='Electricity')
        steam_val_data, steam_val_loader = self._get_data(args, flag='val', load='Steam')
        chill_val_data, chill_val_loader = self._get_data(args, flag='val', load='Chillwater')
        elec_test_data, elec_test_loader = self._get_data(args, flag='test', load='Electricity')
        steam_test_data, steam_test_loader = self._get_data(args, flag='test', load='Steam')
        chill_test_data, chill_test_loader = self._get_data(args, flag='test', load='Chillwater')
        # load server model
        self._model_loading(args, setting)
        self.local_elec.model.to(args.device), self.local_steam.model.to(args.device), self.local_chill.model.to(args.device)
        self.server.model_elec.to(args.device), self.server.model_steam.to(args.device), self.server.model_chill.to(args.device)
        self.local_elec.model.eval(), self.local_steam.model.eval(), self.local_chill.model.eval(), self.server.model_elec.eval(), self.server.model_steam.eval(), self.server.model_chill.eval()   
        with torch.no_grad():
            elec_mus = []
            elec_actuals = []
            elec_local_sigmas = []
            elec_server_sigmas = []
            steam_mus = []
            steam_actuals = []
            steam_local_sigmas = []
            steam_server_sigmas = []
            chill_mus = []
            chill_actuals = []
            chill_local_sigmas = []
            chill_server_sigmas = []
            for i, (elec_data, steam_data, chill_data) in enumerate(zip(elec_val_loader, steam_val_loader, chill_val_loader)):
                """
                X_his: [batch_size, seq_len]
                X_ex: [batch_size, label_len, feature_num(6)]
                y: [batch_size, label_len]
                """
                elec_his=elec_data[0]
                elec_ex=elec_data[1]
                elec_label=elec_data[2]
                steam_his=steam_data[0]
                steam_ex=steam_data[1]
                steam_label=steam_data[2]
                chill_his=chill_data[0] #(batch_size,seq_len,1)
                chill_ex=chill_data[1] #(batch_size,pred_len,6)
                chill_label=chill_data[2] #(batch_size,pred_len,1)

                elec_local_mu, elec_local_sigma = self.local_elec.predict(args, setting, load='Electricity', X_his=elec_his, X_ex=elec_ex)
                steam_local_mu, steam_local_sigma = self.local_steam.predict(args, setting, load='Steam', X_his=steam_his, X_ex=steam_ex)
                chill_local_mu, chill_local_sigma = self.local_chill.predict(args, setting, load='Chillwater', X_his=chill_his, X_ex=chill_ex)

                elec_res = torch.tensor(elec_label).to(args.device) - elec_local_mu
                steam_res = torch.tensor(steam_label).to(args.device) - steam_local_mu
                chill_res = torch.tensor(chill_label).to(args.device) - chill_local_mu
                elec_hidden = self.local_elec.embedding(args, setting, load='Electricity', X_his=elec_his)
                steam_hidden = self.local_steam.embedding(args, setting, load='Steam', X_his=steam_his)
                chill_hidden = self.local_chill.embedding(args, setting, load='Chillwater', X_his=chill_his)

                if args.server_model == "Diff(TimeXer)" or args.server_model == "Diff(MLP)":
                    combined_hidden = torch.cat([elec_hidden.unsqueeze(-1),steam_hidden.unsqueeze(-1),chill_hidden.unsqueeze(-1)],dim=-1) #(batch_size,hidden_dim,3)
                    elec_server_mu,_,elec_server_sigma = diffusion_sample(args,self.server.model_elec,combined_hidden,elec_res.unsqueeze(-1),elec_ex)
                    steam_server_mu,_,steam_server_sigma = diffusion_sample(args,self.server.model_steam,combined_hidden,steam_res.unsqueeze(-1),steam_ex)
                    chill_server_mu,_,chill_server_sigma = diffusion_sample(args,self.server.model_chill,combined_hidden,chill_res.unsqueeze(-1),chill_ex)
                    elec_server_mu, elec_server_sigma = elec_server_mu.squeeze(-1),elec_server_sigma.squeeze(-1)
                    steam_server_mu, steam_server_sigma = steam_server_mu.squeeze(-1),steam_server_sigma.squeeze(-1)
                    chill_server_mu, chill_server_sigma = chill_server_mu.squeeze(-1),chill_server_sigma.squeeze(-1)
                elif args.server_model == "MLP":
                    combined_hidden = torch.cat([elec_hidden,steam_hidden,chill_hidden],dim=-1) #(batch_size,hidden_dim*3)
                    elec_server_mu, elec_server_sigma = self.server.model_elec(combined_hidden, elec_ex)
                    steam_server_mu, steam_server_sigma = self.server.model_steam(combined_hidden, steam_ex)
                    chill_server_mu, chill_server_sigma = self.server.model_chill(combined_hidden, chill_ex)

                elec_final_mu = elec_local_mu + elec_server_mu
                steam_final_mu = steam_local_mu + steam_server_mu
                chill_final_mu = chill_local_mu + chill_server_mu

                elec_mus.append(elec_final_mu.cpu())
                elec_local_sigmas.append(elec_local_sigma.cpu())
                elec_server_sigmas.append(elec_server_sigma.cpu())
                elec_actuals.append(elec_label.cpu())

                steam_mus.append(steam_final_mu.cpu())
                steam_local_sigmas.append(steam_local_sigma.cpu())
                steam_server_sigmas.append(steam_server_sigma.cpu())
                steam_actuals.append(steam_label.cpu())

                chill_mus.append(chill_final_mu.cpu())
                chill_local_sigmas.append(chill_local_sigma.cpu())
                chill_server_sigmas.append(chill_server_sigma.cpu())
                chill_actuals.append(chill_label.cpu())
            
            elec_mus = torch.cat(elec_mus)
            elec_local_sigmas = torch.cat(elec_local_sigmas)
            elec_server_sigmas = torch.cat(elec_server_sigmas)
            elec_actuals = torch.cat(elec_actuals)

            steam_mus = torch.cat(steam_mus)
            steam_local_sigmas = torch.cat(steam_local_sigmas)
            steam_server_sigmas = torch.cat(steam_server_sigmas)
            steam_actuals = torch.cat(steam_actuals)

            chill_mus = torch.cat(chill_mus)
            chill_local_sigmas = torch.cat(chill_local_sigmas)
            chill_server_sigmas = torch.cat(chill_server_sigmas)
            chill_actuals = torch.cat(chill_actuals)

            # 寻找最优系数 sigma = sigma_local + w * sigma_server
            w_range = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
            best_w_elec,_ = find_optimal(elec_mus,elec_local_sigmas,elec_server_sigmas,elec_actuals,w_range)
            best_w_steam,_ = find_optimal(steam_mus,steam_local_sigmas,steam_server_sigmas,steam_actuals,w_range)
            best_w_chill,_ = find_optimal(chill_mus,chill_local_sigmas,chill_server_sigmas,chill_actuals,w_range)

            elec_mus = []
            elec_actuals = []
            elec_sigmas = []
            steam_mus = []
            steam_actuals = []
            steam_sigmas = []
            chill_mus = []
            chill_actuals = []
            chill_sigmas = []
            for i, (elec_data, steam_data, chill_data) in enumerate(zip(elec_test_loader, steam_test_loader, chill_test_loader)):
                """
                X_his: [batch_size, seq_len]
                X_ex: [batch_size, label_len, feature_num(6)]
                y: [batch_size, label_len]
                """
                elec_his=elec_data[0]
                elec_ex=elec_data[1]
                elec_label=elec_data[2]
                steam_his=steam_data[0]
                steam_ex=steam_data[1]
                steam_label=steam_data[2]
                chill_his=chill_data[0] #(batch_size,seq_len,1)
                chill_ex=chill_data[1] #(batch_size,pred_len,6)
                chill_label=chill_data[2] #(batch_size,pred_len,1)

                elec_local_mu, elec_local_sigma = self.local_elec.predict(args, setting, load='Electricity', X_his=elec_his, X_ex=elec_ex)
                steam_local_mu, steam_local_sigma = self.local_steam.predict(args, setting, load='Steam', X_his=steam_his, X_ex=steam_ex)
                chill_local_mu, chill_local_sigma = self.local_chill.predict(args, setting, load='Chillwater', X_his=chill_his, X_ex=chill_ex)

                elec_res = torch.tensor(elec_label).to(args.device) - elec_local_mu
                steam_res = torch.tensor(steam_label).to(args.device) - steam_local_mu
                chill_res = torch.tensor(chill_label).to(args.device) - chill_local_mu
                elec_hidden = self.local_elec.embedding(args, setting, load='Electricity', X_his=elec_his)
                steam_hidden = self.local_steam.embedding(args, setting, load='Steam', X_his=steam_his)
                chill_hidden = self.local_chill.embedding(args, setting, load='Chillwater', X_his=chill_his)

                if args.server_model == "Diff(TimeXer)" or args.server_model == "Diff(MLP)":
                    combined_hidden = torch.cat([elec_hidden.unsqueeze(-1),steam_hidden.unsqueeze(-1),chill_hidden.unsqueeze(-1)],dim=-1) #(batch_size,hidden_dim,3)
                    elec_server_mu,_,elec_server_sigma = diffusion_sample(args,self.server.model_elec,combined_hidden,elec_res.unsqueeze(-1),elec_ex)
                    steam_server_mu,_,steam_server_sigma = diffusion_sample(args,self.server.model_steam,combined_hidden,steam_res.unsqueeze(-1),steam_ex)
                    chill_server_mu,_,chill_server_sigma = diffusion_sample(args,self.server.model_chill,combined_hidden,chill_res.unsqueeze(-1),chill_ex)
                    elec_server_mu, elec_server_sigma = elec_server_mu.squeeze(-1),elec_server_sigma.squeeze(-1)
                    steam_server_mu, steam_server_sigma = steam_server_mu.squeeze(-1),steam_server_sigma.squeeze(-1)
                    chill_server_mu, chill_server_sigma = chill_server_mu.squeeze(-1),chill_server_sigma.squeeze(-1)
                elif args.server_model == "MLP":
                    combined_hidden = torch.cat([elec_hidden,steam_hidden,chill_hidden],dim=-1) #(batch_size,hidden_dim*3)
                    elec_server_mu, elec_server_sigma = self.server.model_elec(combined_hidden, elec_ex)
                    steam_server_mu, steam_server_sigma = self.server.model_steam(combined_hidden, steam_ex)
                    chill_server_mu, chill_server_sigma = self.server.model_chill(combined_hidden, chill_ex)

                elec_final_mu = elec_local_mu + elec_server_mu
                steam_final_mu = steam_local_mu + steam_server_mu
                chill_final_mu = chill_local_mu + chill_server_mu
                elec_final_sigma = elec_local_sigma + best_w_elec * elec_server_sigma
                steam_final_sigma = steam_local_sigma + best_w_steam * steam_server_sigma
                chill_final_sigma = chill_local_sigma + best_w_chill * chill_server_sigma

                elec_mus.append(elec_final_mu.cpu())
                elec_sigmas.append(elec_final_sigma.cpu())
                elec_actuals.append(elec_label.cpu())
                steam_mus.append(steam_final_mu.cpu())
                steam_sigmas.append(steam_final_sigma.cpu())
                steam_actuals.append(steam_label.cpu())
                chill_mus.append(chill_final_mu.cpu())
                chill_sigmas.append(chill_final_sigma.cpu())
                chill_actuals.append(chill_label.cpu())
            
            elec_mus = np.concatenate(elec_mus, axis=0)
            elec_sigmas = np.concatenate(elec_sigmas, axis=0)
            elec_actuals = np.concatenate(elec_actuals, axis=0)
            steam_mus = np.concatenate(steam_mus, axis=0)
            steam_sigmas = np.concatenate(steam_sigmas, axis=0)
            steam_actuals = np.concatenate(steam_actuals, axis=0)
            chill_mus = np.concatenate(chill_mus, axis=0)
            chill_sigmas = np.concatenate(chill_sigmas, axis=0)
            chill_actuals = np.concatenate(chill_actuals, axis=0)

            elec_mu = elec_test_data.inverse_transform(elec_mus).reshape(-1)
            elec_sigma = elec_test_data.scale_() * elec_sigmas
            elec_sigma = elec_sigma.reshape(-1)
            elec_actual = elec_test_data.inverse_transform(elec_actuals).reshape(-1)
            steam_mu = steam_test_data.inverse_transform(steam_mus).reshape(-1)
            steam_sigma = steam_test_data.scale_() * steam_sigmas
            steam_sigma = steam_sigma.reshape(-1)
            steam_actual = steam_test_data.inverse_transform(steam_actuals).reshape(-1)
            chill_mu = chill_test_data.inverse_transform(chill_mus).reshape(-1)
            chill_sigma = chill_test_data.scale_() * chill_sigmas
            chill_sigma = chill_sigma.reshape(-1)
            chill_actual = chill_test_data.inverse_transform(chill_actuals).reshape(-1)
            
            for load in ['Electricity', 'Steam', 'Chillwater']:
                foler_path = './test_results/' + setting +'/'+ 'server_' + load + '/'
                if not os.path.exists(foler_path):
                    os.makedirs(foler_path)
                if load == 'Electricity':
                    mu = elec_mu
                    sigma = elec_sigma
                    actual = elec_actual
                elif load == 'Steam':
                    mu = steam_mu
                    sigma = steam_sigma
                    actual = steam_actual
                elif load == 'Chillwater':
                    mu = chill_mu
                    sigma = chill_sigma
                    actual = chill_actual
                mape, rmse, mae, pinball_loss, crps, winkler_score_25, winkler_score_50, winkler_score_75 = metric(args, mu, sigma, actual)
                np.save(foler_path + 'mu.npy', mu)
                np.save(foler_path + 'sigma.npy', sigma)
                np.save(foler_path + 'actual.npy', actual)
                print(f'type:{i}, mape:{mape}, rmse:{rmse}, mae:{mae}, pinball_loss:{pinball_loss}')
                csv_file = foler_path + 'metrics.csv'
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # 写入表头
                    writer.writerow(['Setting', 'MAPE', 'MAE', 'RMSE', 'PINBALL_LOSS', 'CRPS', 
                                    'WINKLER_SCORE_25', 'WINKLER_SCORE_50', 'WINKLER_SCORE_75'])
                    # 写入当前数据
                    writer.writerow([setting, mape, mae, rmse, pinball_loss, crps, 
                                    winkler_score_25, winkler_score_50, winkler_score_75])

            return
   
class StepbyStepSample:
    def __init__(self, args):
        set_seed(args.seed)
        self.local_elec = LocalModel(args)
        self.local_steam = LocalModel(args)
        self.local_chill = LocalModel(args)
        
        self.server = ServerModel(args)
    
    def _model_loading(self, args, setting):
        self.local_elec.model.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'local_Electricity', 'checkpoint.pth')))
        self.local_steam.model.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'local_Steam', 'checkpoint.pth')))
        self.local_chill.model.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'local_Chillwater', 'checkpoint.pth')))
        self.server.model_elec.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'server_Electricity', 'checkpoint.pth')))
        self.server.model_steam.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'server_Steam', 'checkpoint.pth')))
        self.server.model_chill.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'server_Chillwater', 'checkpoint.pth')))

    def _get_data(self, args, flag, load):
        data_set, data_loader = get_data(args,flag,load)
        return data_set, data_loader
    
    def sample(self, args, setting):
        schedule = cosine_schedule(args.num_steps)
        alphas = schedule['alphas'].to(args.device)
        betas = schedule['betas'].to(args.device)
        alphas_bar = schedule['alpha_bars'].to(args.device)
        alphas_bar_sqrt = torch.sqrt(alphas_bar).to(args.device)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_bar).to(args.device)
        elec_test_data, elec_test_loader = self._get_data(args, flag='test', load='Electricity')
        steam_test_data, steam_test_loader = self._get_data(args, flag='test', load='Steam')
        chill_test_data, chill_test_loader = self._get_data(args, flag='test', load='Chillwater')
        # load server model
        self._model_loading(args, setting)
        self.local_elec.model.to(args.device), self.local_steam.model.to(args.device), self.local_chill.model.to(args.device)
        self.server.model_elec.to(args.device), self.server.model_steam.to(args.device), self.server.model_chill.to(args.device)
        self.local_elec.model.eval(), self.local_steam.model.eval(), self.local_chill.model.eval(), self.server.model_elec.eval(), self.server.model_steam.eval(), self.server.model_chill.eval()   
        with torch.no_grad():
            elec_his=elec_test_data.X_his[:100]
            elec_ex=elec_test_data.X_ex[:100]
            elec_label=elec_test_data.y[:100]
            steam_his=steam_test_data.X_his[:100]
            steam_ex=steam_test_data.X_ex[:100]
            steam_label=steam_test_data.y[:100]
            chill_his=chill_test_data.X_his[:100] #(batch_size,seq_len,1)
            chill_ex=chill_test_data.X_ex[:100] #(batch_size,pred_len,6)
            chill_label=chill_test_data.y[:100]#(batch_size,pred_len,1)
            print(elec_his.shape)

            elec_local_mu, elec_local_sigma = self.local_elec.predict(args, setting, load='Electricity', X_his=elec_his, X_ex=elec_ex)
            steam_local_mu, steam_local_sigma = self.local_steam.predict(args, setting, load='Steam', X_his=steam_his, X_ex=steam_ex)
            chill_local_mu, chill_local_sigma = self.local_chill.predict(args, setting, load='Chillwater', X_his=chill_his, X_ex=chill_ex)

            elec_res = torch.tensor(elec_label).to(args.device) - elec_local_mu
            steam_res = torch.tensor(steam_label).to(args.device) - steam_local_mu
            chill_res = torch.tensor(chill_label).to(args.device) - chill_local_mu
            elec_hidden = self.local_elec.embedding(args, setting, load='Electricity', X_his=elec_his)
            steam_hidden = self.local_steam.embedding(args, setting, load='Steam', X_his=steam_his)
            chill_hidden = self.local_chill.embedding(args, setting, load='Chillwater', X_his=chill_his)

            combined_hidden = torch.cat([elec_hidden.unsqueeze(-1),steam_hidden.unsqueeze(-1),chill_hidden.unsqueeze(-1)],dim=-1) #(batch_size,hidden_dim,3)
            elec_x_t = record_sample_loop(self.server.model_elec, elec_res.unsqueeze(-1).shape, args.num_steps, betas, alphas, alphas_bar, combined_hidden, elec_ex)
            steam_x_t = record_sample_loop(self.server.model_steam, steam_res.unsqueeze(-1).shape, args.num_steps, betas, alphas, alphas_bar, combined_hidden, steam_ex)
            chill_x_t = record_sample_loop(self.server.model_chill, chill_res.unsqueeze(-1).shape, args.num_steps, betas, alphas, alphas_bar, combined_hidden, chill_ex)

        return elec_x_t, steam_x_t, chill_x_t
        
def record_sample_loop(model, shape, n_steps, betas, alphas, alphas_bar, his, c):
    device = next(model.parameters()).device
    x = torch.randn(shape).to(device)
    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_bar = alphas_bar.to(device)
    x_t = []
    for t in range(n_steps - 1, -1, -1):
        z = torch.randn_like(x)
        t_tensor = torch.tensor(t, device=device).repeat(x.shape[0])
        x0_theta,sigma0_conditoned = model(x, t_tensor, his, c)
 
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
        x_t.append(x.squeeze(-1))
    x_t = torch.stack(x_t,dim=0).permute(1,0,2) # （50,100,24)
    return x_t

class CentralModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        set_seed(args.seed)
        self.model = self._build_model(args).to(args.device)

    def _build_model(self, args):
        if args.central_model == "MLP":
            return MLP_central(args)
        # elif args.central_model == "Diff(TimeXer)":
        #     return TimeXerDiff_central(args)
        elif args.central_model == "GRU":
            return GRU_central(args)
        # elif args.central_model == "Diff(MLP)":
        #     return MLPDiff_central(args)
        elif args.central_model == "Bayes":
            return Bayes_central(args)
        else:
            raise ValueError(f"Unknown model type: {args.local_model}")
    
    def _get_data(self, args, flag, load):
        data_set, data_loader = get_data(args,flag,load)
        return data_set, data_loader


    def train(self, args, setting):
        # Load data
        elec_train_data, elec_train_loader = self._get_data(args, flag='train', load='Electricity')
        steam_train_data, steam_train_loader = self._get_data(args, flag='train', load='Steam')
        chill_train_data, chill_train_loader = self._get_data(args, flag='train', load='Chillwater')
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(elec_train_loader)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        criterion = cauchy_likelihood()
        
        train_losses = []
        # Training loop
        for epoch in range(args.epochs_central):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (elec_data, steam_data, chill_data) in enumerate(zip(elec_train_loader, steam_train_loader, chill_train_loader)):
                """
                X_his: [batch_size, seq_len]
                X_ex: [batch_size, label_len, feature_num(6)]
                y: [batch_size, label_len]
                """
                iter_count += 1
                optimizer.zero_grad()
                elec_his=elec_data[0].to(args.device)
                elec_ex=elec_data[1].to(args.device)
                elec_label=elec_data[2].to(args.device)
                steam_his=steam_data[0].to(args.device)
                steam_ex=steam_data[1].to(args.device)
                steam_label=steam_data[2].to(args.device)
                chill_his=chill_data[0].to(args.device) #(batch_size,seq_len)
                chill_ex=chill_data[1].to(args.device) #(batch_size,pred_len,6)
                chill_label=chill_data[2].to(args.device) #(batch_size,pred_len)

                combined_his = torch.cat([elec_his.unsqueeze(-1),steam_his.unsqueeze(-1),chill_his.unsqueeze(-1)],dim=-1) #(batch_size,seq_len,3)
                ex = elec_ex.reshape(elec_ex.shape[0], -1).unsqueeze(-1) #(batch_size,6*24,1)
                combined_label = torch.cat([elec_label.unsqueeze(-1), steam_label.unsqueeze(-1), chill_label.unsqueeze(-1)], dim=-1) #(batch_size,pred_len,3)
                # if args.central_model == "Diff(TimeXer)" or args.central_model == "Diff(MLP)":
                #     loss = diffusion_loss_fn(self.model, combined_label, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, args.num_steps-1, combined_his, elec_ex)

                combined_mu, combined_sigma = self.model(combined_his, ex)
                elec_mu, steam_mu, chill_mu = torch.split(combined_mu, 1, dim=-1) #(batch_size,seq_length,1)
                elec_sigma, steam_sigma, chill_sigma = torch.split(combined_sigma, 1, dim=-1) #(batch_size,seq_length,1)
                elec_loss = criterion(elec_label, elec_mu.squeeze(-1), elec_sigma.squeeze(-1))
                steam_loss = criterion(steam_label, steam_mu.squeeze(-1), steam_sigma.squeeze(-1))
                chill_loss = criterion(chill_label, chill_mu.squeeze(-1), chill_sigma.squeeze(-1))
                loss = elec_loss + steam_loss + chill_loss
                
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_losses.append(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))

        plot_train_loss(train_losses)
        best_model_path = path + '/' + 'checkpoint.pth'
        torch.save(self.model.state_dict(), best_model_path)
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    

    def test(self, args, setting, test=0):
        elec_test_data, elec_test_loader = self._get_data(args, flag='test', load='Electricity')
        steam_test_data, steam_test_loader = self._get_data(args, flag='test', load='Steam')
        chill_test_data, chill_test_loader = self._get_data(args, flag='test', load='Chillwater')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'checkpoint.pth')))
        
        elec_mus = []
        elec_sigmas = []
        elec_actuals = []
        steam_mus = []
        steam_sigmas = []
        steam_actuals = []
        chill_mus = []
        chill_sigmas = []
        chill_actuals = []

        self.model.eval()
        with torch.no_grad():
            for i, (elec_data, steam_data, chill_data) in enumerate(zip(elec_test_loader, steam_test_loader, chill_test_loader)):
                """
                X_his: [batch_size, seq_len]
                X_ex: [batch_size, label_len, feature_num(6)]
                y: [batch_size, label_len]
                """
                elec_his=elec_data[0].to(args.device)
                elec_ex=elec_data[1].to(args.device)
                elec_label=elec_data[2].to(args.device)
                steam_his=steam_data[0].to(args.device)
                steam_ex=steam_data[1].to(args.device)
                steam_label=steam_data[2].to(args.device)
                chill_his=chill_data[0].to(args.device) #(batch_size,seq_len)
                chill_ex=chill_data[1].to(args.device) #(batch_size,pred_len,6)
                chill_label=chill_data[2].to(args.device) #(batch_size,pred_len)

                combined_his = torch.cat([elec_his.unsqueeze(-1),steam_his.unsqueeze(-1),chill_his.unsqueeze(-1)],dim=-1) #(batch_size,seq_len,3)
                ex = elec_ex.reshape(elec_ex.shape[0], -1).unsqueeze(-1) #(batch_size,6*24,1)
                combined_label = torch.cat([elec_label.unsqueeze(-1), steam_label.unsqueeze(-1), chill_label.unsqueeze(-1)], dim=-1) #(batch_size,pred_len,3)

                # if args.central_model == "Diff(TimeXer)" or args.central_model == "Diff(MLP)":
                #     combined_mu,_,combined_sigma = diffusion_sample(args,self.model,combined_his,combined_label,elec_ex)

                combined_mu, combined_sigma = self.model(combined_his, ex)

                elec_mu, steam_mu, chill_mu = torch.split(combined_mu, 1, dim=-1) #(batch_size,seq_length,1)
                elec_sigma, steam_sigma, chill_sigma = torch.split(combined_sigma, 1, dim=-1) #(batch_size,seq_length,1)

                elec_mu, steam_mu, chill_mu = elec_mu.squeeze(-1).detach().cpu(), steam_mu.squeeze(-1).detach().cpu(), chill_mu.squeeze(-1).detach().cpu()
                elec_sigma, steam_sigma, chill_sigma = elec_sigma.squeeze(-1).detach().cpu(), steam_sigma.squeeze(-1).detach().cpu(), chill_sigma.squeeze(-1).detach().cpu()
                elec_label, steam_label, chill_label = elec_label.squeeze(-1).detach().cpu(), steam_label.squeeze(-1).detach().cpu(), chill_label.squeeze(-1).detach().cpu()
                
                elec_mus.append(elec_mu)
                elec_sigmas.append(elec_sigma)
                elec_actuals.append(elec_label)
                steam_mus.append(steam_mu) 
                steam_sigmas.append(steam_sigma)
                steam_actuals.append(steam_label)
                chill_mus.append(chill_mu)
                chill_sigmas.append(chill_sigma)
                chill_actuals.append(chill_label)

            elec_mus_normed = np.concatenate(elec_mus, axis=0)
            elec_sigmas_normed = np.concatenate(elec_sigmas, axis=0)
            elec_actuals_normed = np.concatenate(elec_actuals, axis=0)
            steam_mus_normed = np.concatenate(steam_mus, axis=0)
            steam_sigmas_normed = np.concatenate(steam_sigmas, axis=0)  
            steam_actuals_normed = np.concatenate(steam_actuals, axis=0)
            chill_mus_normed = np.concatenate(chill_mus, axis=0)
            chill_sigmas_normed = np.concatenate(chill_sigmas, axis=0)
            chill_actuals_normed = np.concatenate(chill_actuals, axis=0)

            elec_pred_mu = elec_test_data.inverse_transform(elec_mus_normed)
            elec_pred_sigma = elec_test_data.scale_() * elec_sigmas_normed
            elec_actual = elec_test_data.inverse_transform(elec_actuals_normed)
            steam_pred_mu = steam_test_data.inverse_transform(steam_mus_normed)
            steam_pred_sigma = steam_test_data.scale_() * steam_sigmas_normed
            steam_actual = steam_test_data.inverse_transform(steam_actuals_normed)
            chill_pred_mu = chill_test_data.inverse_transform(chill_mus_normed)
            chill_pred_sigma = chill_test_data.scale_() * chill_sigmas_normed
            chill_actual = chill_test_data.inverse_transform(chill_actuals_normed)

            elec_pred_mu = elec_pred_mu.reshape(-1)
            elec_pred_sigma = elec_pred_sigma.reshape(-1)
            elec_actual = elec_actual.reshape(-1)
            steam_pred_mu = steam_pred_mu.reshape(-1)
            steam_pred_sigma = steam_pred_sigma.reshape(-1)
            steam_actual = steam_actual.reshape(-1)
            chill_pred_mu = chill_pred_mu.reshape(-1)
            chill_pred_sigma = chill_pred_sigma.reshape(-1)
            chill_actual = chill_actual.reshape(-1)

            for load in ['Electricity', 'Steam', 'Chillwater']:
                if load == 'Electricity':
                    pred_mu = elec_pred_mu
                    pred_sigma = elec_pred_sigma
                    actual = elec_actual
                elif load == 'Steam':
                    pred_mu = steam_pred_mu
                    pred_sigma = steam_pred_sigma
                    actual = steam_actual
                elif load == 'Chillwater':
                    pred_mu = chill_pred_mu
                    pred_sigma = chill_pred_sigma
                    actual = chill_actual
                folder_path = './test_results/' + setting +'/'+ load + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                mape, rmse, mae, pinball_loss, crps, winkler_score_25, winkler_score_50, winkler_score_75 = metric(args, pred_mu, pred_sigma, actual)


                np.save(folder_path + 'mu.npy', pred_mu)
                np.save(folder_path + 'sigma.npy', pred_sigma)
                np.save(folder_path + 'actual.npy', actual)
                
                print(f'mape:{mape}, rmse:{rmse}, mae:{mae}, pinball_loss:{pinball_loss}')
                # CSV 文件名
                csv_file = folder_path + 'metrics.csv'
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # 写入表头
                    writer.writerow(['Setting', 'MAPE', 'MAE', 'RMSE', 'PINBALL_LOSS', 'CRPS', 
                                    'WINKLER_SCORE_25', 'WINKLER_SCORE_50', 'WINKLER_SCORE_75'])
                    # 写入当前数据
                    writer.writerow([setting, mape, mae, rmse, pinball_loss, crps, 
                                    winkler_score_25, winkler_score_50, winkler_score_75])
            return
        
class CentralModel_Multi(nn.Module):
    def __init__(self, args):
        super().__init__()
        set_seed(args.seed)
        self.model_elec = self._build_model(args).to(args.device)
        self.model_steam = self._build_model(args).to(args.device)
        self.model_chill = self._build_model(args).to(args.device)

    def _build_model(self, args):
        if args.central_model == "MLP":
            return MLP_central_multi(args)
        # elif args.central_model == "Diff(TimeXer)":
        #     return TimeXerDiff_central(args)
        elif args.central_model == "GRU":
            return GRU_central_multi(args)
        # elif args.central_model == "Diff(MLP)":
        #     return MLPDiff_central(args)
        elif args.central_model == "Bayes":
            return Bayes_central_multi(args)
        else:
            raise ValueError(f"Unknown model type: {args.local_model}")
    
    def _get_data(self, args, flag, load):
        data_set, data_loader = get_data(args,flag,load)
        return data_set, data_loader


    def train(self, args, setting):
        # Load data
        elec_train_data, elec_train_loader = self._get_data(args, flag='train', load='Electricity')
        steam_train_data, steam_train_loader = self._get_data(args, flag='train', load='Steam')
        chill_train_data, chill_train_loader = self._get_data(args, flag='train', load='Chillwater')
        elec_path = os.path.join(args.checkpoints, setting, 'Electricity')
        steam_path = os.path.join(args.checkpoints, setting, 'Steam')
        chill_path = os.path.join(args.checkpoints, setting, 'Chillwater')
        if not os.path.exists(elec_path):
            os.makedirs(elec_path)
        if not os.path.exists(steam_path):
            os.makedirs(steam_path)
        if not os.path.exists(chill_path):
            os.makedirs(chill_path)

        time_now = time.time()
        train_steps = len(elec_train_loader)

        optimizer = torch.optim.Adam(list(self.model_elec.parameters())
                                     +list(self.model_steam.parameters())
                                     +list(self.model_chill.parameters()), lr=args.lr)
        criterion = cauchy_likelihood()
        
        train_losses = []
        # Training loop
        for epoch in range(args.epochs_central):
            iter_count = 0
            train_loss = []

            self.model_elec.train(),self.model_steam.train(),self.model_chill.train()
            epoch_time = time.time()
            for i, (elec_data, steam_data, chill_data) in enumerate(zip(elec_train_loader, steam_train_loader, chill_train_loader)):
                """
                X_his: [batch_size, seq_len]
                X_ex: [batch_size, label_len, feature_num(6)]
                y: [batch_size, label_len]
                """
                iter_count += 1
                optimizer.zero_grad()
                elec_his=elec_data[0].to(args.device)
                elec_ex=elec_data[1].to(args.device)
                elec_label=elec_data[2].to(args.device)
                steam_his=steam_data[0].to(args.device)
                steam_ex=steam_data[1].to(args.device)
                steam_label=steam_data[2].to(args.device)
                chill_his=chill_data[0].to(args.device) #(batch_size,seq_len)
                chill_ex=chill_data[1].to(args.device) #(batch_size,pred_len,6)
                chill_label=chill_data[2].to(args.device) #(batch_size,pred_len)

                # if args.central_model == "Diff(TimeXer)" or args.central_model == "Diff(MLP)":
                #     combined_his = torch.cat([elec_his.unsqueeze(-1),steam_his.unsqueeze(-1),chill_his.unsqueeze(-1)],dim=-1) #(batch_size,hidden_dim,3)
                #     elec_loss = diffusion_loss_fn(self.model_elec, elec_label.unsqueeze(-1), alphas_bar_sqrt, one_minus_alphas_bar_sqrt, args.num_steps-1, combined_his,elec_ex)
                #     steam_loss = diffusion_loss_fn(self.model_steam, steam_label.unsqueeze(-1), alphas_bar_sqrt, one_minus_alphas_bar_sqrt, args.num_steps-1, combined_his,steam_ex)   

                combined_his = torch.cat([elec_his,steam_his,chill_his],dim=-1) #(batch_size,hidden_dim*3)
                elec_mu, elec_sigma = self.model_elec(combined_his, elec_ex)
                steam_mu, steam_sigma = self.model_steam(combined_his, steam_ex)
                chill_mu, chill_sigma = self.model_chill(combined_his, chill_ex)
                elec_loss = criterion(elec_label, elec_mu, elec_sigma)
                steam_loss = criterion(steam_label, steam_mu, steam_sigma)
                chill_loss = criterion(chill_label, chill_mu, chill_sigma)
                    
                loss = elec_loss + steam_loss + chill_loss #multi-task loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_losses.append(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))

        plot_train_loss(train_losses)
        elec_best_model_path = elec_path + '/' + 'checkpoint.pth'
        steam_best_model_path = steam_path + '/' + 'checkpoint.pth'
        chill_best_model_path = chill_path + '/' + 'checkpoint.pth'
        torch.save(self.model_elec.state_dict(), elec_best_model_path)
        torch.save(self.model_steam.state_dict(), steam_best_model_path)
        torch.save(self.model_chill.state_dict(), chill_best_model_path)

        self.model_elec.load_state_dict(torch.load(elec_best_model_path))
        self.model_steam.load_state_dict(torch.load(steam_best_model_path))
        self.model_chill.load_state_dict(torch.load(chill_best_model_path))

        return self.model_elec, self.model_steam, self.model_chill
    

    def test(self, args, setting, test=0):
        elec_test_data, elec_test_loader = self._get_data(args, flag='test', load='Electricity')
        steam_test_data, steam_test_loader = self._get_data(args, flag='test', load='Steam')
        chill_test_data, chill_test_loader = self._get_data(args, flag='test', load='Chillwater')
        if test:
            print('loading model')
            self.model_elec.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'Electricity', 'checkpoint.pth')))
            self.model_steam.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'Steam', 'checkpoint.pth')))
            self.model_chill.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'Chillwater', 'checkpoint.pth')))
        
        elec_mus = []
        elec_sigmas = []
        elec_actuals = []
        steam_mus = []
        steam_sigmas = []
        steam_actuals = []
        chill_mus = []
        chill_sigmas = []
        chill_actuals = []

        self.model_elec.eval(), self.model_steam.eval(), self.model_chill.eval()
        with torch.no_grad():
            for i, (elec_data, steam_data, chill_data) in enumerate(zip(elec_test_loader, steam_test_loader, chill_test_loader)):
                """
                X_his: [batch_size, seq_len]
                X_ex: [batch_size, label_len, feature_num(6)]
                y: [batch_size, label_len]
                """
                elec_his=elec_data[0].to(args.device)
                elec_ex=elec_data[1].to(args.device)
                elec_label=elec_data[2].to(args.device)
                steam_his=steam_data[0].to(args.device)
                steam_ex=steam_data[1].to(args.device)
                steam_label=steam_data[2].to(args.device)
                chill_his=chill_data[0].to(args.device) #(batch_size,seq_len)
                chill_ex=chill_data[1].to(args.device) #(batch_size,pred_len,6)
                chill_label=chill_data[2].to(args.device) #(batch_size,pred_len)

                # if args.central_model == "Diff(TimeXer)" or args.central_model == "Diff(MLP)":
                #     combined_his = torch.cat([elec_his.unsqueeze(-1),steam_his.unsqueeze(-1),chill_his.unsqueeze(-1)],dim=-1) #(batch_size,hidden_dim,3)
                #     elec_mu,_,elec_sigma = diffusion_sample(args,self.model_elec,combined_his,elec_label.unsqueeze(-1),elec_ex)
                #     steam_mu,_,steam_sigma = diffusion_sample(args,self.model_steam,combined_his,steam_label.unsqueeze(-1),steam_ex)
                #     chill_mu,_,chill_sigma = diffusion_sample(args,self.model_chill,combined_his,chill_label.unsqueeze(-1),chill_ex)
                #     elec_mu, elec_sigma = elec_mu.squeeze(-1), elec_sigma.squeeze(-1)
                #     steam_mu, steam_sigma = steam_mu.squeeze(-1), steam_sigma.squeeze(-1)
                #     chill_mu, chill_sigma = chill_mu.squeeze(-1), chill_sigma.squeeze(-1)

                combined_his = torch.cat([elec_his,steam_his,chill_his],dim=-1) #(batch_size,seq_len*3)
                elec_mu, elec_sigma = self.model_elec(combined_his, elec_ex)
                steam_mu, steam_sigma = self.model_steam(combined_his, steam_ex)
                chill_mu, chill_sigma = self.model_chill(combined_his, chill_ex)

                elec_mu, steam_mu, chill_mu = elec_mu.detach().cpu(), steam_mu.detach().cpu(), chill_mu.detach().cpu()
                elec_sigma, steam_sigma, chill_sigma = elec_sigma.detach().cpu(), steam_sigma.detach().cpu(), chill_sigma.detach().cpu()
                elec_label, steam_label, chill_label = elec_label.squeeze(-1).detach().cpu(), steam_label.squeeze(-1).detach().cpu(), chill_label.squeeze(-1).detach().cpu()
                
                elec_mus.append(elec_mu)
                elec_sigmas.append(elec_sigma)
                elec_actuals.append(elec_label)
                steam_mus.append(steam_mu) 
                steam_sigmas.append(steam_sigma)
                steam_actuals.append(steam_label)
                chill_mus.append(chill_mu)
                chill_sigmas.append(chill_sigma)
                chill_actuals.append(chill_label)

            elec_mus_normed = np.concatenate(elec_mus, axis=0)
            elec_sigmas_normed = np.concatenate(elec_sigmas, axis=0)
            elec_actuals_normed = np.concatenate(elec_actuals, axis=0)
            steam_mus_normed = np.concatenate(steam_mus, axis=0)
            steam_sigmas_normed = np.concatenate(steam_sigmas, axis=0)  
            steam_actuals_normed = np.concatenate(steam_actuals, axis=0)
            chill_mus_normed = np.concatenate(chill_mus, axis=0)
            chill_sigmas_normed = np.concatenate(chill_sigmas, axis=0)
            chill_actuals_normed = np.concatenate(chill_actuals, axis=0)

            elec_pred_mu = elec_test_data.inverse_transform(elec_mus_normed)
            elec_pred_sigma = elec_test_data.scale_() * elec_sigmas_normed
            elec_actual = elec_test_data.inverse_transform(elec_actuals_normed)
            steam_pred_mu = steam_test_data.inverse_transform(steam_mus_normed)
            steam_pred_sigma = steam_test_data.scale_() * steam_sigmas_normed
            steam_actual = steam_test_data.inverse_transform(steam_actuals_normed)
            chill_pred_mu = chill_test_data.inverse_transform(chill_mus_normed)
            chill_pred_sigma = chill_test_data.scale_() * chill_sigmas_normed
            chill_actual = chill_test_data.inverse_transform(chill_actuals_normed)

            elec_pred_mu = elec_pred_mu.reshape(-1)
            elec_pred_sigma = elec_pred_sigma.reshape(-1)
            elec_actual = elec_actual.reshape(-1)
            steam_pred_mu = steam_pred_mu.reshape(-1)
            steam_pred_sigma = steam_pred_sigma.reshape(-1)
            steam_actual = steam_actual.reshape(-1)
            chill_pred_mu = chill_pred_mu.reshape(-1)
            chill_pred_sigma = chill_pred_sigma.reshape(-1)
            chill_actual = chill_actual.reshape(-1)

            for load in ['Electricity', 'Steam', 'Chillwater']:
                if load == 'Electricity':
                    pred_mu = elec_pred_mu
                    pred_sigma = elec_pred_sigma
                    actual = elec_actual
                elif load == 'Steam':
                    pred_mu = steam_pred_mu
                    pred_sigma = steam_pred_sigma
                    actual = steam_actual
                elif load == 'Chillwater':
                    pred_mu = chill_pred_mu
                    pred_sigma = chill_pred_sigma
                    actual = chill_actual
                folder_path = './test_results/' + setting +'/'+ load + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                mape, rmse, mae, pinball_loss, crps, winkler_score_25, winkler_score_50, winkler_score_75 = metric(args, pred_mu, pred_sigma, actual)


                np.save(folder_path + 'mu.npy', pred_mu)
                np.save(folder_path + 'sigma.npy', pred_sigma)
                np.save(folder_path + 'actual.npy', actual)
                
                print(f'mape:{mape}, rmse:{rmse}, mae:{mae}, pinball_loss:{pinball_loss}')
                # CSV 文件名
                csv_file = folder_path + 'metrics.csv'
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # 写入表头
                    writer.writerow(['Setting', 'MAPE', 'MAE', 'RMSE', 'PINBALL_LOSS', 'CRPS', 
                                    'WINKLER_SCORE_25', 'WINKLER_SCORE_50', 'WINKLER_SCORE_75'])
                    # 写入当前数据
                    writer.writerow([setting, mape, mae, rmse, pinball_loss, crps, 
                                    winkler_score_25, winkler_score_50, winkler_score_75])
            return


def diffusion_sample(args,model,his,label,c):
    device = args.device
    model = model.to(device)
    schedule = cosine_schedule(args.num_steps)
    alphas = schedule['alphas'].to(device)
    betas = schedule['betas'].to(device)
    alphas_bar = schedule['alpha_bars'].to(device)
    alphas_bar_sqrt = torch.sqrt(alphas_bar).to(device)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_bar).to(device)
    his.to(device),label.to(device),c.to(device)
    pred = []
    # sigma = []
    for i in range(100):
        label_seq,sigma0 = p_sample_loop(model, label.shape, args.num_steps, betas, alphas, alphas_bar, his, c)
        pred.append(label_seq) 
        # sigma.append(sigma0)
    pred_tensor = torch.stack(pred)  # 将列表转换为张量(100, batch_size,3,24)
    pred_tensor = pred_tensor.permute(1, 2, 3, 0)  # 将维度重新排列为 (batch_size, 24, 3,100)
    q_25 = torch.quantile(pred_tensor, 0.25, dim=-1)
    q_75 = torch.quantile(pred_tensor, 0.75, dim=-1)
    sigma = (q_75-q_25)
    # sigma_tensor = torch.stack(sigma)
    # sigma_tensor = sigma_tensor.permute(1, 2, 3, 0)  # 将维度重新排列为 (batch_size, 24, 3,100)
    pred_mean = torch.mean(pred_tensor,dim=-1) #(batch_size,24,3)
    # sigma_mean = torch.mean(sigma_tensor,dim=-1) #(batch_size,24,3)
        
    return pred_mean,label,sigma

def find_optimal(mu,local,server,actual,w_range):
    loss_function = cauchy_likelihood()
    best_w = 0
    best_loss = float('inf')
    for w in w_range:
        sigma = local + w * server
        loss = loss_function(actual,mu,sigma)
        if loss < best_loss:
            best_loss = loss
            best_w = w
    return best_w, best_loss