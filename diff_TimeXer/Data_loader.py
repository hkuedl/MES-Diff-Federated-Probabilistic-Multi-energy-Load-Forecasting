from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import datetime
import torch

def get_data(args,flag,load='Electricity'):
    # 读取数据
    data_path = args.data_path
    data = pd.read_csv(data_path)
    cut_length1 = int(args.train_len*len(data))
    cut_length2 = int((args.train_len+args.val_len)*len(data))
    data_val = data[cut_length1:cut_length2]
    data_train = data[:cut_length1]
    data_test = data[cut_length2:]
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

def feature(df,name):
    X_train_load_features = [name+'_load_t-' + str(i) for i in range(96)]
    y_train_load_features = [name+'_load_t' + str(i) for i in range(24)]
    temperature_features = ['AirTemperature_t' + str(i) for i in range(24)]
    humidity_features = ['DewTemperature_t' + str(i) for i in range(24)]
    cal_features = ['Month_t' + str(i) for i in range(24)]+['Day_t' + str(i) for i in range(24)]+['Hour_t' + str(i) for i in range(24)]+['Week_t' + str(i) for i in range(24)]
    X = df[X_train_load_features+temperature_features+humidity_features+cal_features]
    y = df[y_train_load_features]
    return X,y

def all_feature(data_train,data_val,data_test,name):
    X_train,y_train = feature(data_train,name)
    X_val,y_val = feature(data_val,name)
    X_test,y_test = feature(data_test,name)
    return X_train,y_train,X_val,y_val,X_test,y_test

#数据集处理顺序：平移 -> 数据集划分 -> 标准化 -> 分段 scale参数通常用于指示是否对数据进行标准化或归一化处理
class CreateDataset(Dataset):
    def __init__(self, data_train, data_val, data_test, load = 'Electricity', flag='train', seq_len=4*24, label_len=24):
        self.seq_len = seq_len  # 输入序列长度
        self.label_len = label_len  # 标签序列长度
        self.data_train = data_train  # 训练数据
        self.data_val = data_val  # 验证数据
        self.data_test = data_test  # 测试数据
        self.load = load #冷热电哪种数据
        self.flag = flag  # 数据集模式（train/val/test）
        # 读取数据
        self.X_his, self.X_ex, self.y = self.__read_data__(self.data_train, self.data_val, self.data_test, self.load)

    def __read_data__(self, data_train, data_val, data_test, load):
        X_train,y_train, X_val,y_val,X_test,y_test = all_feature(data_train,data_val,data_test,load)
        self.scaler_x = StandardScaler() #MinMaxScaler()#
        self.scaler_y = StandardScaler() #MinMaxScaler()#

        scaled_train_X = self.scaler_x.fit_transform(np.array(X_train))
        scaled_val_X = self.scaler_x.transform(np.array(X_val))
        scaled_test_X = self.scaler_x.transform(np.array(X_test))
        scaled_train_y = self.scaler_y.fit_transform(np.array(y_train))
        scaled_val_y = self.scaler_y.transform(np.array(y_val))
        scaled_test_y = self.scaler_y.transform(np.array(y_test))

        #把X中的历史负荷和预测窗口对应的日历和天气特征分出来
        scaled_train_X_his = np.array(scaled_train_X)[:,:96].astype(np.float32) #历史负荷
        scaled_train_X_ex = np.array(scaled_train_X)[:,96:].reshape(-1, 6, 24).transpose(0, 2, 1).astype(np.float32) #预测窗口对应的日历和天气特征

        scaled_val_X_his = np.array(scaled_val_X)[:,:96].astype(np.float32)
        scaled_val_X_ex = np.array(scaled_val_X)[:,96:].reshape(-1, 6, 24).transpose(0, 2, 1).astype(np.float32)

        scaled_test_X_his = np.array(scaled_test_X)[:,:96].astype(np.float32)
        scaled_test_X_ex = np.array(scaled_test_X)[:,96:].reshape(-1, 6, 24).transpose(0, 2, 1).astype(np.float32)

        scaled_train_y,scaled_val_y,scaled_test_y = np.array(scaled_train_y).astype(np.float32),np.array(scaled_val_y).astype(np.float32),np.array(scaled_test_y).astype(np.float32)

        # 根据模式（train/val/test）选择数据
        if self.flag == 'train':
            self.X_his = scaled_train_X_his
            self.X_ex = scaled_train_X_ex
            self.y = scaled_train_y
            

        elif self.flag == 'val':
            self.X_his = scaled_val_X_his
            self.X_ex = scaled_val_X_ex
            self.y = scaled_val_y

        elif self.flag == 'test':
            self.X_his = scaled_test_X_his
            self.X_ex = scaled_test_X_ex
            self.y = scaled_test_y

        return torch.tensor(self.X_his), torch.tensor(self.X_ex), torch.tensor(self.y)

    def __getitem__(self, index):
        # 返回指定索引的数据
        seq_x_his = self.X_his[index]
        seq_x_ex = self.X_ex[index]
        seq_y = self.y[index]
        return seq_x_his, seq_x_ex, seq_y
    
    def __len__(self):
        # 返回数据集长度
        return len(self.X_his)

    def inverse_transform(self, y_nor):
        return self.scaler_y.inverse_transform(y_nor)
    
    def scale_(self):
        return self.scaler_y.scale_
        



