import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from util.tools import StandardScaler

import warnings
warnings.filterwarnings('ignore')

class Dataset_ST(Dataset):
    def __init__(self, root_path='./data/', data_path='metr-la.h5', flag='train', seq_len=12, pred_len=12):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.flag = flag
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        
        df_raw = pd.read_hdf(os.path.join(self.root_path,
                                          self.data_path))
        df_rec = df_raw.copy()
        df_raw = df_raw.replace(0, np.nan)
        df_raw = df_raw.bfill()
        df_raw = df_raw.ffill()

        num_samples = len(df_raw)#-self.seq_len-self.pred_len+1
        num_train = round(num_samples*0.7)
        num_test = round(num_samples*0.2)
        num_val = num_samples-num_train-num_test
        
        train_data = df_raw.values[:num_train]
        self.scaler = StandardScaler(train_data.mean(), train_data.std())
        # train_data = self.scaler.fit_transform(train_data)
        
        borders = {'train':[0,num_train],'val':[num_train,num_train+num_val],'test':[num_samples-num_test,num_samples]}
        border1, border2 = borders[self.flag][0], borders[self.flag][1]
        data_raw = df_raw.values[border1:border2]
        data_rec = df_rec.values[border1:border2]
        data = self.scaler.transform(data_raw)
        
        Time = df_raw.index
        dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
        timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                    // Time.freq.delta.total_seconds()
        timeofday = np.reshape(timeofday, newshape = (-1, 1))    
        Time = np.concatenate((dayofweek, timeofday), axis = -1)
        
        self.data_x = data
        self.data_y = data_rec
        self.data_stamp = Time[border1:border2]
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        nodes = seq_x.shape[-1]
        seq_x = np.expand_dims(seq_x, -1); seq_y = np.expand_dims(seq_y, -1)
        seq_x_mark = np.tile(np.expand_dims(seq_x_mark, -2), [1,nodes,1])
        seq_y_mark = np.tile(np.expand_dims(seq_y_mark, -2), [1,nodes,1])

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

