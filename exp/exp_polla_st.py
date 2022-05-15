from data.dataloader import Dataset_ST
from exp.exp_basic import Exp_Basic
from models.modelst import POLLA_gcn, POLLA_diff, POLLA_adpadj, POLLA_adpadj_skip

from util.tools import EarlyStopping, adjust_learning_rate, load_adj
from util.metrics import masked_mae, masked_rmse, masked_mape, metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

class Exp_POLLA(Exp_Basic):
    def __init__(self, args):
        super(Exp_POLLA, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'pollagcnst':POLLA_gcn,
            'polladiffst':POLLA_diff,
            'pollaadpst':POLLA_adpadj,
            'pollaadpskipst':POLLA_adpadj_skip,
        }
        supports, adjinit = self._get_adj()
        if self.args.model=='polladiffst' or self.args.model=='pollaadpst' \
            or self.args.model=='pollaadpskipst':
            self.support = supports
            model = model_dict[self.args.model](
                self.args.c_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.pred_len, 
                self.args.d_model, 
                self.args.n_heads, 
                self.args.n_layers, 
                self.args.d_ff,
                self.args.nodes,
                adjinit,
                supports, 
                self.args.order, 
                self.args.dropout, 
                self.args.activation,
                self.device
            )
        elif self.args.model=='pollagcnst':
            self.support = supports[0]
            model = model_dict[self.args.model](
                self.args.c_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.pred_len, 
                self.args.d_model, 
                self.args.n_heads, 
                self.args.n_layers, 
                self.args.d_ff,
                self.args.nodes,
                self.args.dropout, 
                self.args.activation,
                self.device
            )
        return model.double()

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False
        else:
            shuffle_flag = True

        data_set = Dataset_ST(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            seq_len=args.seq_len,
            pred_len=args.pred_len
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=True)

        return data_set, data_loader

    

    def _get_adj(self):
        sensor_ids, sensor_id_to_ind, adj_mx = load_adj(self.args.adjdata, self.args.adjtype)

        supports = [torch.DoubleTensor(i).double().to(self.device) for i in adj_mx]
        adjinit = None if self.args.randomadj else supports[0]
        if adjinit is not None:
            print('adjinit shape:', adjinit.shape)

        return supports, adjinit

    def _get_se(self):
        SE_file = self.args.sedata
        f = open(SE_file, mode = 'r')
        lines = f.readlines()
        temp = lines[0].split(' ')
        N, dims = int(temp[0]), int(temp[1])
        SE = np.zeros(shape = (N, dims), dtype = np.float32)
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = temp[1:]

        return torch.from_numpy(SE).double().to(self.device)


    def vali(self, vali_data, vali_loader, criterion):
        se_data = self._get_se()

        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double().to(self.device)
            batch_x_mark = batch_x_mark.double().to(self.device)

            outputs = self.model(batch_x, batch_x_mark, se=se_data, support=self.support)

            pred = vali_data.scaler.inverse_transform(outputs.detach().cpu().numpy().squeeze())
            true = batch_y.detach().cpu().numpy().squeeze()
            pred = torch.from_numpy(pred); true = torch.from_numpy(true)
            loss = masked_mae(pred, true, 0.0).item()
            
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        return total_loss
        
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')

        se_data = self._get_se()

        path = './checkpoints/'+setting
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        
        if self.args.loss=='huber':
            criterion =  nn.SmoothL1Loss().to(self.device)
        else:
            criterion =  masked_mae

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double().to(self.device)
                batch_x_mark = batch_x_mark.double().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, se=se_data, support=self.support) 
                outputs = train_data.scaler.inverse_transform(outputs)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
            
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch, self.args)
        
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        se_data = self._get_se()
        
        self.model.eval()

        preds = []
        trues = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double().to(self.device)
            batch_x_mark = batch_x_mark.double().to(self.device)
            
            outputs = self.model(batch_x, batch_x_mark, se=se_data, support=self.support)

            pred = test_data.scaler.inverse_transform(outputs.detach().cpu().numpy().squeeze())
            true = batch_y.detach().cpu().numpy().squeeze()
            
            preds.append(pred.squeeze())
            trues.append(true.squeeze())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds = torch.from_numpy(preds.reshape(-1, preds.shape[-2], preds.shape[-1]))
        trues = torch.from_numpy(trues.reshape(-1, trues.shape[-2], trues.shape[-1]))

        mae,mape,rmse = metric(preds, trues)
        print('mae:{}, mape:{}, rmse:{}'.format(mae, mape, rmse))

        np.save(folder_path+'metrics.npy', np.array([mae,rmse,mape]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return