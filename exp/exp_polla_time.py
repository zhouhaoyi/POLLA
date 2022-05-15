from data.dataloader import Dataset_ST
from exp.exp_basic import Exp_Basic
from models.model import POLLA_gcn, POLLA_diff, POLLA_adpadj

from util.tools import EarlyStopping, adjust_learning_rate, load_adj
from util.metrics import masked_mae, masked_rmse, masked_mape, masked_huber, masked_mse, metric

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
            'pollagcn':POLLA_gcn,
            'polladiff':POLLA_diff,
            'pollaadp':POLLA_adpadj,
        }
        supports, adjinit = self._get_adj()
        if self.args.model=='polladiff' or self.args.model=='pollaadp':
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
        elif self.args.model=='pollagcn':
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
        print('adjinit shape:', adjinit.shape)

        return supports, adjinit

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double().to(self.device)
            batch_x_mark = batch_x_mark.double().to(self.device)

            outputs = self.model(batch_x, batch_x_mark, support=self.support)

            pred = vali_data.scaler.inverse_transform(outputs.detach().cpu().numpy().squeeze())
            true = batch_y.detach().cpu().numpy().squeeze()
            pred = torch.from_numpy(pred); true = torch.from_numpy(true)
            loss = masked_mae(pred, true, 0.0).item()
            
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        return total_loss
        
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'test')


        path = './checkpoints/'+setting
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        
        loss_map = {'huber':masked_huber, 'mae':masked_mae, 'mse':masked_mse, 'mape':masked_mape}

        criterion = loss_map[self.args.loss]

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            t1 = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double().to(self.device)
                batch_x_mark = batch_x_mark.double().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, support=self.support) 
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
            
            t2 = time.time()
            print("Epoch {} time spent: {:.4f}".format(epoch+1, t2-t1))
            
            train_loss = np.average(train_loss)
            
            t1 = time.time()
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            t2 = time.time()
            print("Validation Inference time spent: {:.4f}".format(t2-t1))
            
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

    def test(self, setting): # args.mode=='direct'
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()

        preds = []
        trues = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double().to(self.device)
            batch_x_mark = batch_x_mark.double().to(self.device)
            
            outputs = self.model(batch_x, batch_x_mark, support=self.support)

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

        mae_total=0; mape_total=0; rmse_total=0
        steps = preds.shape[1]
        for step in range(steps):
            pp = preds[:,step,:]
            tt = trues[:,step,:]
            mae,mape,rmse = metric(pp, tt)
            print('step:{}, mae:{}, mape:{}, rmse:{}'.format(step, mae, mape, rmse))
            mae_total += mae; mape_total += mape; rmse_total += rmse
        mae_total /= steps; mape_total /= steps; rmse_total /= steps
        print('total mae:{}, mape:{}, rmse:{}'.format(mae_total, mape_total, rmse_total))

        mae,mape,rmse = metric(preds, trues)
        print('mae:{}, mape:{}, rmse:{}'.format(mae, mape, rmse))

        np.save(folder_path+'metrics.npy', np.array([mae,rmse,mape]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return