import argparse
import os

from exp.exp_polla_exp import Exp_POLLA

parser = argparse.ArgumentParser(description='[POLLA] Spatial Temporal Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='polladiff',help='model of the experiment')

parser.add_argument('--data', type=str, required=True, default='metr',help='data')
parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='metr-la.h5', help='location of the data file')    
parser.add_argument('--adjdata', type=str, default='./data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition',help='adj type')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')

parser.add_argument('--seq_len', type=int, default=12, help='input series length')
parser.add_argument('--pred_len', type=int, default=12, help='predict series length')
parser.add_argument('--c_in', type=int, default=1, help='input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--n_layers', type=int, default=3, help='num of layers')
parser.add_argument('--d_ff', type=int, default=8, help='dimension of fcn')

parser.add_argument('--nodes', type=int, default=207, help='num of nodes')
parser.add_argument('--order', type=int, default=2, help='gcn order')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--Ks', type=int, default=3, help='gcn cheby Ks')
parser.add_argument('--embedtype', type=str, default='comp',help='time embedding type')

parser.add_argument('--itr', type=int, default=1, help='each params run iteration')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='input data batch size')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mae',help='loss function')


parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()

data_parser = {
    'metr':['metr-la.h5','./data/sensor_graph/adj_mx.pkl','./data/sensor_graph/SE(METR).txt', 207],
    'pems':['pems-bay.h5','./data/sensor_graph/adj_mx_bay.pkl','./data/sensor_graph/SE(PeMS).txt', 325],
}

args.data_path, args.adjdata, args.sedata, args.nodes = data_parser[args.data]

Exp = Exp_POLLA

for ii in range(args.itr):
    # setting = 'test'
    setting = '{}_{}_sl{}_pl{}_dm{}_nh{}_nl{}_df{}_ls{}_{}_{}'.format(args.model, args.data, args.seq_len, args.pred_len,
                args.d_model, args.n_heads, args.n_layers, args.d_ff, args.loss, args.des, ii)

    exp = Exp(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)