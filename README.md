# POLLA: Enhancing the Local Structure Awareness in Long Sequence Spatial-temporal Modeling

This is the pytorch implementation of POLLA in the ACM TIST'21 paper: [POLLA: Enhancing the Local Structure Awareness in Long Sequence Spatial-temporal Modeling](https://dl.acm.org/doi/10.1145/3447987).


## Requirements
+ Python 3.6
+ matplotlib == 3.1.1
+ numpy == 1.19.4
+ pandas == 0.25.1
+ scikit_learn == 0.21.3
+ torch == 1.8.0
+ ...

Dependencies can be installed using the following command:

```
pip install -r requirements.txt
```


## Train Commands
Commands for training model on METR-LA:

```
python -u main_polla_exp.py --model polladiff --data metr --seq_len 12 --pred_len 12 --d_model 64 --n_layers 3 --n_heads 8 --d_ff 256 --train_epochs 4 --patience 10 --itr 2 --loss mae
```

## <span id="citelink">Citation</span>
If you find this repository useful in your research, please consider citing the following paper: [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fzhouhaoyi%2FPOLLA&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

```
@article{haoyietal-polla-2021,
  author    = {Haoyi Zhou and
               Hao Peng and
               Jieqi Peng and
               Shuai Zhang and
               Jianxin Li},
  title     = {{POLLA:} Enhancing the Local Structure Awareness in Long Sequence
               Spatial-temporal Modeling},
  journal   = {{ACM} Transactions on Intelligent Systems and Technology},
  volume    = {12},
  number    = {6},
  pages     = {69:1--69:24},
  year      = {2021},
  doi       = {10.1145/3447987},
}
```

## Contact

If you have any questions, feel free to contact Haoyi Zhou through email ([zhouhaoyi1991@gmail.com](mailto:zhouhaoyi1991@gmail.com)) or Github issues. Pull requests are highly welcomed!
