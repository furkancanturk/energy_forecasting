import argparse
import os
import torch
from deep_timeseries_training import TS_Experiment
from utils.print_args import print_args
from pprint import pprint
import random
import numpy as np
from sklearn.model_selection import ParameterGrid
import wandb
root_path = os.getcwd()
use_gpu = torch.cuda.is_available()

print(torch.__version__)
from argparse import Namespace

if __name__ == '__main__':

    WANDB_LOG = True

    n_iters = 1
    dt_name = 'wind_plant'
    target = 'demand' if dt_name == 'electricity_demand' else 'prod'

    data_params = dict(  
        data=[dt_name], 
        target=[target], 
        freq=['h'] #[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], 
        )

    hyperparams = dict(
         train_epochs=[40], 
         batch_size=[64], 
         patience=[5], 
         lradj=['cosine'], 
         learning_rate=[0.0005], 
         model=['iTransformer'], 
         loss=['MSE'], 
         features=['MS'],   #'forecasting task options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
         seq_len=[24],   #'input sequence length'
         label_len=[1],   #'start token length'
         pred_len=[1],    #'prediction sequence length'
         inverse=[True], #'inverse output data for testing'
         top_k=[5],     #'for TimesBlock'
         num_kernels=[6], #'for Inception'
         enc_in=[16],   #'encoder input size'
         dec_in=[16],   #'decoder input size'
         c_out=[16],    #'output size'
         d_model=[64], #'dimension of model'
         n_heads=[4],  #'num of heads'
         e_layers=[2],  #'num of encoder layers'
         d_layers=[1],  #'num of decoder layers'
         d_ff=[128],  #'dimension of fcn'
         dropout=[0.1], 
         #p_hidden_dims=[32, 64], 
         #p_hidden_layers=[1, 2],
        # seed=[1], 
        #seasonal_patterns='Monthly', 
        embed=['timeF'], #'time features encoding, options:[timeF, fixed, learned]
        activation=['gelu'], 
        #  moving_avg=24, 
        factor=[1], 
        #  distil=True, 
        output_attention=[False], 
        tag = ['no_ds_embed'],
         )
    
    device_params = dict(
         num_workers=[1], 
         use_amp = [False], 
         use_gpu = [use_gpu], 
         gpu = [0],
        use_multi_gpu=[False], 
        #  devices='0,1,2,3'
    )
        #    des='test', 
        #     task_name='long_term_forecast', 
        #  is_training=1, 
        #  model_id='test', 

    path_params = dict(
        root_path=[root_path], 
        checkpoints=['./checkpoints/'], 
        data_path=['train.csv'])

    param_grid = ParameterGrid({**hyperparams, **data_params, **device_params, **path_params})
    print("Parameter grid size:",len(param_grid))

    if WANDB_LOG:
        wandb.login(force=True)

    for ID, config in enumerate(param_grid):
        for seed in range(n_iters):      

            config["random_state"] = seed
            args = Namespace(**config)
            
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            print(">>>>",seed, ID, (ID+1)/len(param_grid))
            pprint(config)
            
            if not (args.enc_in == args.dec_in == args.c_out):
                continue

            exp = TS_Experiment(args) 
        
            setting = '{}/{}_ft{}_sl{}_ll{}_pl{}_co{}_dm{}_nh{}_el{}_dl{}_df{}_do{}_lr{}_la{}'.format(
                args.data,
                #args.model_id,
                args.model,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.c_out,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
            # args.p_hidden_dims,
            # args.p_hidden_layers,
                #args.factor,
                args.dropout,
                #args.embed,
                #args.distil,
                args.learning_rate,
                args.lradj,
                #args.des
                )
            
            if args.tag:
                setting += "_"+args.tag

            model_name = setting.split('/')[1]
            print('>>>>start training : {}'.format(setting))
            
            if WANDB_LOG:
                run = wandb.init(project=f'{dt_name}_forecasting', config=config, force=True, name=model_name, tags=[args.tag]) #, 
            
            exp.train(setting, WANDB_LOG)

            if args.tag != 'alldata':
                print('>>>>testing : {}'.format(setting))
                exp.test(setting, test=0)
            
            torch.cuda.empty_cache()
            
            if WANDB_LOG:
                run.finish()