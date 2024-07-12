import numpy as np
import pandas as pd
import gc
import shutil
try:
    shutil.rmtree('./saved')
    shutil.rmtree('./log')
    shutil.rmtree('./log_tensorboard')
except: pass

base_config = './configs/base_configs.yaml'
for dataset_ in ['ml-1m', 'amazon-beauty', 'gowalla-merged']:
    # df = pd.read_csv(f'./dataset/{dataset_}/{dataset_}.inter', sep='\t')
    # tss = np.percentile(df['timestamp:float'], np.arange(10,61,10))
    # del df
    for model_ in ['SASRec', 'BERT4Rec', 'STAMP', 'SINE', 'CORE', 'FEARec']:
        # for i, ts in enumerate(tss):
            i=0
            print('******************************', i, model_, dataset_)
            parameter_dict = {'epochs': 1} #, 'val_interval': {'timestamp':f'[{ts}, inf)'}}
            from recbole.quick_start import run_recbole 
            run_recbole(model = model_, dataset = dataset_, 
                config_file_list = [base_config], config_dict = parameter_dict)
            gc.collect()
            shutil.rmtree('./saved')
