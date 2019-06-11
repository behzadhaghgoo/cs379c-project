import pandas as pd
import os

# https://stackoverflow.com/questions/50786266/writing-dictionary-of-dataframes-to-file
def save(meta_dict, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for k in meta_dict:
        assert type(k) == tuple
    
    for k, v in meta_dict.items():
        v.to_csv(save_dir + '/data_{}.csv'.format(str(k)))
        
    with open(save_dir + '/keys.txt', 'w') as f:
        f.write(str(list(meta_dict.keys())))
        
def load(save_dir):
    with open(save_dir + 'keys.txt', 'r') as f:
        keys = eval(f.read())
        
    meta_dict = {}
    for k in keys:
        meta_dict[k] = pd.read_csv(save_dir + '/data_{}.csv'.format(str(k)))
        
    return meta_dict
                