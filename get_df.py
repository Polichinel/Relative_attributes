# if local run in new_torch_env but should run on most

import os

import numpy as np
import pandas as pd

import pickle


def make_df(): 
    file_name1 = 'dict_of_dfs1.pkl'

    with open(file_name1, 'rb') as file:      
        dict_of_dfs1 = pickle.load(file)

    file_name2 = 'dict_of_dfs2.pkl'

    with open(file_name2, 'rb') as file:      
        dict_of_dfs2 = pickle.load(file)

    file_name3 = 'dict_of_dfs3.pkl'

    with open(file_name3, 'rb') as file:      
        dict_of_dfs3 = pickle.load(file)


    new_df = pd.DataFrame({'img':[]})

    dict_of_dicts = {'annot1' : dict_of_dfs1, 'annot2' : dict_of_dfs2, 'annot3' : dict_of_dfs3}

    for d in dict_of_dicts.keys():
        annot_dict = dict_of_dicts[d]
        
        for k in annot_dict.keys():
            temp_df = annot_dict[k].copy()

            # standardize
            
            att_mean = temp_df.iloc[: , :2].mean()
            att_std = temp_df.iloc[: , :2].std()
            s_temp_df = (temp_df.iloc[: , :2] - att_mean)/att_std
            s_temp_df['img'] = temp_df.loc[:,'img']
            
            for c in s_temp_df.columns:
                if 'mean' in c:
                    s_temp_df.rename(columns = {c : f'{d}_{k}_{c}'}, inplace= True)

            new_df = new_df.merge(s_temp_df, how='outer', on = 'img')
            # break

    list_of_att = list(set(['_'.join(c.split('_')[1:-2]) for c in new_df.columns if c !=  'img']))

    for i, att in enumerate(list_of_att):
        new_df[f'all_{att}_ens_mean'] = new_df.loc[:,new_df.columns.str.contains(list_of_att[i])].mean(axis = 1)
        new_df[f'all_{att}_ens_std'] = new_df.loc[:,new_df.columns.str.contains(list_of_att[i])].std(axis = 1)

    return new_df


def compile_and_dump():

    df = make_df()

    dir_path = '/home/simon/Documents/Bodies/data/RA/dfs/'

    with open(f'{dir_path}a_full_annotated_df.pkl', 'wb') as file: # should be in some data folder
        pickle.dump(df, file)

    # more consice ensample df
    ens_df = df.loc[:,df.columns.str.contains('all')].copy()
    ens_df['img'] = df['img']

    with open(f'{dir_path}ra_ens_annotated_df.pkl', 'wb') as file: # should be in some data folder
        pickle.dump(ens_df, file)


if __name__ == "__main__":
   compile_and_dump()