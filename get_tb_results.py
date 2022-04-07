# can run from base

import os
import pickle

# from functions import * # import util functions
from collections import Counter
import numpy as np
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

import choix


def get_df(att_dict = 1):
    """att_dict should be 1,2,3"""

    path = "/home/simon/Documents/Bodies/data/RA/att_dicts"

    file_name1 = "pregenerated_indx_list.pkl"
    file_path = os.path.join(path, file_name1)

    # Open the file in binary mode
    with open(file_path, 'rb') as file:
        
        # Call load method to deserialze
        pregenerated_indx_list = pickle.load(file)

    file_name2 = f"att_dict_{att_dict}.pkl"
    file_path = os.path.join(path, file_name2)

    if os.path.exists(file_path):

        # Open the file in binary mode
        with open(file_path, 'rb') as file:
        
            # Call load method to deserialze
            att_dict = pickle.load(file)

        df_img = pd.DataFrame(pregenerated_indx_list, columns=['img1', 'img2'])
        df_att = pd.DataFrame(att_dict, columns= att_dict.keys())
        df_att.drop(['indx_indicator'], axis=1, inplace= True)
        df = df_att.join(df_img)

        columns_dict= { 'att0' : 'negative_emotions_t1', 'att1': 'negative_emotions_t2', 'att2': 'mass_protest', 
                        'att3': 'damaged_property', 'att4': 'privat', 'att5': 'public', 
                        'att6': 'militarized', 'att7': 'rural', 'att8': 'urban', 'att9': 'formal' }

        df.rename(columns= columns_dict, inplace= True)

    else: 
        print('That att_dict does not exist. Use 1, 2, or 3.')

    return(df)



def print_zero_ratio(df):

    print("Ratio of (0,0)'s in each feature:\n")

    for i in df.columns[:-2]:
        ratio = (df[i] == (0,0)).sum() / df.shape[0]
        print(f'{i}: {ratio*100:.3}%')

    print('\n')



def analyse_network(G):

    print(f'Number of edges: {len(G.edges)}') # same as len(indx_list)
    print(f'Number of nodes: {len(G.nodes)}') # same as len(indx_list)
    print(f'Connected network: {nx.is_connected(G)}')

    G_degrees = list(dict(G.degree).values())

    print(f'Mean degrees: {np.mean(G_degrees)}')
    print(f'Min degrees: {np.min(G_degrees)}')
    print(f'Max degrees: {np.max(G_degrees)}')



def get_non_draw_connected_sub_df(df, att):

    # remove draws
    non_draw_sub = df[(df[att] != (0,0)) & (df[att] != (1,1))][[att, 'img1', 'img2']]

    # get edge list from non-draw subset
    edge_list_non_zero = list(zip(non_draw_sub['img1'], non_draw_sub['img2']))

    # Full graph g
    g = nx.Graph()
    g.add_edges_from(edge_list_non_zero)

    # Get larges connected subset:
    connected_img = sorted(nx.connected_components(g), key = len, reverse=True)[0] # take the larges connected component - really the list you need.

    edge_list_connected = [(node1, node2) for node1, node2 in edge_list_non_zero if node1 in connected_img or node2 in connected_img]

    # Larges connected subgraph gc - just to check
    gc = nx.Graph()
    gc.add_edges_from(edge_list_connected)
    analyse_network(gc)
    
    # sub df
    non_draw_connected_sub_df = non_draw_sub[(non_draw_sub['img1'].isin(connected_img)) | (non_draw_sub['img2'].isin(connected_img))]

    return(non_draw_connected_sub_df) 



def get_input_data(df, att):

    non_draw_connected_sub_df = get_non_draw_connected_sub_df(df, att)

    img_list = list(set(list(non_draw_connected_sub_df['img1']) +  list(non_draw_connected_sub_df['img2'])))
    n_imgs = len(img_list)
    img_idx_generator = img_list.index

    data = []

    for i in range(non_draw_connected_sub_df.shape[0]):

        img1_name = non_draw_connected_sub_df['img1'].iloc[i]
        img2_name = non_draw_connected_sub_df['img2'].iloc[i]

        img1_idx = img_idx_generator(img1_name)
        img2_idx = img_idx_generator(img2_name)

        
        if non_draw_connected_sub_df[att].iloc[i][0] > non_draw_connected_sub_df[att].iloc[i][1]:
            directed_edge = (img1_idx, img2_idx)
            data.append(directed_edge)

        elif non_draw_connected_sub_df[att].iloc[i][0] < non_draw_connected_sub_df[att].iloc[i][1]:
            directed_edge = (img2_idx, img1_idx)
            data.append(directed_edge)

        else: 
            print(f'something wrong w/ edge')
            pass

    return(data, n_imgs, img_list)


def get_att_data_dict(df):

    att_data_dict = {}

    for att in df.columns[:-2]:

        print(att)
        
        data, n_imgs, img_list = get_input_data(df, att)

        att_data_dict[f'{att}_data'] = data
        att_data_dict[f'{att}_n'] = n_imgs
        att_data_dict[f'{att}_img_list'] = img_list


    return(att_data_dict)


def get_results(df):

    att_data_dict = get_att_data_dict(df)    
    dict_of_dfs = {}

    for att in df.columns[:-2]:

        data = att_data_dict[f'{att}_data']
        n_imgs = att_data_dict[f'{att}_n']

        lsr_mean = choix.ilsr_pairwise(n_imgs, data, alpha=0.01) # 10-20 sec
        print(f'{att} lsr done')
        mm_mean = choix.mm_pairwise(n_imgs, data, alpha=0.05) # 2 ish min, needs a bit more reg/aalpha to converge
        print(f'{att} mm done')
        #eb_mean, eb_cov = choix.ep_pairwise(n_imgs, data, alpha=0.05, model = 'logit') # 20 ish min
        
        result_dict = {'lsr_mean' : lsr_mean, 'mm_mean' : mm_mean , 'img' : att_data_dict[f'{att}_img_list']}
        #result_dict = {'lsr_mean' : lsr_mean, 'mm_mean' : mm_mean , 'eb_mean' : eb_mean , 'img' : att_data_dict[f'{att}_img_list']}

        results_df = pd.DataFrame(result_dict)

        dict_of_dfs[att] = results_df

    return(dict_of_dfs)


def run_and_dump():

    for i in [1,3]:#[1,2,3]
        
        df = get_df(i)
        print_zero_ratio(df)
        dict_of_dfs = get_results(df)


        file_name = f'dict_of_dfs{i}.pkl'

        with open(file_name, 'wb') as file:
            pickle.dump(dict_of_dfs, file)

if __name__ == "__main__":
   run_and_dump()