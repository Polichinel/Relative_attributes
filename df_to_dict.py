import numpy as np
import pandas as pd
import pickle

def change_and_dump():

    dir_path = '/home/simon/Documents/Bodies/data/RA/dfs/'

    with open(f'{dir_path}ra_ens_annotated_df.pkl', 'rb') as file:
        df = pickle.load(file)

        df_no_nans = df.iloc[:,:-1].fillna(df.iloc[:,:-1].mean())
        df_no_nans['img'] = df['img']

    # make dict
    attribute_dict = {}

    for i in df_no_nans.columns:
        attribute_dict[i]= list(df_no_nans[i])

    # check
    new_df = pd.DataFrame(attribute_dict)
    if df_no_nans.equals(new_df) == True:
        print('it works')

    with open(f'{dir_path}ra_ens_annotated_dict.pkl', 'wb') as file: # should be in some data folder
        pickle.dump(attribute_dict, file)


# alot gets esier if you just go nan -> attribute mean

if __name__ == "__main__":
    change_and_dump()