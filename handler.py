import os

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES_TO_DROP = ['mRNA_start', 'label', 'mRNA_name', 'target sequence', 'microRNA_name', 'miRNA sequence',
                    'full_mrna',
                    'canonic_seed', 'duplex_RNAplex_equals', 'non_canonic_seed', 'site_start', 'num_of_pairs',
                    'mRNA_end', 'constraint']

PROCESSED_PATH = r"C:\Users\eyalhad\Desktop\transfer_mirna\data"


def datasets_list_data_extraction(dataset_list, what_data):
    data = pd.DataFrame()
    for d in dataset_list:
        df = pd.read_csv(os.path.join(PROCESSED_PATH, f"{d}_{what_data}.csv"), index_col=False)
        data = data.append(df)
    return data



def get_data(org_name,data_type):
    data = pd.read_csv(os.path.join("data", f"{org_name}_{data_type}.csv"), index_col=False)
    X = data.drop(FEATURES_TO_DROP, axis=1)
    y = data['label']
    X = X.astype("float")
    print(f"Dataset shape of {org_name} is:{X.shape}")
    return X,y

