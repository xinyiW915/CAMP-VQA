import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
import logging

#NR: original
def process_lsvq(train_data_name, test_data_name, metadata_path, feature_path, network_name):
    train_df = pd.read_csv(f'{metadata_path}/{train_data_name.upper()}_metadata.csv')
    test_df = pd.read_csv(f'{metadata_path}/{test_data_name.upper()}_metadata.csv')

    # grayscale videos, do not consider them for fair comparison
    # grey_df_train = pd.read_csv(f'{metadata_path}/greyscale_report/{train_data_name.upper()}_greyscale_metadata.csv')
    # grey_df_test = pd.read_csv(f'{metadata_path}/greyscale_report/{test_data_name.upper()}_greyscale_metadata.csv')
    # grey_indices_train = grey_df_train.iloc[:, 0].tolist()
    # grey_indices_test = grey_df_test.iloc[:, 0].tolist()
    # train_df = train_df.drop(index=grey_indices_train).reset_index(drop=True)
    # test_df = test_df.drop(index=grey_indices_test).reset_index(drop=True)
    test_vids = test_df['vid']

    # mos scores
    train_scores = train_df['mos'].tolist()
    test_scores = test_df['mos'].tolist()
    train_mos_list = train_scores
    test_mos_list = test_scores

    # reorder columns
    sorted_train_df = pd.DataFrame({'vid': train_df['vid'],  'framerate': train_df['framerate'], 'MOS': train_mos_list, 'MOS_raw': train_df['mos']})
    sorted_test_df = pd.DataFrame({'vid': test_df['vid'], 'framerate': test_df['framerate'], 'MOS': test_mos_list, 'MOS_raw': test_df['mos']})

    # use indices from the train and test DataFrames to split features
    train_features = torch.load(f'{feature_path}/{network_name}_{train_data_name}_features.pt')
    print(f"loaded {train_data_name}: dimensions are {train_features.shape}")
    test_features = torch.load(f'{feature_path}/{network_name}_{test_data_name}_features.pt')

    # grayscale videos
    # train_mask = torch.ones(train_features.size(0), dtype=torch.bool, device=train_features.device)
    # test_mask = torch.ones(test_features.size(0), dtype=torch.bool, device=test_features.device)
    # train_mask[grey_indices_train] = False
    # test_mask[grey_indices_test] = False
    # train_features = train_features[train_mask]
    # test_features = test_features[test_mask]
    print(len(train_features))
    print(len(test_features))

    # save the files
    sorted_train_df.to_csv(f'{metadata_path}mos_files/{train_data_name}_MOS_train.csv', index=False)
    sorted_test_df.to_csv(f'{metadata_path}mos_files/{train_data_name}_MOS_test.csv', index=False)
    os.makedirs(os.path.join(feature_path, "split_train_test"), exist_ok=True)
    torch.save(train_features, f'{feature_path}/split_train_test/{network_name}_{train_data_name}_train_features.pt')
    torch.save(test_features, f'{feature_path}/split_train_test/{network_name}_{test_data_name}_test_features.pt')

    return train_features, test_features, test_vids

def process_other(data_name, test_size, random_state, metadata_path, feature_path, network_name):
    metadata_name = f'{data_name.upper()}_metadata.csv'
    # for test finevd mos dimensions
    # if data_name == 'finevd':
    #     metadata_name = f'{data_name.upper()}_MOS_temporal_metadata.csv'
    #     print(metadata_name)

    # load CSV data
    df = pd.read_csv(f'{metadata_path}/{metadata_name}')
    # if data_name == 'youtube_ugc':
    #     # grayscale videos, do not consider them for fair comparison
    #     grey_df = pd.read_csv(f'{metadata_path}/greyscale_report/{data_name.upper()}_greyscale_metadata.csv')
    #     grey_indices = grey_df.iloc[:, 0].tolist()
    #     df = df.drop(index=grey_indices).reset_index(drop=True)

    # get unique vids
    unique_vids = df['vid'].unique()

    # split videonames into train and test sets
    train_vids, test_vids = train_test_split(unique_vids, test_size=test_size, random_state=random_state)

    # split all_dfs into train and test based on vids
    train_df = df[df['vid'].isin(train_vids)]
    test_df = df[df['vid'].isin(test_vids)]

    # mos scores
    train_scores = train_df['mos'].tolist()
    test_scores = test_df['mos'].tolist()
    train_mos_list = train_scores
    test_mos_list = test_scores

    # reorder columns
    sorted_train_df = pd.DataFrame({'vid': train_df['vid'],  'framerate': train_df['framerate'], 'MOS': train_mos_list, 'MOS_raw': train_df['mos']})
    sorted_test_df = pd.DataFrame({'vid': test_df['vid'], 'framerate': test_df['framerate'], 'MOS': test_mos_list, 'MOS_raw': test_df['mos']})

    # use indices from the train and test DataFrames to split features
    features = torch.load(f'{feature_path}/{network_name}_{data_name}_features.pt')
    # if data_name == 'youtube_ugc':
    #     # features = np.delete(features, grey_indices, axis=0)
    #     mask = torch.ones(features.size(0), dtype=torch.bool, device=features.device)
    #     mask[grey_indices] = False
    #     features = features[mask]
    train_features = features[train_df.index]
    test_features = features[test_df.index]

    # save the files
    sorted_train_df.to_csv(f'{metadata_path}mos_files/{data_name}_MOS_train.csv', index=False)
    sorted_test_df.to_csv(f'{metadata_path}mos_files/{data_name}_MOS_test.csv', index=False)
    os.makedirs(os.path.join(feature_path, "split_train_test"), exist_ok=True)
    torch.save(train_features, f'{feature_path}/split_train_test/{network_name}_{data_name}_train_features.pt')
    torch.save(test_features, f'{feature_path}/split_train_test/{network_name}_{data_name}_test_features.pt')

    return train_features, test_features, test_vids


if __name__ == '__main__':
    network_name = "slowfast"
    data_name = "test"
    metadata_path = '../../metadata/'
    feature_path = '../../features/konvid_1k_test/slowfast/'

    # train test split
    test_size = 0.2
    random_state = None

    if data_name == 'lsvq_train':
        test_data_name = 'lsvq_test'
        process_lsvq(data_name, test_data_name, metadata_path, feature_path, network_name)

    else:
        process_other(data_name, test_size, random_state, metadata_path, feature_path, network_name)
