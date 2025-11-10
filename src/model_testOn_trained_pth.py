import argparse
import pandas as pd
import numpy as np
import math
import os
import scipy.io
import scipy.stats
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from joblib import dump, load
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, TensorDataset
from model_regression import Mlp, MAEAndRankLoss, preprocess_data, compute_correlation_metrics, logistic_func, plot_results


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == "cuda":
    torch.cuda.set_device(0)

def create_results_dataframe(data_list, network_name, srcc_list, krcc_list, plcc_list, rmse_list, select_criteria_list):
    df_results = pd.DataFrame(columns=['DATASET', 'MODEL', 'SRCC', 'KRCC', 'PLCC', 'RMSE', 'SELECT_CRITERIC'])
    df_results['DATASET'] = data_list
    df_results['MODEL'] = network_name
    df_results['SRCC'] = srcc_list
    df_results['KRCC'] = krcc_list
    df_results['PLCC'] = plcc_list
    df_results['RMSE'] = rmse_list
    df_results['SELECT_CRITERIC'] = select_criteria_list
    return df_results

def process_test_set(test_data_name, select_dimension, metadata_path, feature_path, network_name):
    if select_dimension == 'overall':
        test_df = pd.read_csv(f'{metadata_path}{test_data_name.upper()}_metadata.csv')
    else:
        test_df = pd.read_csv(f'{metadata_path}{test_data_name.upper()}_MOS_{select_dimension}_metadata.csv')

    test_vids = test_df['vid']
    mos = torch.tensor(test_df['mos'].astype(float), dtype=torch.float32)
    if test_data_name in ('konvid_1k', 'youtube_ugc_h264'):
        test_scores = ((mos - 1) * (99 / 4) + 1.0)
    else:
        test_scores = mos

    sorted_test_df = pd.DataFrame({
        'vid': test_df['vid'],
        'framerate': test_df['framerate'],
        'MOS': test_scores,
        'MOS_raw': mos
    })
    test_features = torch.load(f'{feature_path}/{network_name}_{test_data_name}_features.pt')
    print(f'num of {test_data_name} features ({feature_path}/{network_name}_{test_data_name}_features.pt): {len(test_features)}')
    return test_features, test_vids, test_scores, sorted_test_df

def fix_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        elif k == 'n_averaged':
            continue
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def collate_to_device(batch, device):
    data, targets = zip(*batch)
    return torch.stack(data).to(device), torch.stack(targets).to(device)

def model_test(best_model, X, y, device):
    test_dataset = TensorDataset(X, y)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    best_model.eval()
    y_pred = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            outputs = best_model(inputs)
            y_pred.extend(outputs.view(-1).tolist())
    return y_pred

def wo_fine_tune_model(model, device, model_path, X_test, y_test, loss_type, test_data_name):
    state_dict = torch.load(model_path)
    fixed_state_dict = fix_state_dict(state_dict)
    try:
        model.load_state_dict(fixed_state_dict)
    except RuntimeError as e:
        print(e)
    model.eval().to(device) # to gpu

    if loss_type == 'MAERankLoss':
        criterion = MAEAndRankLoss()
    else:
        criterion = torch.nn.MSELoss()

    # evaluate the model
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    test_loss = 0.0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        test_loss += loss.item() * inputs.size(0)
    average_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {average_loss}")

    y_test_pred = model_test(model, X_test, y_test, device)
    y_test_pred = torch.tensor(list(y_test_pred), dtype=torch.float32)
    if test_data_name in ('konvid_1k', 'youtube_ugc_h264'):
        y_test_convert = ((y_test - 1) / (99 / 4) + 1.0)
        y_test_pred_convert = ((y_test_pred - 1) / (99 / 4) + 1.0)
    else:
        y_test_convert = y_test
        y_test_pred_convert = y_test_pred

    y_test_pred_logistic, plcc_test, rmse_test, srcc_test, krcc_test = compute_correlation_metrics(y_test_convert.cpu().numpy(), y_test_pred_convert.cpu().numpy())
    test_pred_score = {'MOS': y_test_convert, 'y_test_pred': y_test_pred_convert, 'y_test_pred_logistic': y_test_pred_logistic}
    df_test_pred = pd.DataFrame(test_pred_score)
    return df_test_pred, y_test_convert, y_test_pred_logistic, plcc_test, rmse_test, srcc_test, krcc_test

def run(args):
    os.makedirs(os.path.join(args.report_path, 'predict_score'), exist_ok=True)
    results_csv = f'{args.report_path}/predict_score/{args.test_data_name}_mos_{args.select_dimension}_{args.network_name}_correlation.csv'
    predict_score_csv = f"{args.report_path}/predict_score/{args.test_data_name}_mos_{args.select_dimension}_{args.network_name}_predictScore.csv"

    print(f'Test dataset: {args.test_data_name}')
    print(f"MOS select_dimension: {args.select_dimension}")
    test_features, test_vids, test_scores, sorted_test_df = process_test_set(args.test_data_name, args.select_dimension, args.metadata_path, args.feature_path, args.network_name)
    X_test, y_test = preprocess_data(test_features, test_scores)

    # get save model param
    model = Mlp(input_features=X_test.shape[1], out_features=1, drop_rate=0.2, act_layer=nn.GELU)
    model = model.to(device)
    if args.select_dimension == 'overall':
        model_path = os.path.join(args.model_path, f"best_model/{args.train_data_name}_{args.network_name}_{args.model_name}_{args.select_criteria}_trained_model.pth")
    else:
        train_data_name = f"{args.train_data_name}_mos_{args.select_dimension}"
        model_path = os.path.join(args.model_path, f"semantic_embs_finevd_ablation/dimension/{train_data_name}_{args.network_name}_{args.model_name}_{args.select_criteria}_trained_model.pth")
    print(f'Trained Model: {model_path}')

    # without fine tune on the test dataset
    df_test_pred, y_test_convert, y_test_pred_logistic, plcc_test, rmse_test, srcc_test, krcc_test = wo_fine_tune_model(model, device, model_path, X_test, y_test, args.loss_type, args.test_data_name )

    # ===== predict score per video =====
    per_video_df = pd.DataFrame({
        'vid': test_vids.values,
        'MOS_raw': sorted_test_df['MOS_raw'].values,                # ground truth
        'MOS_used': df_test_pred['MOS'].values,                     # MOS （已按数据集规则转换
        'pred': df_test_pred['y_test_pred'].values,                 # predict score
        'pred_logistic': df_test_pred['y_test_pred_logistic'].values # logistic fit
    })
    per_video_df.to_csv(predict_score_csv, index=False, encoding="UTF-8")

    # ===== correlation metrics =====
    df_results = create_results_dataframe(
        data_list=[args.test_data_name],
        network_name=args.network_name,
        srcc_list=[srcc_test],
        krcc_list=[krcc_test],
        plcc_list=[plcc_test],
        rmse_list=[rmse_test],
        select_criteria_list=[args.select_criteria]
    )
    print(df_results.T)
    df_results.to_csv(results_csv, index=False, encoding="UTF-8")
    print(f"prediction scores saved to: {predict_score_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--train_data_name', type=str, default='finevd')
    parser.add_argument('--test_data_name', type=str, default='finevd')
    parser.add_argument('--network_name', type=str, default='camp-vqa')
    parser.add_argument('--model_name', type=str, default='Mlp')
    parser.add_argument('--select_criteria', type=str, default='byrmse', choices=['byrmse', 'bykrcc'])
    parser.add_argument('--select_dimension', type=str, default='overall', choices=['artifact', 'blur', 'color', 'noise', 'temporal', 'overall'])

    # paths
    parser.add_argument('--metadata_path', type=str, default='../metadata/')
    parser.add_argument('--feature_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='../model/')
    parser.add_argument('--report_path', type=str, default='../figs/captions_log/')

    # training parameters
    parser.add_argument('--n_repeats', type=int, default=21)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)

    # misc
    parser.add_argument('--loss_type', type=str, default='MAERankLoss')
    parser.add_argument('--optimizer_type', type=str, default='sgd')
    parser.add_argument('--initial_lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--use_swa', type=bool, default=True, help="Enable SWA (default: True)")
    parser.add_argument('--l1_w', type=float, default=0.6)
    parser.add_argument('--rank_w', type=float, default=1.0)

    args = parser.parse_args()
    if args.feature_path is None:
        args.feature_path = f'../features/{args.network_name}/'
    print(f"[Paths] metadata: {args.metadata_path}; features: {args.feature_path}; model: {args.model_path}; report: {args.report_path}")
    run(args)