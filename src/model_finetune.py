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
from model_regression_lsvq import Mlp, MAEAndRankLoss, preprocess_data, compute_correlation_metrics, logistic_func, plot_results

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

def process_test_set(test_data_name, metadata_path, feature_path, network_name):
    test_df = pd.read_csv(f'{metadata_path}/{test_data_name.upper()}_metadata.csv')

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
    print(f'num of {test_data_name} features: {len(test_features)}')
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

def fine_tune_model(model, device, model_path, X_fine_tune, y_fine_tune, save_path, batch_size, epochs, loss_type, optimizer_type, initial_lr, weight_decay, use_swa, l1_w, rank_w):
    state_dict = torch.load(model_path)
    fixed_state_dict = fix_state_dict(state_dict)
    try:
        model.load_state_dict(fixed_state_dict)
    except RuntimeError as e:
        print(e)

    for param in model.parameters():
        param.requires_grad = True
    model.train().to(device) # to gpu

    fine_tune_dataset = TensorDataset(X_fine_tune, y_fine_tune)
    fine_tune_loader = DataLoader(dataset=fine_tune_dataset, batch_size=batch_size, shuffle=False)

    # initialisation of loss function, optimiser
    if loss_type == 'MAERankLoss':
        criterion = MAEAndRankLoss()
        criterion.l1_w = l1_w
        criterion.rank_w = rank_w
    else:
        criterion = nn.MSELoss()

    if optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)# initial eta_min=1e-5
    else:
        optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)  # L2 Regularisation initial: 0.01, 1e-5
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)  # step_size=10, gamma=0.1: every 10 epochs lr*0.1
    if use_swa:
        swa_model = AveragedModel(model).to(device)
        swa_scheduler = SWALR(optimizer, swa_lr=initial_lr, anneal_strategy='cos')
    swa_start = int(epochs * 0.75) if use_swa else epochs  # SWA starts after 75% of total epochs, only set SWA start if SWA is used

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in fine_tune_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        scheduler.step()
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print(f"Current learning rate with SWA: {swa_scheduler.get_last_lr()}")
        avg_loss = epoch_loss / len(fine_tune_loader.dataset)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # decide which model to evaluate: SWA model or regular model
        current_model = swa_model if use_swa and epoch >= swa_start else model
        # Save best model state
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = copy.deepcopy(current_model)

    # decide which model to evaluate: SWA model or regular model
    if use_swa and epoch >= swa_start:
        train_loader = DataLoader(dataset=fine_tune_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_to_device(x, device))
        best_model = best_model.to(device)
        best_model.eval()
        torch.optim.swa_utils.update_bn(train_loader, best_model)
    # model_path_new = os.path.join(save_path, f"{test_data_name}_diva-vqa_fine_tuned_model.pth")
    # torch.save(best_model.state_dict(), model_path_new)  # save finetuned model
    return best_model

def fine_tuned_model_test(model, device, X_test, y_test, test_data_name):
    model.eval()
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
    data_list, srcc_list, krcc_list, plcc_list, rmse_list, select_criteria_list = [], [], [], [], [], []

    os.makedirs(os.path.join(args.report_path, 'fine_tune'), exist_ok=True)
    if args.is_finetune:
        csv_name = f'{args.report_path}/fine_tune/{args.test_data_name}_{args.network_name}_{args.select_criteria}_finetune.csv'
    else:
        csv_name = f'{args.report_path}/fine_tune/{args.test_data_name}_{args.network_name}_{args.select_criteria}_wo_finetune.csv'
    print(f'Test dataset: {args.test_data_name}')
    test_features, test_vids, test_scores, sorted_test_df = process_test_set(args.test_data_name, args.metadata_path, args.feature_path, args.network_name)
    X_test, y_test = preprocess_data(test_features, test_scores)

    # get save model param
    model = Mlp(input_features=X_test.shape[1], out_features=1, drop_rate=0.2, act_layer=nn.GELU)
    model = model.to(device)
    model_path = os.path.join(args.model_path, f"{args.train_data_name}_{args.network_name}_{args.model_name}_{args.select_criteria}_trained_model_kfold.pth")

    model_results = []
    for i in range(1, args.n_repeats + 1):
        print(f"{i}th repeated 80-20 hold out test")
        X_fine_tune, X_final_test, y_fine_tune, y_final_test = train_test_split(X_test, y_test, test_size=0.2, random_state=math.ceil(8.8 * i))
        if args.is_finetune:
            # test fine tuned model on the test dataset
            ft_model = fine_tune_model(model, device, model_path, X_fine_tune, y_fine_tune, args.report_path, args.batch_size,
                                       args.epochs, args.loss_type, args.optimizer_type, args.initial_lr, args.weight_decay, args.use_swa, args.l1_w, args.rank_w)
            df_test_pred, y_test_convert, y_test_pred_logistic, plcc_test, rmse_test, srcc_test, krcc_test = fine_tuned_model_test(ft_model, device, X_final_test, y_final_test, args.test_data_name)
            best_model = copy.deepcopy(ft_model)
        else:
            # without fine tune on the test dataset
            df_test_pred, y_test_convert, y_test_pred_logistic, plcc_test, rmse_test, srcc_test, krcc_test = wo_fine_tune_model(model, device, model_path, X_test, y_test, args.loss_type, args.test_data_name)
            print(y_test_pred_logistic)
            best_model = copy.deepcopy(model)

        model_results.append({
            'model': best_model,
            'srcc': srcc_test,
            'krcc': krcc_test,
            'plcc': plcc_test,
            'rmse': rmse_test,
            'df_pred': df_test_pred
        })
        print('\n')

    if args.select_criteria == 'byrmse':
        sorted_results = sorted(model_results, key=lambda x: x['rmse'])
    elif args.select_criteria == 'bykrcc':
        sorted_results = sorted(model_results, key=lambda x: x['krcc'], reverse=True)
    else:
        raise ValueError(f"Unknown select_criteria: {args.select_criteria}")
    median_index = len(sorted_results) // 2
    median_result = sorted_results[median_index]
    median_model = median_result['model']
    median_df_test_pred = median_result['df_pred']
    median_srcc_test = median_result['srcc']
    median_krcc_test = median_result['krcc']
    median_plcc_test = median_result['plcc']
    median_rmse_test = median_result['rmse']
    data_list.append(args.test_data_name)
    srcc_list.append(median_srcc_test)
    krcc_list.append(median_krcc_test)
    plcc_list.append(median_plcc_test)
    rmse_list.append(median_rmse_test)
    select_criteria_list.append(args.select_criteria)
    median_df_test_pred.head()

    # save finetuned model
    if args.is_finetune:
        model_path_new = os.path.join(args.report_path, f"{args.test_data_name}_{args.network_name}_fine_tuned_model.pth")
        torch.save(median_model.state_dict(), model_path_new)
        print(f"Median model select {args.select_criteria} saved to {model_path_new}")

    df_results = create_results_dataframe(data_list, args.network_name, srcc_list, krcc_list, plcc_list, rmse_list, select_criteria_list)
    print(df_results.T)
    df_results.to_csv(csv_name, index=None, encoding="UTF-8")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--train_data_name', type=str, default='lsvq_train')
    parser.add_argument('--test_data_name', type=str, default='finevd')
    parser.add_argument('--network_name', type=str, default='camp-vqa')
    parser.add_argument('--model_name', type=str, default='Mlp')
    parser.add_argument('--select_criteria', type=str, default='byrmse', choices=['byrmse', 'bykrcc'])

    # paths
    parser.add_argument('--metadata_path', type=str, default='../metadata/')
    parser.add_argument('--feature_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='../model/')
    parser.add_argument('--report_path', type=str, default='../log/')

    # training parameters
    parser.add_argument('--is_finetune', action='store_true', help="Enable fine-tuning")
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