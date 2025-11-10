import logging
import time
import os
import pandas as pd
import numpy as np
import math
import scipy.io
import scipy.stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import joblib

import seaborn as sns
import matplotlib.pyplot as plt
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from data_processing import split_train_test

# ignore all warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Mlp(nn.Module):
    def __init__(self, input_features, hidden_features=256, out_features=1, drop_rate=0.2, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        # self.bn1 = nn.BatchNorm1d(hidden_features)
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, hidden_features // 2)
        self.act2 = act_layer()
        self.drop2 = nn.Dropout(drop_rate)
        self.fc3 = nn.Linear(hidden_features // 2, out_features)

    def forward(self, input_feature):
        x = self.fc1(input_feature)
        # x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        output = self.fc3(x)
        return output


class MAEAndRankLoss(nn.Module):
    def __init__(self, l1_w=1.0, rank_w=1.0, margin=0.0, use_margin=False):
        super(MAEAndRankLoss, self).__init__()
        self.l1_w = l1_w
        self.rank_w = rank_w
        self.margin = margin
        self.use_margin = use_margin

    def forward(self, y_pred, y_true):
        # L1 loss/MAE loss
        l_mae = F.l1_loss(y_pred, y_true, reduction='mean') * self.l1_w
        # Rank loss
        n = y_pred.size(0)
        pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
        true_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)

        # e(ytrue_i, ytrue_j)
        masks = torch.sign(true_diff)

        if self.use_margin and self.margin > 0:
            true_diff = true_diff.abs() - self.margin
            true_diff = F.relu(true_diff)
            masks = true_diff.sign()

        l_rank = F.relu(true_diff - masks * pred_diff)
        l_rank = l_rank.sum() / (n * (n - 1))

        loss = l_mae + l_rank * self.rank_w
        return loss

def load_data(csv, data, data_name, set_name):
    try:
        df = pd.read_csv(csv, skiprows=[], header=None)
    except Exception as e:
        logging.error(f'Read CSV file error: {e}')
        raise

    y_data = df.values[1:, 2].astype(float)
    y = torch.tensor(y_data, dtype=torch.float32)

    if set_name == 'test':
        print(f"Modified y_true: {y}")
    X = data
    return X, y

def preprocess_data(X, y):
    X[torch.isnan(X)] = 0
    X[torch.isinf(X)] = 0

    # MinMaxScaler (use PyTorch implementation)
    X_min = X.min(dim=0, keepdim=True).values
    X_max = X.max(dim=0, keepdim=True).values
    X = (X - X_min) / (X_max - X_min)
    y = y.view(-1, 1).squeeze()
    return X, y

# define 4-parameter logistic regression
def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_logistic_regression(y_pred, y_true):
    beta = [np.max(y_true), np.min(y_true), np.mean(y_pred), 0.5]
    popt, _ = curve_fit(logistic_func, y_pred, y_true, p0=beta, maxfev=100000000)
    y_pred_logistic = logistic_func(y_pred, *popt)
    return y_pred_logistic, beta, popt

def compute_correlation_metrics(y_true, y_pred):
    y_pred_logistic, beta, popt = fit_logistic_regression(y_pred, y_true)

    plcc = scipy.stats.pearsonr(y_true, y_pred_logistic)[0]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_logistic))
    srcc = scipy.stats.spearmanr(y_true, y_pred)[0]

    try:
        krcc = scipy.stats.kendalltau(y_true, y_pred)[0]
    except Exception as e:
        logging.error(f'krcc calculation: {e}')
        krcc = scipy.stats.kendalltau(y_true, y_pred, method='asymptotic')[0]
    return y_pred_logistic, plcc, rmse, srcc, krcc

def plot_results(y_test, y_test_pred_logistic, df_pred_score, model_name, data_name, network_name, select_criteria):
    # nonlinear logistic fitted curve / logistic regression
    if isinstance(y_test, torch.Tensor):
        mos1 = y_test.numpy()
    y1 = y_test_pred_logistic

    try:
        beta = [np.max(mos1), np.min(mos1), np.mean(y1), 0.5]
        popt, pcov = curve_fit(logistic_func, y1, mos1, p0=beta, maxfev=100000000)
        sigma = np.sqrt(np.diag(pcov))
    except:
        raise Exception('Fitting logistic function time-out!!')
    x_values1 = np.linspace(np.min(y1), np.max(y1), len(y1))
    plt.plot(x_values1, logistic_func(x_values1, *popt), '-', color='#c72e29', label='Fitted f(x)')

    fig1 = sns.scatterplot(x="y_test_pred_logistic", y="MOS", data=df_pred_score, markers='o', color='steelblue', label=network_name)
    plt.legend(loc='upper left')
    if data_name == 'live_vqc' or data_name == 'live_qualcomm' or data_name == 'cvd_2014' or data_name == 'lsvq_train' or data_name == 'live_yt_gaming' or data_name == "finevd":
        plt.ylim(0, 100)
        plt.xlim(0, 100)
    else:
        plt.ylim(1, 5)
        plt.xlim(1, 5)
    plt.title(f"Algorithm {network_name} with {model_name} on dataset {data_name}", fontsize=10)
    plt.xlabel('Predicted Score')
    plt.ylabel('MOS')
    reg_fig1 = fig1.get_figure()

    fig_path = f'../figs/{data_name}/'
    os.makedirs(fig_path, exist_ok=True)
    reg_fig1.savefig(fig_path + f"{network_name}_{model_name}_{data_name}_{select_criteria}.png", dpi=300)
    plt.clf()
    plt.close()

def plot_and_save_losses(avg_train_losses, avg_val_losses, model_name, data_name, network_name, test_vids, i):
    plt.figure(figsize=(10, 6))

    plt.plot(avg_train_losses, label='Average Training Loss')
    plt.plot(avg_val_losses, label='Average Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Average Training and Validation Loss Across Folds - {network_name} with {model_name} (test_vids: {test_vids})', fontsize=10)

    plt.legend()
    fig_par_path = f'../log/result/{data_name}/'
    os.makedirs(fig_par_path, exist_ok=True)
    plt.savefig(f'{fig_par_path}/{network_name}_Average_Training_Loss_test{i}.png', dpi=50)
    plt.clf()
    plt.close()

def configure_logging(log_path, model_name, data_name, network_name, select_criteria):
    log_file_name = os.path.join(log_path, f"{data_name}_{network_name}_{model_name}_{select_criteria}.log")
    logging.basicConfig(filename=log_file_name, filemode='w', level=logging.DEBUG, format='%(levelname)s - %(message)s')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.info(f"Evaluating algorithm {network_name} with {model_name} on dataset {data_name}")
    logging.info(f"torch cuda: {torch.cuda.is_available()}")

def load_and_preprocess_data(metadata_path, feature_path, data_name, network_name, train_features, test_features):
    if data_name == 'lsvq_train':
        train_csv = os.path.join(metadata_path, f'mos_files/{data_name}_MOS_train.csv')
        test_csv = os.path.join(metadata_path, f'mos_files/{data_name}_MOS_test.csv')
        X_train, y_train = load_data(train_csv, train_features, data_name, 'train')
        X_test, y_test = load_data(test_csv, test_features, data_name, 'test')

    else:
        train_csv = os.path.join(metadata_path, f'mos_files/{data_name}_MOS_train.csv')
        test_csv = os.path.join(metadata_path, f'mos_files/{data_name}_MOS_test.csv')
        train_data = torch.load(f'{feature_path}split_train_test/{network_name}_{data_name}_train_features.pt')
        test_data = torch.load(f'{feature_path}split_train_test/{network_name}_{data_name}_test_features.pt')
        X_train, y_train = load_data(train_csv, train_data, data_name, 'train')
        X_test, y_test = load_data(test_csv, test_data, data_name, 'test')

    # standard min-max normalization of training features
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    return X_train, y_train, X_test, y_test

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)
    return train_loss

def evaluate(model, val_loader, criterion, device):
    """Evaluate model performance on validation sets"""
    model.eval()
    val_loss = 0.0
    y_val_pred = []
    y_val_true = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            y_val_pred.append(outputs)
            y_val_true.append(targets)
            loss = criterion(outputs, targets.view(-1, 1))
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    y_val_pred = torch.cat(y_val_pred, dim=0)
    y_val_true = torch.cat(y_val_true, dim=0)
    return val_loss, y_val_pred, y_val_true

def update_best_model(select_criteria, best_metric, current_val, model):
    is_better = False
    if select_criteria == 'byrmse' and current_val < best_metric:
        is_better = True
    elif select_criteria == 'bykrcc' and current_val > best_metric:
        is_better = True

    if is_better:
        return current_val, copy.deepcopy(model), is_better
    return best_metric, model, is_better

def train_and_evaluate(X_train, y_train, config):
    # parameters
    n_repeats = config['n_repeats']
    batch_size = config['batch_size']
    epochs = config['epochs']
    hidden_features = config['hidden_features']
    drop_rate = config['drop_rate']
    loss_type = config['loss_type']
    optimizer_type = config['optimizer_type']
    select_criteria = config['select_criteria']
    initial_lr = config['initial_lr']
    weight_decay = config['weight_decay']
    patience = config['patience']
    l1_w = config['l1_w']
    rank_w = config['rank_w']
    use_swa = config.get('use_swa', False)
    logging.info(f'Parameters - Number of repeats for 80-20 hold out test: {n_repeats}, Batch size: {batch_size}, Number of epochs: {epochs}')
    logging.info(f'Network Parameters - hidden_features: {hidden_features}, drop_rate: {drop_rate}, patience: {patience}')
    logging.info(f'Optimizer Parameters - loss_type: {loss_type}, optimizer_type: {optimizer_type}, initial_lr: {initial_lr}, weight_decay: {weight_decay}, use_swa: {use_swa}')
    logging.info(f'MAEAndRankLoss - l1_w: {l1_w}, rank_w: {rank_w}')

    # Split data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    best_model = None
    best_metric = float('inf') if select_criteria == 'byrmse' else float('-inf')

    # loss for every fold
    all_train_losses = []
    all_val_losses = []

    # initialisation of model, loss function, optimiser
    model = Mlp(input_features=X_train.shape[1], hidden_features=hidden_features, drop_rate=drop_rate)
    model = model.to(device) # to gpu

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
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)  # L2 Regularisation initial: 0.01, 1e-5
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)  # step_size=10, gamma=0.1: every 10 epochs lr*0.1
    if use_swa:
        swa_model = AveragedModel(model).to(device)
        swa_scheduler = SWALR(optimizer, swa_lr=initial_lr, anneal_strategy='cos')

    # dataset loader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []

    # early stopping parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_active = False
    swa_start = int(epochs * 0.7) if use_swa else epochs  # SWA starts after 70% of total epochs, only set SWA start if SWA is used

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        scheduler.step() # update learning rate
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            early_stop_active = True
            print(f"Current learning rate with SWA: {swa_scheduler.get_last_lr()}")

        lr = optimizer.param_groups[0]['lr']
        print('Epoch %d: Learning rate: %f' % (epoch + 1, lr))

        # decide which model to evaluate: SWA model or regular model
        current_model = swa_model if use_swa and epoch >= swa_start else model
        current_model.eval()
        val_loss, y_val_pred, y_val_true = evaluate(current_model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

        y_val_pred = torch.cat([pred for pred in y_val_pred])
        _, _, rmse_val, _, krcc_val = compute_correlation_metrics(y_val.cpu().numpy(), y_val_pred.cpu().numpy())
        current_metric = rmse_val if select_criteria == 'byrmse' else krcc_val
        best_metric, best_model, is_better = update_best_model(select_criteria, best_metric, current_metric, current_model)
        if is_better:
            logging.info(f"Epoch {epoch + 1}:")
            y_val_pred_logistic_tmp, plcc_valid_tmp, rmse_valid_tmp, srcc_valid_tmp, krcc_valid_tmp = compute_correlation_metrics(y_val.cpu().numpy(), y_val_pred.cpu().numpy())
            logging.info(f'Validation set - Evaluation Results - SRCC: {srcc_valid_tmp}, KRCC: {krcc_valid_tmp}, PLCC: {plcc_valid_tmp}, RMSE: {rmse_valid_tmp}')

            X_train_fold_tensor = X_train
            y_tra_pred_tmp = best_model(X_train_fold_tensor).detach().cpu().squeeze()
            y_tra_pred_logistic_tmp, plcc_train_tmp, rmse_train_tmp, srcc_train_tmp, krcc_train_tmp = compute_correlation_metrics(y_train.cpu().numpy(), y_tra_pred_tmp.cpu().numpy())
            logging.info(f'Train set - Evaluation Results - SRCC: {srcc_train_tmp}, KRCC: {krcc_train_tmp}, PLCC: {plcc_train_tmp}, RMSE: {rmse_train_tmp}')

        # check for loss improvement
        if early_stop_active:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # save the best model if validation loss improves
                best_model = copy.deepcopy(model)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    # epochs to wait for improvement before stopping
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        # saving SWA models and updating BN statistics
        if use_swa:
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            best_model = best_model.to(device)
            best_model.eval()
            torch.optim.swa_utils.update_bn(train_loader, best_model)

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        max_length = max(len(x) for x in all_train_losses)
        all_train_losses = [x + [x[-1]] * (max_length - len(x)) for x in all_train_losses]
        max_length = max(len(x) for x in all_val_losses)
        all_val_losses = [x + [x[-1]] * (max_length - len(x)) for x in all_val_losses]

    return best_model, all_train_losses, all_val_losses

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

def main(config):
    model_name = config['model_name']
    data_name = config['data_name']
    network_name = config['network_name']

    metadata_path = config['metadata_path']
    feature_path = config['feature_path']
    log_path = config['log_path']
    save_path = config['save_path']
    score_path = config['score_path']
    result_path = config['result_path']

    # parameters
    select_criteria = config['select_criteria']
    n_repeats = config['n_repeats']

    # logging and result
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(score_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    result_file = f'{result_path}{data_name}_{network_name}_{model_name}_{select_criteria}.mat'
    pred_score_filename = os.path.join(score_path, f"{data_name}_{network_name}_{model_name}_Predicted_Score_{select_criteria}.csv")
    file_path = os.path.join(save_path, f"{data_name}_{network_name}_{model_name}_{select_criteria}_trained_model.pth")
    configure_logging(log_path, model_name, data_name, network_name, select_criteria)

    '''======================== Main Body ==========================='''
    PLCC_all_repeats_test = []
    SRCC_all_repeats_test = []
    KRCC_all_repeats_test = []
    RMSE_all_repeats_test = []
    PLCC_all_repeats_train = []
    SRCC_all_repeats_train = []
    KRCC_all_repeats_train = []
    RMSE_all_repeats_train = []
    all_repeats_test_vids = []
    all_repeats_df_test_pred = []
    best_model_list = []

    for i in range(1, n_repeats + 1):
        print(f"{i}th repeated 80-20 hold out test")
        logging.info(f"{i}th repeated 80-20 hold out test")
        t0 = time.time()

        # train test split
        test_size = 0.2
        random_state = math.ceil(8.8 * i)
        # NR: original
        if data_name == 'lsvq_train':
            test_data_name = 'lsvq_test' #lsvq_test, lsvq_test_1080p
            train_features, test_features, test_vids = split_train_test.process_lsvq(data_name, test_data_name, metadata_path, feature_path, network_name)
        else:
            _, _, test_vids = split_train_test.process_other(data_name, test_size, random_state, metadata_path, feature_path, network_name)

        '''======================== read files =============================== '''
        if data_name == 'lsvq_train':
            X_train, y_train, X_test, y_test = load_and_preprocess_data(metadata_path, feature_path, data_name, network_name, train_features, test_features)
        else:
            X_train, y_train, X_test, y_test = load_and_preprocess_data(metadata_path, feature_path, data_name, network_name, None, None)

        '''======================== regression model =============================== '''
        best_model, all_train_losses, all_val_losses = train_and_evaluate(X_train, y_train, config)

        # average loss plots
        avg_train_losses = np.mean(all_train_losses, axis=0)
        avg_val_losses = np.mean(all_val_losses, axis=0)
        test_vids = test_vids.tolist()
        plot_and_save_losses(avg_train_losses, avg_val_losses, model_name, data_name, network_name, len(test_vids), i)

        # predict best model on the train dataset
        y_train_pred = model_test(best_model, X_train, y_train, device)
        y_train_pred = torch.tensor(list(y_train_pred), dtype=torch.float32)
        y_train_pred_logistic, plcc_train, rmse_train, srcc_train, krcc_train = compute_correlation_metrics(y_train.cpu().numpy(), y_train_pred.cpu().numpy())

        # test best model on the test dataset
        y_test_pred = model_test(best_model, X_test, y_test, device)
        y_test_pred = torch.tensor(list(y_test_pred), dtype=torch.float32)
        y_test_pred_logistic, plcc_test, rmse_test, srcc_test, krcc_test = compute_correlation_metrics(y_test.cpu().numpy(), y_test_pred.cpu().numpy())

        # save the predict score results
        test_pred_score = {'MOS': y_test, 'y_test_pred': y_test_pred, 'y_test_pred_logistic': y_test_pred_logistic}
        df_test_pred = pd.DataFrame(test_pred_score)

        # logging logistic predicted scores
        logging.info("============================================================================================================")
        SRCC_all_repeats_test.append(srcc_test)
        KRCC_all_repeats_test.append(krcc_test)
        PLCC_all_repeats_test.append(plcc_test)
        RMSE_all_repeats_test.append(rmse_test)
        SRCC_all_repeats_train.append(srcc_train)
        KRCC_all_repeats_train.append(krcc_train)
        PLCC_all_repeats_train.append(plcc_train)
        RMSE_all_repeats_train.append(rmse_train)
        all_repeats_test_vids.append(test_vids)
        all_repeats_df_test_pred.append(df_test_pred)
        best_model_list.append(copy.deepcopy(best_model))

        # logging.info results for each iteration
        logging.info('Best results in Mlp model within one split')
        logging.info(f'MODEL: {best_model}')
        logging.info('======================================================')
        logging.info(f'Train set - Evaluation Results')
        logging.info(f'SRCC_train: {srcc_train}')
        logging.info(f'KRCC_train: {krcc_train}')
        logging.info(f'PLCC_train: {plcc_train}')
        logging.info(f'RMSE_train: {rmse_train}')
        logging.info('======================================================')
        logging.info(f'Test set - Evaluation Results')
        logging.info(f'SRCC_test: {srcc_test}')
        logging.info(f'KRCC_test: {krcc_test}')
        logging.info(f'PLCC_test: {plcc_test}')
        logging.info(f'RMSE_test: {rmse_test}')
        logging.info('======================================================')
        logging.info(' -- {} seconds elapsed...\n\n'.format(time.time() - t0))

    logging.info('')
    SRCC_all_repeats_test = torch.tensor(SRCC_all_repeats_test, dtype=torch.float32)
    KRCC_all_repeats_test = torch.tensor(KRCC_all_repeats_test, dtype=torch.float32)
    PLCC_all_repeats_test = torch.tensor(PLCC_all_repeats_test, dtype=torch.float32)
    PLCC_all_repeats_test = PLCC_all_repeats_test[~torch.isnan(PLCC_all_repeats_test)]
    RMSE_all_repeats_test = torch.tensor(RMSE_all_repeats_test, dtype=torch.float32)
    SRCC_all_repeats_train = torch.tensor(SRCC_all_repeats_train, dtype=torch.float32)
    KRCC_all_repeats_train = torch.tensor(KRCC_all_repeats_train, dtype=torch.float32)
    PLCC_all_repeats_train = torch.tensor(PLCC_all_repeats_train, dtype=torch.float32)
    RMSE_all_repeats_train = torch.tensor(RMSE_all_repeats_train, dtype=torch.float32)

    logging.info('======================================================')
    logging.info('Average training results among all repeated 80-20 holdouts:')
    logging.info('SRCC: %f (std: %f)', torch.median(SRCC_all_repeats_train).item(), torch.std(SRCC_all_repeats_train).item())
    logging.info('KRCC: %f (std: %f)', torch.median(KRCC_all_repeats_train).item(), torch.std(KRCC_all_repeats_train).item())
    logging.info('PLCC: %f (std: %f)', torch.median(PLCC_all_repeats_train).item(), torch.std(PLCC_all_repeats_train).item())
    logging.info('RMSE: %f (std: %f)', torch.median(RMSE_all_repeats_train).item(), torch.std(RMSE_all_repeats_train).item())
    logging.info('======================================================')
    logging.info('Average testing results among all repeated 80-20 holdouts:')
    logging.info('SRCC: %f (std: %f)', torch.median(SRCC_all_repeats_test).item(), torch.std(SRCC_all_repeats_test).item())
    logging.info('KRCC: %f (std: %f)', torch.median(KRCC_all_repeats_test).item(), torch.std(KRCC_all_repeats_test).item())
    logging.info('PLCC: %f (std: %f)', torch.median(PLCC_all_repeats_test).item(), torch.std(PLCC_all_repeats_test).item())
    logging.info('RMSE: %f (std: %f)', torch.median(RMSE_all_repeats_test).item(), torch.std(RMSE_all_repeats_test).item())
    logging.info('======================================================')
    logging.info('\n')

    # find the median model and the index of the median
    print('======================================================')
    if select_criteria == 'byrmse':
        median_metrics = torch.median(RMSE_all_repeats_test).item()
        indices = (RMSE_all_repeats_test == median_metrics).nonzero(as_tuple=True)[0].tolist()
        select_criteria = select_criteria.replace('by', '').upper()
        print(RMSE_all_repeats_test)
        logging.info(f'all {select_criteria}: {RMSE_all_repeats_test}')
    elif select_criteria == 'bykrcc':
        median_metrics = torch.median(KRCC_all_repeats_test).item()
        indices = (KRCC_all_repeats_test == median_metrics).nonzero(as_tuple=True)[0].tolist()
        select_criteria = select_criteria.replace('by', '').upper()
        print(KRCC_all_repeats_test)
        logging.info(f'all {select_criteria}: {KRCC_all_repeats_test}')

    median_test_vids = [all_repeats_test_vids[i] for i in indices]
    test_vids = [arr.tolist() for arr in median_test_vids] if len(median_test_vids) > 1 else (median_test_vids[0] if median_test_vids else [])

    # select the model with the first index where the median is located
    # Note: If there are multiple iterations with the same median RMSE, the first index is selected here
    median_model = None
    if len(indices) > 0:
        median_index = indices[0]  # select the first index
        median_model = best_model_list[median_index]
        median_model_df_test_pred = all_repeats_df_test_pred[median_index]

        median_model_df_test_pred.to_csv(pred_score_filename, index=False)
        plot_results(y_test, y_test_pred_logistic, median_model_df_test_pred, model_name, data_name, network_name, select_criteria)

    print(f'Median Metrics: {median_metrics}')
    print(f'Indices: {indices}')
    # print(f'Test Videos: {test_vids}')
    print(f'Best model: {median_model}')

    logging.info(f'median test {select_criteria}: {median_metrics}')
    logging.info(f"Indices of median metrics: {indices}")
    # logging.info(f'Best training and test dataset: {test_vids}')
    logging.info(f'Best model predict score: {median_model_df_test_pred}')
    logging.info(f'Best model: {median_model}')

    # ================================================================================
    # save mats
    scipy.io.savemat(result_file, mdict={'SRCC_train': SRCC_all_repeats_train.numpy(),
                                         'KRCC_train': KRCC_all_repeats_train.numpy(),
                                         'PLCC_train': PLCC_all_repeats_train.numpy(),
                                         'RMSE_train': RMSE_all_repeats_train.numpy(),
                                         'SRCC_test': SRCC_all_repeats_test.numpy(),
                                         'KRCC_test': KRCC_all_repeats_test.numpy(),
                                         'PLCC_test': PLCC_all_repeats_test.numpy(),
                                         'RMSE_test': RMSE_all_repeats_test.numpy(),
                                         f'Median_{select_criteria}': median_metrics,
                                         'Test_Videos_list': all_repeats_test_vids,
                                         'Test_videos_Median_model': test_vids})

    # save model
    torch.save(median_model.state_dict(), file_path)
    print(f"Model state_dict saved to {file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--model_name', type=str, default='Mlp')
    parser.add_argument('--data_name', type=str, default='konvid_1k')
    parser.add_argument('--network_name', type=str, default='camp-vqa')

    parser.add_argument('--metadata_path', type=str, default='../metadata/')
    parser.add_argument('--feature_path', type=str, default=f'../features/camp-vqa/')
    parser.add_argument('--log_path', type=str, default='../log/')
    parser.add_argument('--save_path', type=str, default='../model/')
    parser.add_argument('--score_path', type=str, default='../log/predict_score/')
    parser.add_argument('--result_path', type=str, default='../log/result/')
    # training parameters
    parser.add_argument('--select_criteria', type=str, default='byrmse')
    parser.add_argument('--n_repeats', type=int, default=21)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden_features', type=int, default=256)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    # misc
    parser.add_argument('--loss_type', type=str, default='MAERankLoss')
    parser.add_argument('--optimizer_type', type=str, default='sgd')
    parser.add_argument('--initial_lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--use_swa', type=bool, default=True)
    parser.add_argument('--l1_w', type=float, default=0.6)
    parser.add_argument('--rank_w', type=float, default=1.0)

    args = parser.parse_args()
    config = vars(args)  # args to dict
    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    if device.type == "cuda":
        torch.cuda.set_device(0)

    main(config)