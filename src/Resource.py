import torch
import numpy as np
from torch import nn
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import csv

def stampa_metriche(nome, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # Calcola la radice quadrata dell'MSE
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Performance in {nome}:")
    print(f"\tMSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R^2: {r2:.4f}")


def stampa_colonne(percorso_file):
    with open(percorso_file, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # legge solo la prima riga

    print(f"Colonne di '{percorso_file}' ({len(header)} totali):")
    for i, colonna in enumerate(header):
        print(f"  {i}: {colonna}")

class LinearRegressor(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressor, self).__init__()

        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

def train_regression_model(
    model,
    X_train, y_train,
    X_test,  y_test,
    exp_name,
    criterion,
    optimizer=None,
    epochs=100,
    lr=0.01,
    momentum=0.0,
    weight_decay=0.0,
    device=None,
    verbose=False,
    clear_gpu_after=False
):
    if device is None:
        device = next(model.parameters()).device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test  = X_test.to(device)
    y_test  = y_test.to(device)

    if optimizer is None:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    writer = SummaryWriter(log_dir=f"logs/{exp_name}")

    for e in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train).squeeze()
        loss = criterion(out, y_train)
        writer.add_scalar('loss/train', loss.item(), global_step=e)
        loss.backward()
        optimizer.step()
        model.eval()

        with torch.no_grad():
            preds_test = model(X_test).squeeze()
            loss_test = criterion(preds_test, y_test)
            writer.add_scalar('loss/val', loss_test.item(), global_step=e)

        if (e + 1) % 10 == 0 and verbose:
            print(f"Epoch [{e+1}/{epochs}] \t\t loss training: {loss.item():.4f} \t loss_val: {loss_test.item():.4f}")

    writer.close()
    if clear_gpu_after:
        torch.cuda.empty_cache()

    return model