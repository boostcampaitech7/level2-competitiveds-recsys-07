import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd
import optuna
import logging


# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)  # Set seed for reproducibility

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Load data
ryu_data = pd.read_csv("/data/ephemeral/home/data/ryu/train.csv")
ryu_data_test = pd.read_csv("/data/ephemeral/home/data/ryu/test.csv")
yang_data = pd.read_csv("/data/ephemeral/home/data/yang/train.csv")
yang_data_test = pd.read_csv("/data/ephemeral/home/data/yang/test.csv")
sample_submission = pd.read_csv("/data/ephemeral/home/data/original/sample_submission.csv")

new_columns_train_ryu = [col for col in ryu_data.columns if col != "index" and col not in yang_data.columns]

merge1_train_data = pd.merge(yang_data, ryu_data[["index"] + new_columns_train_ryu], on="index", how="left")

new_columns_test_ryu = [col for col in ryu_data_test.columns if col != "index" and col not in yang_data_test.columns]

merge1_test_data = pd.merge(yang_data_test, ryu_data_test[["index"] + new_columns_test_ryu], on="index", how="left")

train_data = merge1_train_data
test_data = merge1_test_data

# Split features and target
X_train = train_data.drop(columns=["deposit", "index"]).values  # numpy 배열로 변환
y_train = train_data["deposit"].values
X_test = test_data.drop(columns="index").values

# Preprocess data (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors and move to device (GPU/CPU)
print("Starting tensor conversion...")
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
print("Tensor conversion done.")

# Define the neural network model class
class Net(nn.Module):
    def __init__(self, input_size, num_layers, units):
        super(Net, self).__init__()
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_size, units[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(1, num_layers):
            layers.append(nn.Linear(units[i-1], units[i]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(units[-1], 1))  # 1 output for regression
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# Define the objective function for Optuna
def objective_deeplearning(trial):
    # Hyperparameter search space
    num_layers = trial.suggest_int('num_layers', 1, 3)
    units = [trial.suggest_int(f'n_units_l{i}', 16, 128) for i in range(num_layers)]
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 256)
    epochs = trial.suggest_int('epochs', 10, 300)  # Suggesting number of epochs
    
    # Create the model and move it to the device (GPU/CPU)
    model = Net(input_size=X_train.shape[1], num_layers=num_layers, units=units).to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # K-Fold Cross Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scores = []

    for train_idx, valid_idx in kfold.split(X_train_scaled):
        # Split data into training and validation folds
        x_train_fold = X_train_tensor[train_idx]
        y_train_fold = y_train_tensor[train_idx]
        x_valid_fold = X_train_tensor[valid_idx]
        y_valid_fold = y_train_tensor[valid_idx]

        # Create DataLoader for the current training fold
        train_dataset = TensorDataset(x_train_fold, y_train_fold)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        model = Net(input_size=X_train.shape[1], num_layers=num_layers, units=units).to(device)

        # Train the model for the number of epochs suggested by Optuna
        for epoch in range(epochs):  # Epoch is now part of the hyperparameter search space
            model.train()
            for batch_x, batch_y in train_dataloader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            # Make predictions on the validation set
            y_pred_valid = model(x_valid_fold)
            mae = torch_mean_absolute_error(y_valid_fold, y_pred_valid)
            mae_scores.append(mae.item())  # Convert MAE tensor to a Python float

    # Return the average MAE across the folds
    return np.mean(mae_scores)

# MAE using PyTorch
def torch_mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))


# Create the Optuna study and optimize
optuna.logging.set_verbosity(optuna.logging.DEBUG)
study = optuna.create_study(direction="minimize")
study.optimize(objective_deeplearning, n_trials=1)

# Output the results of the best trial
trial = study.best_trial
print(f"Best MAE: {trial.value}")
print("Best hyperparameters: {}".format(trial.params))

# Final Model Training with Best Hyperparameters
best_params = trial.params
num_layers = best_params['num_layers']
units = [best_params[f'n_units_l{i}'] for i in range(num_layers)]
learning_rate = best_params['learning_rate']
batch_size = best_params['batch_size']
epochs = best_params['epochs']

# Create the final model and move

with open("parameters_DN.txt", "w+") as f:
    f.write("lgb_params :\n")
    for key, value in best_params.items():
        f.write(f"{key}: {value}\n")
        
model = Net(input_size=X_train.shape[1], num_layers=num_layers, units=units).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    for batch_x, batch_y in train_dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
model.eval()
with torch.no_grad():
    test_pred = model(X_test_tensor).cpu().numpy()

sample_submission["deposit"] = test_pred  # 예측 결과를 "deposit" 열에 추가
sample_submission.to_csv("output_pytorch.csv", index=False, encoding="utf-8-sig")