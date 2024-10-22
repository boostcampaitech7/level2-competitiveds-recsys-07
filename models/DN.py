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

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)  # Set seed for reproducibility

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_data = pd.read_csv("../../data/train.csv")
test_data = pd.read_csv("../../data/test.csv")

# Split features and target
X_train = train_data.drop(columns="deposit").values
y_train = train_data["deposit"].values
X_test = test_data.drop(columns="deposit").values
y_test = test_data["deposit"].values

# Preprocess data (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors and move to device (GPU/CPU)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# Define the neural network model class
class Net(nn.Module):
    def __init__(self, input_size, num_layers, units, activation):
        super(Net, self).__init__()
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_size, units[0]))
        layers.append(activation())
        
        # Hidden layers
        for i in range(1, num_layers):
            layers.append(nn.Linear(units[i-1], units[i]))
            layers.append(activation())
        
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
    activation = trial.suggest_categorical('activation', [nn.ReLU(), nn.Tanh()])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    epochs = trial.suggest_int('epochs', 10, 300)  # Suggesting number of epochs

    # Create the model and move it to the device (GPU/CPU)
    model = Net(input_size=X_train.shape[1], num_layers=num_layers, units=units, activation=activation).to(device)

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
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


        model = Net(input_size=X_train.shape[1], num_layers=num_layers, units=units, activation=activation).to(device)

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
            y_pred_valid = model(x_valid_fold).cpu().numpy()
            mae = mean_absolute_error(y_valid_fold.cpu().numpy(), y_pred_valid)
            mae_scores.append(mae)

    # Return the average MAE across the folds
    return np.mean(mae_scores)

# Create the Optuna study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective_deeplearning, n_trials=100)

# Output the results of the best trial
trial = study.best_trial
print(f"Best MAE: {trial.value}")
print("Best hyperparameters: {}".format(trial.params))

# Final Model Training with Best Hyperparameters
best_params = trial.params
num_layers = best_params['num_layers']
units = [best_params[f'n_units_l{i}'] for i in range(num_layers)]
activation = best_params['activation']
learning_rate = best_params['learning_rate']
batch_size = best_params['batch_size']

# Create the final model and move
