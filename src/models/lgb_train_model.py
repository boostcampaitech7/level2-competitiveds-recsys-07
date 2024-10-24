from typing import Any

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

# sample_submission = pd.read_csv("/data/ephemeral/home/data/original/sample_submission.csv")
train_path = "/data/ephemeral/home/data/han/final_train.csv"
test_path = "/data/ephemeral/home/data/han/final_test.csv"


train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# train 및 test 구분
X_train = train_data.drop(columns="deposit")
y_train = train_data["deposit"]
X_test = test_data.copy()

if "index" in X_train.columns:
    X_train = X_train.drop(columns="index")
if "index" in X_test.columns:
    X_test = X_test.drop(columns="index")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Train and Test columns do not match.")
    sys.exit()

def lgb_model_train(trial: Any, X_train: pd.DataFrame, y_train: pd.Series, cv: int) -> float:

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 6, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),  # L1 regularization
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 10),  # L2 regularization
        "random_state": 42,
        "metric": "mae",
        "early_stopping_rounds": 35
    }

    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    mae_scores = []

    for train_idx, valid_idx in kfold.split(X_train):
        x_train_fold, x_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        # Create LightGBM Regressor model
        lgb_model = LGBMRegressor(**params)

        # Train the model
        lgb_model.fit(
            x_train_fold,
            y_train_fold,
            eval_set=[(x_valid_fold, y_valid_fold)],
        )

        # Predict on validation set
        y_pred_valid = lgb_model.predict(x_valid_fold)

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_valid_fold, y_pred_valid)
        mae_scores.append(mae)

    # Return the mean MAE across all folds
    return np.mean(mae_scores)


# Initialize and run the study
seed = 42
sampler = TPESampler(seed=seed)

study = optuna.create_study(direction="minimize", sampler=sampler)  # Minimize MAE
study.optimize(lambda trial: lgb_model_train(trial, X_train, y_train, cv=5), n_trials=100)

# Output the results of the best trial
trial = study.best_trial
print(f"Sampler is {study.sampler.__class__.__name__}")
print("Best MAE: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

# Train the LightGBM model with the best parameters on the entire dataset
best_params = trial.params
lgb_model = LGBMRegressor(**best_params)
lgb_model.fit(X_train, y_train)

# Feature importance
feature_importances = lgb_model.feature_importances_
features = X_train.columns

# Print feature importances
importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances}).sort_values(
    by="Importance", ascending=False
)

print("\nFeature Importances:")
print(importance_df)

# Save the best hyperparameters to parameters.txt
with open("parameters_lgb.txt", "w+") as f:
    f.write("lgb_params :\n")
    for key, value in trial.params.items():
        f.write(f"{key}: {value}\n")
