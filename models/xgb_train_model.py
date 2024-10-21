import sys
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

# 함수화 함수 따로 만들기 (data_pre_processor로 넣기)
ryu_data = pd.read_csv("/data/ephemeral/home/data/ryu/train.csv")
ryu_data_test = pd.read_csv("/data/ephemeral/home/data/ryu/test.csv")
yang_data = pd.read_csv("/data/ephemeral/home/data/yang/train.csv")
yang_data_test = pd.read_csv("/data/ephemeral/home/data/yang/test.csv")

new_columns = [col for col in yang_data.columns if col != "index" and col not in ryu_data.columns]
train_data = pd.merge(ryu_data, yang_data[["index"] + new_columns], on="index", how="left")
test_data = pd.merge(ryu_data_test, yang_data_test[["index"] + new_columns], on="index", how="left")

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

# Model(optuna를 이용)


def xgb_model_train(trial: Any, X_train: pd.DataFrame, y_train: pd.Series, cv: int) -> float:

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "random_state": 42,
        "tree_method": "hist",
        "device": "cuda",
    }

    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    mae_scores = []

    for train_idx, valid_idx in kfold.split(X_train):
        x_train_fold, x_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        # Create XGBoost Regressor model
        xgb_model = XGBRegressor(**params)

        # Train the model
        xgb_model.fit(
            x_train_fold,
            y_train_fold,
            eval_set=[(x_valid_fold, y_valid_fold)],
        )

        # Predict on validation set
        y_pred_valid = xgb_model.predict(x_valid_fold)

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_valid_fold, y_pred_valid)
        mae_scores.append(mae)

    # Return the mean MAE across all folds
    return np.mean(mae_scores)


# Initialize and run the study
seed = 42
sampler = TPESampler(seed=seed)

# Create the study with the defined sampler (using the seed)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(lambda trial: xgb_model_train(trial, X_train, y_train, cv=5), n_trials=100)

# Output the results of the best trial
trial = study.best_trial
print(f"Sampler is {study.sampler.__class__.__name__}")
print("Best MAE: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))


# 최적의 파라미터들을 parameters.txt에 넣음.
with open("parameters_xgb.txt", "w+") as f:
    f.write("xgb_params :\n")
    for key, value in trial.params.items():
        f.write(f"{key}: {value}\n")


# 추가로 importance feature보기.
best_params = trial.params
xgb_final_model = XGBRegressor(**best_params)
xgb_final_model.fit(X_train, y_train)

# Extract and display feature importances
importance_df = pd.DataFrame({"Feature": X_train.columns, "Importance": xgb_final_model.feature_importances_})

# Sort by importance
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Print feature importance
print(importance_df)
