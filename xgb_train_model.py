from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

## 함수화 함수 따로 만들기 (data_pre_processor로 넣기)
train_data = pd.read_csv("../../data/train.csv")
test_data = pd.read_csv("../../data/test.csv")


# train 및 test 구분
X_train = train_data.drop(columns="deposit")
y_train = train_data["deposit"]
X_test = test_data.drop(columns="deposit")
y_test = test_data["deposit"]

# Model(optuna를 이용)


def xgb_model_train(
    trial: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> float:

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "random_state": 42,
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
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
            early_stopping_rounds=50,
            verbose=False,
        )

        # Predict on validation set
        y_pred_valid = xgb_model.predict(x_valid_fold)

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_valid_fold, y_pred_valid)
        mae_scores.append(mae)

    # Return the mean MAE across all folds
    return np.mean(mae_scores)


# Initialize and run the study
study = optuna.create_study(direction="minimize")  # Minimize MAE
study.optimize(lambda trial: xgb_model_train(trial, X_train, y_train), n_trials=100)

# Output the results of the best trial
trial = study.best_trial
print(f"Sampler is {study.sampler.__class__.__name__}")
print("Best MAE: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))


# 최적의 파라미터들을 parameters.txt에 넣음.
with open("parameters.txt", "w+") as f:
    for key, value in trial.params.items():
        f.write(f"{key}: {value}\n")


# 추가로 importance feature보기.
best_params = trial.params
xgb_final_model = XGBRegressor(**best_params)
xgb_final_model.fit(X_train, y_train)

# Extract and display feature importances
importance_df = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": xgb_final_model.feature_importances_}
)

# Sort by importance
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Print feature importance
print(importance_df)
