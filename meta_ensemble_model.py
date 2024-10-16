import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

# 데이타 불러오기
train_data = pd.read_csv("../../data/train.csv")
test_data = pd.read_csv("../../data/test.csv")

X_train = train_data.drop(columns="deposit")
y_train = train_data["deposit"]
X_test = test_data.drop(columns="deposit")
y_test = test_data["deposit"]
# 파라미터들 받아오기

lgb_params = {}
rf_params = {}
xgb_params = {}
with open("parameters.txt", "r") as file:
    current_model = None
    for line in file:
        line = line.strip()

        # Check if the line indicates a new model's parameters
        if line == "lgb_params :":
            current_model = lgb_params
        elif line == "rf_params :":
            current_model = rf_params
        elif line == "xgb_params :":
            current_model = xgb_params
        elif current_model is not None and ": " in line:
            # Split the key and value, and store them in the current model's dictionary
            key, value = line.split(": ")
            # Try to convert the value to int or float if possible, else keep as string
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass  # If it can't be converted, keep it as a string
            current_model[key] = value


# meta (ridge) 를 이용하여 한번더 학습
def objective(trial, X_train, y_train):
    # Define the base models (first layer)
    base_models = [
        ("xgb", XGBRegressor(**xgb_params)),
        ("lgb", lgb.LGBMRegressor(**lgb_params)),
        ("rf", RandomForestRegressor(**rf_params)),
    ]

    # Define the meta-model (second layer)
    meta_model = Ridge(alpha=trial.suggest_float("alpha_ridge", 0.01, 10.0))

    # Create the StackingRegressor ensemble
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,  # Ridge is the meta-model
        cv=5,  # Cross-validation for training meta-model
    )

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scores = []

    for train_idx, valid_idx in kfold.split(X_train):
        x_train_fold, x_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        # Train the model
        stacking_model.fit(x_train_fold, y_train_fold)

        # Predict on validation set
        y_pred_valid = stacking_model.predict(x_valid_fold)

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_valid_fold, y_pred_valid)
        mae_scores.append(mae)
    return np.mean(mae_scores)


study = optuna.create_study(direction="minimize")  # Minimize MAE
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100)

# Output the results of the best trial
trial = study.best_trial
print(f"Sampler is {study.sampler.__class__.__name__}")
print("Best MAE: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

# 베스트 하이퍼파라미터 설정
best_params = trial.params
meta_model = Ridge(alpha=best_params)
base_models = [
    ("xgb", XGBRegressor(**xgb_params)),
    ("lgb", lgb.LGBMRegressor(**lgb_params)),
    ("rf", RandomForestRegressor(**rf_params)),
]
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,  # Ridge is the meta-model
    cv=5,  # Cross-validation for training meta-model
)
stacking_model.fit(X_train, y_train)

# 예측
y_test_pred = stacking_model.predict(y_test)
sample_submission = pd.read_csv("../../data/sample_submission.csv")
sample_submission["deposit"] = y_test_pred
sample_submission.to_csv("output.csv", index=False, encoding="utf-8-sig")
