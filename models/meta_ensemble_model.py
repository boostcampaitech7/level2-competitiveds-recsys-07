import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


def load_data(train_path: str, test_path: str):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X_train = train_data.drop(columns="deposit")
    y_train = train_data["deposit"]
    X_test = test_data.drop(columns="deposit")
    y_test = test_data["deposit"]
    return X_train, y_train, X_test, y_test


def load_model_params(file_path: list):
    params = {"lgb_params": {}, "rf_params": {}, "xgb_params": {}}
    for file in file_path:
        current_model = None
        with open(file, "r") as file:
            for line in file:
                line = line.strip()
                if line in params.keys():
                    current_model = params[line]
                elif current_model is not None and ": " in line:
                    key, value = line.split(": ")
                    try:
                        value = float(value) if "." in value else int(value)
                    except ValueError:
                        pass
                    current_model[key] = value
    return params


def objective(trial, X_train, y_train, params):
    base_models = [
        ("xgb", XGBRegressor(**params["xgb_params"])),
        ("lgb", lgb.LGBMRegressor(**params["lgb_params"])),
        ("rf", RandomForestRegressor(**params["rf_params"])),
    ]

    meta_model = Ridge(alpha=trial.suggest_float("alpha_ridge", 0.01, 10.0))

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
    )

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scores = []

    for train_idx, valid_idx in kfold.split(X_train):
        x_train_fold, x_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        stacking_model.fit(x_train_fold, y_train_fold)
        y_pred_valid = stacking_model.predict(x_valid_fold)
        mae = mean_absolute_error(y_valid_fold, y_pred_valid)
        mae_scores.append(mae)

    return np.mean(mae_scores)


def optimize_hyperparameters(X_train, y_train, params):
    seed = 42
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, X_train, y_train, params), n_trials=100)
    trial = study.best_trial

    print(f"Sampler is {study.sampler.__class__.__name__}")
    print(f"Best MAE: {trial.value}")
    print("Best hyperparameters: {}".format(trial.params))
    best_params = trial.params

    return best_params


def train_and_predict(X_train, y_train, X_test, best_params, params):
    meta_model = Ridge(alpha=best_params["alpha_ridge"])
    base_models = [
        ("xgb", XGBRegressor(**params["xgb_params"])),
        ("lgb", lgb.LGBMRegressor(**params["lgb_params"])),
        ("rf", RandomForestRegressor(**params["rf_params"])),
    ]

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
    )

    stacking_model.fit(X_train, y_train)
    y_test_pred = stacking_model.predict(X_test)

    return y_test_pred


def save_submission(y_test_pred, submission_path, output_path):
    sample_submission = pd.read_csv(submission_path)
    sample_submission["deposit"] = y_test_pred
    sample_submission.to_csv(output_path, index=False, encoding="utf-8-sig")


train_path = "../../data/train.csv"
test_path = "../../data/test.csv"
params_path = ["parameters_lgb.txt", "parameters_xgb.txt", "parameters_rf.txt"]
submission_path = "../../data/sample_submission.csv"
output_path = "output.csv"


X_train, y_train, X_test, y_test = load_data(train_path, test_path)

params = load_model_params(params_path)

best_params = optimize_hyperparameters(X_train, y_train, params)

y_test_pred = train_and_predict(X_train, y_train, X_test, best_params, params)

save_submission(y_test_pred, submission_path, output_path)
