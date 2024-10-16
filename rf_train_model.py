from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

# 함수화 함수 따로 만들기 (data_pre_processor로 넣기)
train_data = pd.read_csv("../../data/train.csv")
test_data = pd.read_csv("../../data/test.csv")


# train 및 test 구분
X_train = train_data.drop(columns="deposit")
y_train = train_data["deposit"]
X_test = test_data.drop(columns="deposit")
y_test = test_data["deposit"]


def rf_model_train(
    trial: Any, X_train: pd.DataFrame, y_train: pd.Series, cv: int
) -> float:

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["auto", "sqrt", "log2"]
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 42,
    }

    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    mae_scores = []

    for train_idx, valid_idx in kfold.split(X_train):
        x_train_fold, x_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        # Create Random Forest Regressor model
        rf_model = RandomForestRegressor(**params)

        # Train the model
        rf_model.fit(x_train_fold, y_train_fold)

        # Predict on validation set
        y_pred_valid = rf_model.predict(x_valid_fold)

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_valid_fold, y_pred_valid)
        mae_scores.append(mae)

    # Return the mean MAE across all folds
    return np.mean(mae_scores)


# Initialize and run the study
study = optuna.create_study(direction="minimize")  # Minimize MAE
study.optimize(
    lambda trial: rf_model_train(trial, X_train, y_train, cv=5), n_trials=100
)

# Output the results of the best trial
trial = study.best_trial
print(f"Sampler is {study.sampler.__class__.__name__}")
print("Best MAE: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))


# 피쳐 중요성 표시
best_params = trial.params
rf_model = RandomForestRegressor(**best_params)
rf_model.fit(X_train, y_train)

feature_importances = rf_model.feature_importances_
features = X_train.columns

# Print feature importances
importance_df = pd.DataFrame(
    {"Feature": features, "Importance": feature_importances}
).sort_values(by="Importance", ascending=False)

print("\nFeature Importances:")
print(importance_df)

# Save the best hyperparameters to parameters.txt
with open("parameters.txt", "w+") as f:
    for key, value in trial.params.items():
        f.write(f"{key}: {value}\n")
