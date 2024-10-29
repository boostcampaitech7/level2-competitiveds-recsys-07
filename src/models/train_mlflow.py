import sys

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


def main(train_data, test_data, sample_submission):
    # Set the MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:30004")

    # Set the experiment name
    mlflow.set_experiment("LightGBM Regression")

    # train 및 test 구분
    X_train = train_data.drop(columns=["deposit"])
    y_train = train_data[["contract_year_month", "deposit"]]
    X_test = test_data.copy()

    if "index" in X_train.columns:
        X_train = X_train.drop(columns="index")
    if "index" in X_test.columns:
        X_test = X_test.drop(columns="index")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and Test columns do not match.")
        sys.exit()

    # validation 파트

    # holdout 구분
    holdout_start = 202307
    holdout_end = 202312
    X_holdout = X_train[
        (X_train["contract_year_month"] >= holdout_start) & (X_train["contract_year_month"] <= holdout_end)
    ]
    X_trainout = X_train[
        ~((X_train["contract_year_month"] >= holdout_start) & (X_train["contract_year_month"] <= holdout_end))
    ]
    y_holdout = y_train[
        (y_train["contract_year_month"] >= holdout_start) & (y_train["contract_year_month"] <= holdout_end)
    ].drop(columns="contract_year_month")
    y_trainout = y_train[
        ~(y_train["contract_year_month"] >= holdout_start) & (y_train["contract_year_month"] <= holdout_end)
    ].drop(columns="contract_year_month")

    params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "min_child_weight": 0.001,
        "subsample": 1.0,
        "subsample_freq": 0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": 42,
        "importance_type": "split",
        "verbose": -1,
    }
    # 훈련
    # seed ensemble 활용가능 여기서
    # random_seed=[42,32,25,12,9] -> prediction 에 각각 1/5

    # Start MLflow run
    run_name = f"{'_'.join(dname_lst)}"
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(params)

        lgb_model = LGBMRegressor(**params)
        lgb_model.fit(X_trainout, y_trainout)

        # hold out 검증
        lgb_holdout_pred = lgb_model.predict(X_holdout)
        lgb_holdout_mae = mean_absolute_error(y_holdout, lgb_holdout_pred)
        print("Holdout 데이터셋 성능:")
        print(f"LightGBM MAE: {lgb_holdout_mae:.2f}")

        # Log metrics
        mlflow.log_metric("holdout_mae", lgb_holdout_mae)

        # 피쳐 임포턴스
        feature_importance = lgb_model.feature_importances_
        feature_names = X_train.columns
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        print(importance_df)

        # Plotting feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=importance_df)
        plt.title("LightGBM Feature Importance")
        # plt.show()

        # Save plot as artifact
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")

        # 제출
        y_train = y_train.drop(columns="contract_year_month")
        lgb_model = LGBMRegressor(**params)
        lgb_model.fit(X_train, y_train)

        # Log model
        mlflow.lightgbm.log_model(lgb_model, "model")

        lgb_test_pred = lgb_model.predict(X_test)
        sample_submission["deposit"] = lgb_test_pred
        sample_submission.to_csv("output.csv", index=False, encoding="utf-8-sig")

        # Save predictions
        mlflow.log_artifact("output.csv")


def load_data(dname_lst):
    root_dir = "/data/ephemeral/home/yang/level2-competitiveds-recsys-07-main/data"

    sample_submission = pd.read_csv(f"{root_dir}/base/sample_submission.csv")

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for dname in tqdm(dname_lst):
        train = pd.read_csv(f"{root_dir}/{dname}/train.csv")
        test = pd.read_csv(f"{root_dir}/{dname}/test.csv")
        train_data = train_data.combine_first(train)
        test_data = test_data.combine_first(test)

    return train_data, test_data, sample_submission


if __name__ == "__main__":

    lst = [
        ["facility", "region median center"],
        ["facility", "region median center", "cluster_reg_lat_lon_5"],
        ["facility", "region median center", "cluster_reg_lat_lon_5", "Interest Rate_diff"],
    ]

    for dname_lst in tqdm(lst):
        train_data, test_data, sample_submission = load_data(dname_lst)

        main(train_data, test_data, sample_submission)
