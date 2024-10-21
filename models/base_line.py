import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

# 베이스라인 모델 LightGBM
han_data = pd.read_csv("/data/ephemeral/home/data/han/train.csv")
han_data_test = pd.read_csv("/data/ephemeral/home/data/han/test.csv")
ryu_data = pd.read_csv("/data/ephemeral/home/data/ryu/train.csv")
ryu_data_test = pd.read_csv("/data/ephemeral/home/data/ryu/test.csv")
yang_data = pd.read_csv("/data/ephemeral/home/data/yang/train.csv")
yang_data_test = pd.read_csv("/data/ephemeral/home/data/yang/test.csv")

sample_submission = pd.read_csv("/data/ephemeral/home/data/original/sample_submission.csv")

han_data = han_data.sort_values(by="index")
han_data = han_data.reset_index().drop(columns="level_0")
han_data_test = han_data_test.sort_values(by="index")
han_data_test = han_data_test.reset_index().drop(columns="level_0")

new_columns_train_ryu = [col for col in ryu_data.columns if col != "index" and col not in han_data.columns]

merge1_train_data = pd.merge(han_data, ryu_data[["index"] + new_columns_train_ryu], on="index", how="left")

new_columns_train_yang = [col for col in yang_data.columns if col != "index" and col not in merge1_train_data.columns]

merge2_train_data = pd.merge(merge1_train_data, yang_data[["index"] + new_columns_train_yang], on="index", how="left")

new_columns_test_ryu = [col for col in ryu_data_test.columns if col != "index" and col not in han_data_test.columns]

merge1_test_data = pd.merge(han_data_test, ryu_data_test[["index"] + new_columns_test_ryu], on="index", how="left")

new_columns_test_yang = [
    col for col in yang_data_test.columns if col != "index" and col not in merge1_test_data.columns
]

merge2_test_data = pd.merge(merge1_test_data, yang_data_test[["index"] + new_columns_test_yang], on="index", how="left")

train_data = merge2_train_data
test_data = merge2_test_data

# train 및 test 구분
X_train = train_data.drop(columns=["deposit", "deposit_mean"])
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
X_holdout = X_train[(X_train["contract_year_month"] >= holdout_start) & (X_train["contract_year_month"] <= holdout_end)]
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

lgb_model1 = LGBMRegressor(**params)
lgb_model1.fit(X_trainout, y_trainout)

# hold out 검증
lgb_holdout_pred = lgb_model1.predict(X_holdout)
lgb_holdout_mae = mean_absolute_error(y_holdout, lgb_holdout_pred)
print("Holdout 데이터셋 성능:")
print(f"LightGBM MAE: {lgb_holdout_mae:.2f}")


# 피쳐 임포턴스
feature_importance = lgb_model1.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

print(importance_df)

# Plotting feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("LightGBM Feature Importance")
plt.show()


# 제출
y_train = y_train.drop(columns="contract_year_month")
lgb_model = LGBMRegressor(**params)
lgb_model.fit(X_train, y_train)

lgb_test_pred = lgb_model.predict(X_test)
sample_submission["deposit"] = lgb_test_pred
sample_submission.to_csv("output.csv", index=False, encoding="utf-8-sig")
