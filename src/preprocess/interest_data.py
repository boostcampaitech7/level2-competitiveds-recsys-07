import os

import pandas as pd

path = "../../data"

train_data = pd.read_csv(os.path.join(path, "train.csv"))
test_data = pd.read_csv(os.path.join(path, "test.csv"))
sample_submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))
subwayinfo = pd.read_csv(os.path.join(path, "subwayinfo.csv"))
schoolinfo = pd.read_csv(os.path.join(path, "schoolinfo.csv"))
parkinfo = pd.read_csv(os.path.join(path, "parkinfo.csv"))
interestRate = pd.read_csv(os.path.join(path, "interestRate.csv"))

# train: 중복제거, age 음수 제거
train_data = (
    train_data[train_data.age >= 0].reset_index(drop=True).drop(columns="index")
)  # .to_csv("train.csv", index=False)

# 공원: 중복제거, 면적 0 이하 제거
parkinfo = parkinfo.drop_duplicates()
parkinfo = parkinfo[parkinfo.area > 0]
# parkinfo.reset_index(drop=True).to_csv("park.csv", index=False)

# 학교: 중복제거
schoolinfo = schoolinfo.drop_duplicates()
# schoolinfo.reset_index(drop=True).to_csv("school.csv", index=False)

# 지하철: 중복 제거
subwayinfo = subwayinfo.drop_duplicates()
# subwayinfo.reset_index(drop=True).to_csv("subway.csv", index=False)

train_data = train_data.sort_values("contract_year_month")
interestRate = interestRate.sort_values("year_month")
# train data: 201904 ~ 202312
# test data: 202401 ~ 202406
# interestRate: 201812 ~ 202405

interestRate["interest_rate_lag1"] = interestRate["interest_rate"].shift(1)
interestRate["interest_rate_lag2"] = interestRate["interest_rate"].shift(2)
interestRate["interest_rate_lag3"] = interestRate["interest_rate"].shift(3)
interestRate["interest_rate_lag4"] = interestRate["interest_rate"].shift(4)
interestRate["interest_rate_lag5"] = interestRate["interest_rate"].shift(5)

merged_data = pd.merge(train_data, interestRate, left_on="contract_year_month", right_on="year_month", how="left")
merged_data_test = pd.merge(test_data, interestRate, left_on="contract_year_month", right_on="year_month", how="left")

merged_data.drop("date", axis=1, inplace=True)
merged_data_test.drop("date", axis=1, inplace=True)
merged_data.drop("year_month", axis=1, inplace=True)
merged_data_test.drop("year_month", axis=1, inplace=True)

# 날짜별(deposit의 contract_year_month)로 평균값을 계산하여 새로운 컬럼 deposit_mean으로 추가
merged_data["deposit_mean"] = merged_data.groupby("contract_year_month")["deposit"].transform("mean")

merged_data["floor * area_m2"] = merged_data["area_m2"] * merged_data["floor"]
merged_data["area_m2 / age"] = merged_data["area_m2"] / (merged_data["age"] + 1)
merged_data["built_year * area_m2"] = merged_data["built_year"] * merged_data["area_m2"]

merged_data_test["floor * area_m2"] = merged_data_test["area_m2"] * merged_data_test["floor"]
merged_data_test["area_m2 / age"] = merged_data_test["area_m2"] / (merged_data_test["age"] + 1)
merged_data_test["built_year * area_m2"] = merged_data_test["built_year"] * merged_data_test["area_m2"]

merged_data["area_m2 * interest_rate_lag1"] = merged_data["area_m2"] * merged_data["interest_rate_lag1"]
merged_data["area_m2 * interest_rate_lag2"] = merged_data["area_m2"] * merged_data["interest_rate_lag2"]
merged_data["area_m2 * interest_rate_lag3"] = merged_data["area_m2"] * merged_data["interest_rate_lag3"]
merged_data["area_m2 * interest_rate_lag4"] = merged_data["area_m2"] * merged_data["interest_rate_lag4"]
merged_data["area_m2 * interest_rate_lag5"] = merged_data["area_m2"] * merged_data["interest_rate_lag5"]

merged_data_test["area_m2 * interest_rate_lag1"] = merged_data_test["area_m2"] * merged_data_test["interest_rate_lag1"]
merged_data_test["area_m2 * interest_rate_lag2"] = merged_data_test["area_m2"] * merged_data_test["interest_rate_lag2"]
merged_data_test["area_m2 * interest_rate_lag3"] = merged_data_test["area_m2"] * merged_data_test["interest_rate_lag3"]
merged_data_test["area_m2 * interest_rate_lag4"] = merged_data_test["area_m2"] * merged_data_test["interest_rate_lag4"]
merged_data_test["area_m2 * interest_rate_lag5"] = merged_data_test["area_m2"] * merged_data_test["interest_rate_lag5"]

# CSV 파일로 저장
merged_data.to_csv("train.csv", index=False)
merged_data_test.to_csv("test.csv", index=False)
