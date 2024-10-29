import os

import pandas as pd

path = "./data_test"

train_data = pd.read_csv(os.path.join(path, "train.csv"))
test_data = pd.read_csv(os.path.join(path, "test.csv"))
# sample_submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
# subwayinfo = pd.read_csv(os.path.join(path, 'subwayinfo.csv'))
# schoolinfo = pd.read_csv(os.path.join(path, 'schoolinfo.csv'))
# parkinfo = pd.read_csv(os.path.join(path, 'parkinfo.csv'))
interestRate = pd.read_csv(os.path.join(path, "interestRate.csv"))

# train: 중복제거, age 음수 제거
train_data = train_data[train_data.age >= 0]  # .reset_index(drop=True) #.to_csv("train.csv", index=False)
# df_train[df_train.age >= 0]

# # 공원: 중복제거, 면적 0 이하 제거
# parkinfo = parkinfo.drop_duplicates()
# parkinfo = parkinfo[parkinfo.area > 0]
# # parkinfo.reset_index(drop=True).to_csv("park.csv", index=False)

# # 학교: 중복제거
# schoolinfo = schoolinfo.drop_duplicates()
# # schoolinfo.reset_index(drop=True).to_csv("school.csv", index=False)

# # 지하철: 중복 제거
# subwayinfo = subwayinfo.drop_duplicates()
# # subwayinfo.reset_index(drop=True).to_csv("subway.csv", index=False)

print("train shape: ", train_data.shape)
# print("school shape: ", schoolinfo.shape)
# print("park shape: ", parkinfo.shape)
# print("subway shape: ", subwayinfo.shape)
print("interestRate shape: ", interestRate.shape)

train_data = train_data.sort_values("contract_year_month")
interestRate = interestRate.sort_values("year_month")

train_data = train_data.sort_index()

# 면적 범주화 적용
train_data["area_m2_category"] = (train_data["area_m2"] // 50) * 50
train_data["area_m2_category"] = (
    train_data["area_m2_category"].astype(str) + "~" + (train_data["area_m2_category"] + 49).astype(str)
)

# 연도와 월 추출
train_data["contract_year_month"] = train_data["contract_year_month"].astype(str)
train_data["year"] = train_data["contract_year_month"].str[:4].astype(int)
train_data["month"] = train_data["contract_year_month"].str[4:6].astype(int)
train_data["contract_year_month"] = train_data["contract_year_month"].astype(int)
# 거래일을 10일 단위로 범주화
train_data["contract_day_category"] = train_data["contract_day"].apply(
    lambda x: 10 if 1 <= x <= 10 else (20 if 11 <= x <= 20 else 30)
)

# 거래 가격을 1억 단위로 범주화
scale = 10000
train_data["deposit_category"] = (train_data["deposit"] // scale) * scale
train_data["deposit_category"] = (
    train_data["deposit_category"].astype(str) + "~" + (train_data["deposit_category"] + scale - 1).astype(str)
)

# 타입 변환
train_data["area_m2_category"] = train_data["area_m2_category"].astype("category")
train_data["year"] = train_data["year"].astype("category")
train_data["month"] = train_data["month"].astype("category")
train_data["contract_day_category"] = train_data["contract_day_category"].astype("category")
train_data["deposit_category"] = train_data["deposit_category"].astype("category")

# 시차 적용된 금리 변화율도 계산
interestRate["interest_rate_diff"] = interestRate["interest_rate"].diff()

interestRate["interest_rate_lag1"] = interestRate["interest_rate"].shift(1)
interestRate["interest_rate_lag3"] = interestRate["interest_rate"].shift(3)
interestRate["interest_rate_lag5"] = interestRate["interest_rate"].shift(5)

interestRate["interest_rate_lag1_diff"] = interestRate["interest_rate_diff"].shift(1)
# interestRate['interest_rate_lag2_diff'] = interestRate['interest_rate_diff'].shift(2)
interestRate["interest_rate_lag3_diff"] = interestRate["interest_rate_diff"].shift(3)
# interestRate['interest_rate_lag4_diff'] = interestRate['interest_rate_diff'].shift(4)
interestRate["interest_rate_lag5_diff"] = interestRate["interest_rate_diff"].shift(5)

merged_data = pd.merge(train_data, interestRate, left_on="contract_year_month", right_on="year_month", how="left")
merged_data_test = pd.merge(test_data, interestRate, left_on="contract_year_month", right_on="year_month", how="left")

merged_data.drop("date", axis=1, inplace=True)
merged_data_test.drop("date", axis=1, inplace=True)
merged_data.drop("year_month", axis=1, inplace=True)
merged_data_test.drop("year_month", axis=1, inplace=True)

# 날짜별(deposit의 contract_year_month)로 평균값을 계산하여 새로운 컬럼 deposit_mean으로 추가

merged_data["floor * area_m2"] = merged_data["area_m2"] * merged_data["floor"]
merged_data["built_year * area_m2"] = merged_data["built_year"] * merged_data["area_m2"]

merged_data_test["floor * area_m2"] = merged_data_test["area_m2"] * merged_data_test["floor"]
merged_data_test["built_year * area_m2"] = merged_data_test["built_year"] * merged_data_test["area_m2"]


merged_data("interest_rate", axis=1, inplace=True)
merged_data_test("interest_rate", axis=1, inplace=True)
merged_data("interest_rate_diff", axis=1, inplace=True)
merged_data_test("interest_rate_diff", axis=1, inplace=True)


# CSV 파일로 저장
merged_data.to_csv("train.csv", index=False)
merged_data_test.to_csv("test.csv", index=False)
