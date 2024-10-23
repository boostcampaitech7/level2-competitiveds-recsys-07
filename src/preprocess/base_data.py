import pandas as pd

# Raw 데이터 디렉토리 경로
data_path = "../../../data"

df_train = pd.read_csv(f"{data_path}/train.csv")
# df_test = pd.read_csv(f"{data_path}/test.csv")
df_school = pd.read_csv(f"{data_path}/schoolinfo.csv")
df_park = pd.read_csv(f"{data_path}/parkInfo.csv")
# df_ir = pd.read_csv(f"{data_path}/interestRate.csv")
df_subway = pd.read_csv(f"{data_path}/subwayInfo.csv")
# df_sample_submission = pd.read_csv(f"{data_path}/sample_submission.csv")

print("train shape: ", df_train.shape)
print("school shape: ", df_school.shape)
print("park shape: ", df_park.shape)
print("subway shape: ", df_subway.shape)


# train: 중복제거, age 음수 제거
df_train[df_train.age >= 0].to_csv("train.csv", index=False)

# 공원: 중복제거, 면적 0 이하 제거
df_park_pp = df_park.drop_duplicates()
df_park_pp = df_park_pp[df_park_pp.area > 0]
df_park_pp.reset_index(drop=True).to_csv("park.csv", index=False)

# 학교: 중복제거
df_school_pp = df_school.drop_duplicates()
df_school_pp.reset_index(drop=True).to_csv("school.csv", index=False)

# 지하철: 중복 제거
df_subway_pp = df_subway.drop_duplicates()
df_subway_pp.reset_index(drop=True).to_csv("subway.csv", index=False)
