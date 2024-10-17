import pandas as pd
from lat_lng_range import get_extended_lat_lng_range

# Raw 데이터 디렉토리 경로
data_path = "../../../data"

# lat_lng_range.py에서 정의한 함수 호출하여 10km 확장된 범위 계산
lat_min_10km, lat_max_10km, lon_min_10km, lon_max_10km = get_extended_lat_lng_range(
    f"{data_path}/train.csv", f"{data_path}/test.csv"
)

# 학교 정보와 공원 정보 CSV 불러오기
school_df = pd.read_csv(f"{data_path}/schoolinfo.csv")
park_df = pd.read_csv(f"{data_path}/parkInfo.csv")

# 범위 내에 있는 학교 데이터만 필터링
filtered_school_df = school_df[
    (school_df["latitude"] >= lat_min_10km)
    & (school_df["latitude"] <= lat_max_10km)
    & (school_df["longitude"] >= lon_min_10km)
    & (school_df["longitude"] <= lon_max_10km)
]

# 범위 내에 있는 공원 데이터만 필터링
filtered_park_df = park_df[
    (park_df["latitude"] >= lat_min_10km)
    & (park_df["latitude"] <= lat_max_10km)
    & (park_df["longitude"] >= lon_min_10km)
    & (park_df["longitude"] <= lon_max_10km)
]

# 필터링된 결과를 새로운 CSV 파일로 저장
filtered_school_df.to_csv(f"{data_path}/filtered_schoolInfo.csv", index=False)
filtered_park_df.to_csv(f"{data_path}/filtered_parkInfo.csv", index=False)

print("필터링된 CSV 파일 저장 완료!")
